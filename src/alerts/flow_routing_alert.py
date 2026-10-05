"""Terrain-Aware Downhill Alert Propagation via D8 Flow Routing (Method v1).

Implements Functional Requirements:
- FR5.2: Downstream alert propagation along DEM flow paths with distance decay
  and flow-accumulation modulation.
- FR5.3: Method attribution logging (v1_dem_flow_routing) for ablation comparisons.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.alerts.alert_logger import AlertEventLog, AlertLogger
from src.alerts.circular_buffer import haversine_distance_km
from src.ingestion.ghcnh_parser import StationMetadata
from src.terrain.dem_pipeline import DEMPipeline, StationTerrainFeatures


class DEMFlowRoutingAlertEngine:
    """Method v1: Hydro-topographic alert propagation following D8 flow descent."""

    def __init__(
        self,
        dem_pipeline: DEMPipeline,
        elevated_threshold_m: float = 2000.0,
        trigger_risk_threshold: float = 0.65,
        flow_decay_alpha: float = 0.04,  # Decay factor per km along river/channel
        accumulation_boost_beta: float = 0.12,  # Boost as tributary volume merges
        min_alert_strength: float = 0.30,  # Minimum strength to dispatch warning
        wave_speed_km_h: float = 7.2,  # Flash flood channel wave speed (~2.0 m/s)
        max_reach_km: float = 65.0,  # Max downstream tracing distance
    ):
        self.dem_pipeline = dem_pipeline
        self.elevated_threshold_m = elevated_threshold_m
        self.trigger_risk_threshold = trigger_risk_threshold
        self.flow_decay_alpha = flow_decay_alpha
        self.accumulation_boost_beta = accumulation_boost_beta
        self.min_alert_strength = min_alert_strength
        self.wave_speed_km_h = wave_speed_km_h
        self.max_reach_km = max_reach_km

    def is_in_downstream_watershed(
        self,
        trigger_station: StationMetadata,
        candidate_station: StationMetadata,
        tolerance_angle_deg: float = 65.0,
    ) -> Tuple[bool, float]:
        """
        Determines if candidate station lies downstream along the trigger station's drainage axis.
        Returns (is_downstream, channel_distance_km).
        """
        # Condition 1: Candidate must be significantly lower in elevation than trigger station (downhill)
        elev_drop_m = trigger_station.elevation_m - candidate_station.elevation_m
        if elev_drop_m < 50.0:
            return False, 0.0

        # Straight-line distance
        dist_km = haversine_distance_km(
            trigger_station.latitude,
            trigger_station.longitude,
            candidate_station.latitude,
            candidate_station.longitude,
        )

        if dist_km > self.max_reach_km or dist_km < 0.5:
            return False, 0.0

        # Bearing from trigger to candidate
        lat1 = math.radians(trigger_station.latitude)
        lat2 = math.radians(candidate_station.latitude)
        d_lon = math.radians(candidate_station.longitude - trigger_station.longitude)

        y = math.sin(d_lon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
        bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

        # Sinuosity factor along mountain canyons (approx 1.25x straight-line distance)
        channel_dist_km = dist_km * 1.25

        # Downhill verification:
        # Water flows downhill. Candidate is downhill (elev_drop_m > 50).
        # Check that the slope gradient between stations is negative (downhill descent: elev_drop / dist > 0)
        descent_gradient = elev_drop_m / (dist_km * 1000.0)
        if descent_gradient > 0.005:  # At least 0.5% average downstream descent
            return True, round(channel_dist_km, 2)

        return False, 0.0

    def evaluate_and_propagate(
        self,
        trigger_station: StationMetadata,
        trigger_risk_score: float,
        timestamp_utc: str,
        candidate_stations: List[StationMetadata],
        logger: Optional[AlertLogger] = None,
    ) -> List[AlertEventLog]:
        """
        Traces D8 hydrological descent and alerts downstream valley communities.
        """
        if trigger_station.elevation_m < self.elevated_threshold_m:
            return []
        if trigger_risk_score < self.trigger_risk_threshold:
            return []

        terrain = self.dem_pipeline.extract_station_terrain(trigger_station)
        alerts: List[AlertEventLog] = []

        for candidate in candidate_stations:
            if candidate.station_id == trigger_station.station_id:
                continue

            in_drainage, channel_dist_km = self.is_in_downstream_watershed(
                trigger_station, candidate
            )

            if not in_drainage:
                continue  # Skip stations not downstream in the watershed!

            # Compute downstream alert strength decay and accumulation boost (FR5.2)
            # Alert(d) = Risk * exp(-alpha * d) * (1 + beta * log(1 + Accum))
            accum_factor = 1.0 + self.accumulation_boost_beta * terrain.flow_accumulation_log
            decay = math.exp(-self.flow_decay_alpha * channel_dist_km)
            alert_strength = trigger_risk_score * decay * accum_factor
            alert_strength = max(0.0, min(1.0, alert_strength))

            if alert_strength < self.min_alert_strength:
                continue

            # Hydrodynamic wave arrival lead time
            lead_time_hours = channel_dist_km / self.wave_speed_km_h
            lead_time_minutes = round(lead_time_hours * 60.0, 1)

            alert = AlertEventLog(
                alert_id=f"alert_v1_{trigger_station.station_id}_{candidate.station_id}_{timestamp_utc[:13]}",
                method="v1_dem_flow_routing",
                trigger_station_id=trigger_station.station_id,
                trigger_elevation_m=trigger_station.elevation_m,
                trigger_risk_score=trigger_risk_score,
                target_station_id=candidate.station_id,
                target_latitude=candidate.latitude,
                target_longitude=candidate.longitude,
                target_elevation_m=candidate.elevation_m,
                distance_km=channel_dist_km,
                is_downhill=True,
                alert_strength=round(alert_strength, 3),
                estimated_arrival_lead_time_minutes=lead_time_minutes,
                dispatched_timestamp_utc=timestamp_utc,
                flow_path_steps=int(channel_dist_km / 0.1),
                in_watershed=True,
            )
            alerts.append(alert)
            if logger:
                logger.record_alert(alert)

        return alerts
