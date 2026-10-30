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
from src.ingestion.ghcnh_parser import StationMetadata
from src.terrain.dem_pipeline import DEMPipeline, StationTerrainFeatures


class DEMFlowRoutingAlertEngine:
    """Method v1: Hydro-topographic alert propagation following D8 flow descent."""

    def __init__(
        self,
        dem_pipeline: DEMPipeline,
        elevated_threshold_m: float,
        trigger_risk_threshold: float,
        flow_decay_alpha: float,
        accumulation_boost_beta: float,
        min_alert_strength: float,
        wave_speed_km_h: float,
        max_reach_km: float,
        max_steps: int,
        min_lead_time_minutes: float,
        max_lead_time_minutes: float,
        synthetic: bool = False,
    ):
        self.dem_pipeline = dem_pipeline
        self.elevated_threshold_m = elevated_threshold_m
        self.trigger_risk_threshold = trigger_risk_threshold
        self.flow_decay_alpha = flow_decay_alpha
        self.accumulation_boost_beta = accumulation_boost_beta
        self.min_alert_strength = min_alert_strength
        self.wave_speed_km_h = wave_speed_km_h
        self.max_reach_km = max_reach_km
        self.max_steps = max_steps
        self.min_lead_time_minutes = min_lead_time_minutes
        self.max_lead_time_minutes = max_lead_time_minutes
        self.synthetic = synthetic

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
        routing = self.dem_pipeline.trace_station_to_station(
            trigger_station, candidate_station, max_steps=self.max_steps
        )
        if routing is None or routing["distance_km"] > self.max_reach_km:
            return False, 0.0
        return True, round(routing["distance_km"], 2)

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

            routing = self.dem_pipeline.trace_station_to_station(
                trigger_station, candidate, max_steps=self.max_steps
            )

            if routing is None or routing["distance_km"] > self.max_reach_km:
                continue  # Skip stations not downstream in the watershed!
            channel_dist_km = routing["distance_km"]

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
            if not (
                self.min_lead_time_minutes
                <= lead_time_minutes
                <= self.max_lead_time_minutes
            ):
                continue

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
                flow_path_steps=routing["flow_path_steps"],
                in_watershed=True,
                synthetic=self.synthetic,
            )
            alerts.append(alert)
            if logger:
                logger.record_alert(alert)

        return alerts
