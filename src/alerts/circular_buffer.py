"""Baseline Circular-Buffer Alert Propagation (Method v0).

Implements Functional Requirements:
- FR5.1: Baseline circular-buffer alert: when a high-elevation station's risk score
  exceeds a threshold, alert all stations/locations within a fixed radius.
- FR5.3: Method attribution logging for ablation study.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.alerts.alert_logger import AlertEventLog, AlertLogger
from src.ingestion.ghcnh_parser import StationMetadata


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two coordinates in kilometers."""
    r_earth = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth * c


class CircularBufferAlertEngine:
    """Baseline v0: Flat circular buffer alert propagation."""

    def __init__(
        self,
        radius_km: float = 25.0,
        elevated_threshold_m: float = 2000.0,
        trigger_risk_threshold: float = 0.65,
    ):
        self.radius_km = radius_km
        self.elevated_threshold_m = elevated_threshold_m
        self.trigger_risk_threshold = trigger_risk_threshold

    def evaluate_and_propagate(
        self,
        trigger_station: StationMetadata,
        trigger_risk_score: float,
        timestamp_utc: str,
        candidate_stations: List[StationMetadata],
        logger: Optional[AlertLogger] = None,
    ) -> List[AlertEventLog]:
        """
        If trigger station is elevated and risk exceeds threshold,
        propagates alerts to all candidate stations within radius_km.
        """
        # Check elevation and risk conditions
        if trigger_station.elevation_m < self.elevated_threshold_m:
            return []
        if trigger_risk_score < self.trigger_risk_threshold:
            return []

        alerts: List[AlertEventLog] = []

        for candidate in candidate_stations:
            if candidate.station_id == trigger_station.station_id:
                continue

            dist_km = haversine_distance_km(
                trigger_station.latitude,
                trigger_station.longitude,
                candidate.latitude,
                candidate.longitude,
            )

            if dist_km <= self.radius_km:
                is_downhill = candidate.elevation_m < trigger_station.elevation_m
                # Naive distance decay
                decayed_strength = trigger_risk_score * max(0.2, 1.0 - (dist_km / self.radius_km))
                # Naive lead time assuming 10 km/h flood propagation
                est_lead_min = round((dist_km / 10.0) * 60.0, 1)

                alert = AlertEventLog(
                    alert_id=f"alert_v0_{trigger_station.station_id}_{candidate.station_id}_{timestamp_utc[:13]}",
                    method="v0_circular_buffer",
                    trigger_station_id=trigger_station.station_id,
                    trigger_elevation_m=trigger_station.elevation_m,
                    trigger_risk_score=trigger_risk_score,
                    target_station_id=candidate.station_id,
                    target_latitude=candidate.latitude,
                    target_longitude=candidate.longitude,
                    target_elevation_m=candidate.elevation_m,
                    distance_km=round(dist_km, 2),
                    is_downhill=is_downhill,
                    alert_strength=round(decayed_strength, 3),
                    estimated_arrival_lead_time_minutes=est_lead_min,
                    dispatched_timestamp_utc=timestamp_utc,
                    flow_path_steps=0,
                    in_watershed=True if is_downhill else False,
                )
                alerts.append(alert)
                if logger:
                    logger.record_alert(alert)

        return alerts
