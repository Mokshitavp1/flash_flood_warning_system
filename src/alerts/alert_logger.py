"""Alert Logger and Event Audit Trail.

Implements Functional Requirements:
- FR5.3: System shall log which of the two alert methods (v0 vs v1) triggered
  any given alert, to support the ablation study.
- Records source station, recipient station/coordinates, alert strength,
  lead time (minutes), distance (km), and terrain flow routing metadata.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AlertEventLog:
    """Detailed record of an alert dispatched to a downstream or nearby community."""
    alert_id: str
    method: str  # "v0_circular_buffer" or "v1_dem_flow_routing"
    trigger_station_id: str
    trigger_elevation_m: float
    trigger_risk_score: float
    target_station_id: Optional[str]
    target_latitude: float
    target_longitude: float
    target_elevation_m: float
    distance_km: float
    is_downhill: bool
    alert_strength: float  # Decayed alert intensity [0, 1]
    estimated_arrival_lead_time_minutes: float
    dispatched_timestamp_utc: str
    flow_path_steps: int = 0
    in_watershed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlertLogger:
    """Manages recording and export of alert event audit logs."""

    def __init__(self):
        self.logs: List[AlertEventLog] = []

    def record_alert(self, event: AlertEventLog) -> None:
        self.logs.append(event)

    def get_summary_by_method(self) -> Dict[str, Any]:
        """Summarizes alert statistics partitioned by method v0 vs v1."""
        v0_alerts = [a for a in self.logs if a.method == "v0_circular_buffer"]
        v1_alerts = [a for a in self.logs if a.method == "v1_dem_flow_routing"]

        v0_downhill = sum(1 for a in v0_alerts if a.is_downhill)
        v0_uphill = sum(1 for a in v0_alerts if not a.is_downhill)

        v1_downhill = sum(1 for a in v1_alerts if a.is_downhill)
        v1_uphill = sum(1 for a in v1_alerts if not a.is_downhill)

        return {
            "total_alerts": len(self.logs),
            "v0_circular_buffer": {
                "total": len(v0_alerts),
                "downhill_alerts": v0_downhill,
                "uphill_unnecessary_alerts": v0_uphill,
                "avg_distance_km": round(sum(a.distance_km for a in v0_alerts) / max(1, len(v0_alerts)), 2),
            },
            "v1_dem_flow_routing": {
                "total": len(v1_alerts),
                "downhill_alerts": v1_downhill,
                "uphill_unnecessary_alerts": v1_uphill,
                "avg_distance_km": round(sum(a.distance_km for a in v1_alerts) / max(1, len(v1_alerts)), 2),
                "avg_lead_time_min": round(
                    sum(a.estimated_arrival_lead_time_minutes for a in v1_alerts) / max(1, len(v1_alerts)), 1
                ),
            },
        }

    def to_csv(self, file_path: str | Path) -> None:
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "alert_id",
            "method",
            "trigger_station_id",
            "trigger_elevation_m",
            "trigger_risk_score",
            "target_station_id",
            "target_latitude",
            "target_longitude",
            "target_elevation_m",
            "distance_km",
            "is_downhill",
            "alert_strength",
            "estimated_arrival_lead_time_minutes",
            "dispatched_timestamp_utc",
            "flow_path_steps",
            "in_watershed",
        ]
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for log in self.logs:
                writer.writerow(log.to_dict())

    def to_json(self, file_path: str | Path) -> None:
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump([l.to_dict() for l in self.logs], f, indent=2)
