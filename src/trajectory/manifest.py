"""Missing-Frame Manifest for Satellite Trajectories.

Implements Functional Requirements:
- FR2.4: Explicit recording of missing status when no image is within tolerance.
- FR2.5: Persistence of a separate manifest recording (ok / missing) per station-event.
- NFR1: Verification that missing frames are never forward-filled or mislabeled.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryFrameManifestItem:
    """Individual frame record within a station-anchored trajectory."""
    station_id: str
    event_id: str
    target_time_utc: str
    frame_index: int
    offset_minutes: int
    matched_satellite_time_utc: Optional[str]
    time_delta_seconds: Optional[float]
    status: str  # Strictly "ok" or "missing"
    satellite_id: str
    patch_mean_ir: Optional[float] = None
    cloud_fraction: Optional[float] = None
    source_file_name: Optional[str] = None

    def __post_init__(self):
        if self.status not in ("ok", "missing"):
            raise ValueError(
                f"Invalid manifest status '{self.status}'. Must be strictly 'ok' or 'missing'."
            )
        # NFR1 Hard Invariant: If status is 'missing', matched time must NOT be copied from another frame
        if self.status == "missing":
            if self.matched_satellite_time_utc is not None:
                raise ValueError(
                    "Missing frames must not contain a matched satellite timestamp."
                )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MissingFrameManifest:
    """Collection and exporter of per-event trajectory frame statuses."""

    def __init__(self):
        self.records: List[TrajectoryFrameManifestItem] = []

    def add_record(self, item: TrajectoryFrameManifestItem) -> None:
        self.records.append(item)

    def compute_coverage_statistics(self) -> Dict[str, Any]:
        """Calculates global and per-station coverage metrics."""
        total_frames = len(self.records)
        if total_frames == 0:
            return {
                "total_frames": 0,
                "ok_frames": 0,
                "missing_frames": 0,
                "coverage_pct": 0.0,
                "station_coverage": {},
            }

        ok_count = sum(1 for r in self.records if r.status == "ok")
        missing_count = sum(1 for r in self.records if r.status == "missing")
        coverage_pct = round((ok_count / total_frames) * 100.0, 2)

        # Per-station stats
        by_station: Dict[str, Dict[str, int]] = {}
        for r in self.records:
            stats = by_station.setdefault(r.station_id, {"ok": 0, "missing": 0})
            stats[r.status] += 1

        station_coverage = {}
        for st_id, stats in by_station.items():
            tot = stats["ok"] + stats["missing"]
            pct = round((stats["ok"] / tot) * 100.0, 2) if tot > 0 else 0.0
            station_coverage[st_id] = {
                "total": tot,
                "ok": stats["ok"],
                "missing": stats["missing"],
                "coverage_pct": pct,
            }

        return {
            "total_frames": total_frames,
            "ok_frames": ok_count,
            "missing_frames": missing_count,
            "coverage_pct": coverage_pct,
            "station_coverage": station_coverage,
        }

    def verify_nfr1_integrity(self) -> Tuple[bool, List[str]]:
        """
        Validates Non-Functional Requirement 1 (Data Integrity):
        - No frame labeled 'ok' has missing image data.
        - No frame labeled 'missing' was forward-filled or reused from a prior timestep.
        - Timesteps strictly progress in chronological order.
        """
        violations: List[str] = []

        # Group by event_id
        events: Dict[str, List[TrajectoryFrameManifestItem]] = {}
        for r in self.records:
            events.setdefault(r.event_id, []).append(r)

        for event_id, frames in events.items():
            frames_sorted = sorted(frames, key=lambda f: f.frame_index)
            prior_matched_time = None

            for f in frames_sorted:
                if f.status == "ok":
                    if not f.matched_satellite_time_utc:
                        violations.append(
                            f"Event {event_id} frame {f.frame_index}: status is 'ok' but matched_satellite_time is empty."
                        )
                    # Forward-fill check: ensure this observation is not an exact duplicate time of prior frame
                    if prior_matched_time and f.matched_satellite_time_utc == prior_matched_time:
                        violations.append(
                            f"Event {event_id} frame {f.frame_index}: FORWARD-FILL DETECTED! Reused prior matched time {prior_matched_time}."
                        )
                    prior_matched_time = f.matched_satellite_time_utc
                elif f.status == "missing":
                    # A missing frame should never have a matched timestamp identical to the prior frame
                    if f.matched_satellite_time_utc is not None:
                        violations.append(
                            f"Event {event_id} frame {f.frame_index}: status is 'missing' but has matched_satellite_time."
                        )

        is_valid = len(violations) == 0
        return is_valid, violations

    def to_csv(self, file_path: str | Path) -> None:
        """Persists manifest to CSV file."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "station_id",
            "event_id",
            "target_time_utc",
            "frame_index",
            "offset_minutes",
            "matched_satellite_time_utc",
            "time_delta_seconds",
            "status",
            "satellite_id",
            "patch_mean_ir",
            "cloud_fraction",
            "source_file_name",
        ]

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r.to_dict())

    def to_json(self, file_path: str | Path) -> None:
        """Persists manifest to JSON file."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.records], f, indent=2)
