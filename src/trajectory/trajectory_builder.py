"""Station-Anchored Satellite Trajectory Builder.

Implements Functional Requirements:
- FR2.1: Query satellite imagery in ±2h window around station observation.
- FR2.2: Spatial patch extraction centered on station coordinates.
- FR2.3: Fixed sub-interval sampling (e.g., 30-min cadence = 9 frames).
- FR2.4: Tolerance check (±7.5 min); record status='missing' without forward-filling.
- FR2.5: Manifest generation per station-event.
- NFR1: Verification of missing data integrity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.ingestion.ghcnh_parser import GHCNhObservation, StationMetadata
from src.trajectory.gee_client import GEEClient
from src.trajectory.manifest import MissingFrameManifest, TrajectoryFrameManifestItem


@dataclass
class SatelliteTrajectorySequence:
    """Multimodal satellite trajectory for a single station observation event."""
    station_id: str
    event_id: str
    event_timestamp_utc: str
    latitude: float
    longitude: float
    timesteps_minutes: List[int]  # e.g. [-120, -90, -60, -30, 0, 30, 60, 90, 120]
    frames_data: List[List[List[List[float]]]]  # Shape: [T, C, H, W]
    valid_mask: List[int]  # 1 if frame is present and valid ('ok'), 0 if 'missing'
    frame_statuses: List[str]  # "ok" or "missing" for each frame
    coverage_rate: float  # Percentage of valid frames (0.0 to 1.0)
    synthetic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "event_id": self.event_id,
            "event_timestamp_utc": self.event_timestamp_utc,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timesteps_minutes": self.timesteps_minutes,
            "valid_mask": self.valid_mask,
            "frame_statuses": self.frame_statuses,
            "coverage_rate": self.coverage_rate,
            "synthetic": self.synthetic,
            "shape": [
                len(self.frames_data),
                len(self.frames_data[0]) if self.frames_data else 0,
                len(self.frames_data[0][0]) if self.frames_data and self.frames_data[0] else 0,
                len(self.frames_data[0][0][0]) if self.frames_data and self.frames_data[0] and self.frames_data[0][0] else 0,
            ],
        }


class SatelliteTrajectoryBuilder:
    """Builds per-station satellite trajectories with explicit missing-frame handling."""

    def __init__(
        self,
        window_hours: float = 2.0,
        step_minutes: int = 30,
        tolerance_minutes: float = 7.5,
        patch_size: int = 32,
        gee_client: Optional[GEEClient] = None,
    ):
        self.window_hours = window_hours
        self.step_minutes = step_minutes
        self.tolerance_seconds = tolerance_minutes * 60.0
        self.patch_size = patch_size
        self.gee_client = gee_client or GEEClient(patch_size=patch_size)

        # Generate offset steps e.g. [-120, -90, -60, -30, 0, 30, 60, 90, 120]
        max_offset = int(window_hours * 60)
        self.offset_minutes_list: List[int] = list(
            range(-max_offset, max_offset + step_minutes, step_minutes)
        )

    def _create_placeholder_patch(self, num_channels: int = 3) -> List[List[List[float]]]:
        """Creates an explicit zero placeholder tensor for missing frames.
        
        CRITICAL NFR1 NOTE: This placeholder is accompanied by a mask bit of 0.
        The downstream neural network's masked attention ensures this placeholder is
        never treated as real observational data!
        """
        return [[[0.0 for _ in range(self.patch_size)] for _ in range(self.patch_size)] for _ in range(num_channels)]

    def build_trajectory(
        self,
        station_meta: StationMetadata,
        observation: GHCNhObservation,
        manifest: Optional[MissingFrameManifest] = None,
    ) -> SatelliteTrajectorySequence:
        """
        Builds the satellite trajectory around the station's observation timestamp.
        Appends frame records to the manifest if provided.
        """
        event_dt = datetime.fromisoformat(observation.timestamp.replace("Z", "+00:00"))
        event_id = f"{station_meta.station_id}_{event_dt.strftime('%Y%m%d%H%M')}"

        frames_data: List[List[List[List[float]]]] = []
        valid_mask: List[int] = []
        frame_statuses: List[str] = []

        storm_intensity = observation.precipitation_1h_mm or 0.0

        for frame_idx, offset_min in enumerate(self.offset_minutes_list):
            target_time = event_dt + timedelta(minutes=offset_min)
            target_time_iso = target_time.isoformat()

            # Query GEE client for frame within tolerance
            image_result = self.gee_client.query_nearest_image(
                latitude=station_meta.latitude,
                longitude=station_meta.longitude,
                target_time=target_time,
                tolerance_seconds=self.tolerance_seconds,
                storm_intensity=storm_intensity,
            )

            if image_result is not None:
                # Frame successfully acquired within tolerance (FR2.4)
                status = "ok"
                frames_data.append(image_result["data"])
                valid_mask.append(1)
                frame_statuses.append(status)

                if manifest is not None:
                    manifest_item = TrajectoryFrameManifestItem(
                        station_id=station_meta.station_id,
                        event_id=event_id,
                        target_time_utc=target_time_iso,
                        frame_index=frame_idx,
                        offset_minutes=offset_min,
                        matched_satellite_time_utc=image_result["matched_time_utc"],
                        time_delta_seconds=image_result["time_delta_seconds"],
                        status=status,
                        satellite_id=image_result.get("satellite_id", "GOES-16"),
                        patch_mean_ir=image_result.get("patch_mean_ir"),
                        cloud_fraction=image_result.get("cloud_fraction"),
                        source_file_name=image_result.get("source_file_name"),
                        synthetic=image_result.get("synthetic", self.gee_client.synthetic),
                    )
                    manifest.add_record(manifest_item)
            else:
                # Missing frame! Insert placeholder and record status='missing' (FR2.4, NFR1)
                # NO forward-filling or label reuse!
                status = "missing"
                frames_data.append(self._create_placeholder_patch())
                valid_mask.append(0)
                frame_statuses.append(status)

                if manifest is not None:
                    manifest_item = TrajectoryFrameManifestItem(
                        station_id=station_meta.station_id,
                        event_id=event_id,
                        target_time_utc=target_time_iso,
                        frame_index=frame_idx,
                        offset_minutes=offset_min,
                        matched_satellite_time_utc=None,
                        time_delta_seconds=None,
                        status=status,
                        satellite_id="NONE",
                        patch_mean_ir=None,
                        cloud_fraction=None,
                        source_file_name=None,
                        synthetic=self.gee_client.synthetic,
                    )
                    manifest.add_record(manifest_item)

        coverage = sum(valid_mask) / len(valid_mask) if valid_mask else 0.0

        return SatelliteTrajectorySequence(
            station_id=station_meta.station_id,
            event_id=event_id,
            event_timestamp_utc=observation.timestamp,
            latitude=station_meta.latitude,
            longitude=station_meta.longitude,
            timesteps_minutes=self.offset_minutes_list,
            frames_data=frames_data,
            valid_mask=valid_mask,
            frame_statuses=frame_statuses,
            coverage_rate=round(coverage, 3),
            synthetic=self.gee_client.synthetic,
        )
