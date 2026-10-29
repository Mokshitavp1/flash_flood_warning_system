"""Multimodal Dataset for Station-Anchored Flash Flood Prediction.

Packages satellite trajectory patches with valid-frame masks, GHCNh weather sequences,
and static terrain features into aligned training samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.ingestion.ghcnh_parser import GHCNhObservation, StationMetadata
from src.terrain.dem_pipeline import StationTerrainFeatures
from src.trajectory.trajectory_builder import SatelliteTrajectorySequence


@dataclass
class MultimodalEventSample:
    """A single aligned multimodal observation sample."""
    sample_id: str
    station_id: str
    timestamp_utc: str
    # Satellite trajectory [T, C, H, W] and valid mask [T]
    trajectory_patches: List[List[List[List[float]]]]
    trajectory_mask: List[int]  # 1 for valid frame, 0 for missing frame (FR4.5)
    # Tabular GHCNh time series [Seq_len, Num_features]
    tabular_features: List[List[float]]
    # Static terrain vector [5]
    terrain_features: List[float]
    # Ground truth label: 1 if flash flood occurred downstream/at station, 0 otherwise
    target_flood_label: int
    rainfall_1h_mm: float
    rainfall_3h_mm: float
    elevation_m: float
    is_elevated_terrain: bool
    # 1 means precipitation was missing and its numeric value was imputed to 0.
    # This is a mask, not a model feature.
    precipitation_missing_mask: List[int] = field(default_factory=list)
    rainfall_6h_mm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "station_id": self.station_id,
            "timestamp_utc": self.timestamp_utc,
            "trajectory_mask": self.trajectory_mask,
            "tabular_shape": [len(self.tabular_features), len(self.tabular_features[0]) if self.tabular_features else 0],
            "precipitation_missing_mask": self.precipitation_missing_mask,
            "terrain_features": self.terrain_features,
            "target_flood_label": self.target_flood_label,
            "rainfall_1h_mm": self.rainfall_1h_mm,
            "rainfall_3h_mm": self.rainfall_3h_mm,
            "elevation_m": self.elevation_m,
            "is_elevated_terrain": self.is_elevated_terrain,
        }


class StationAnchoredFloodDataset:
    """Manages collection of multimodal training/validation samples."""

    def __init__(self, samples: Optional[List[MultimodalEventSample]] = None):
        self.samples: List[MultimodalEventSample] = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultimodalEventSample:
        return self.samples[idx]

    def add_sample(self, sample: MultimodalEventSample) -> None:
        self.samples.append(sample)

    @classmethod
    def create_sample(
        cls,
        station_meta: StationMetadata,
        observation: GHCNhObservation,
        historical_obs: List[GHCNhObservation],
        trajectory: SatelliteTrajectorySequence,
        terrain: StationTerrainFeatures,
        flood_label: Optional[int] = None,
    ) -> MultimodalEventSample:
        """Constructs an aligned multimodal sample from pipeline outputs."""
        # 1. Tabular feature sequence: last 6 hours
        # Normalize variables: precip (0-100mm), temp (-20 to 50C), rh (0-100), pressure (800-1050), wind (0-40)
        recent = historical_obs[-6:] if len(historical_obs) >= 6 else historical_obs
        tabular_seq: List[List[float]] = []
        precip_missing_mask: List[int] = []

        for obs in recent:
            p1 = (obs.precipitation_1h_mm or 0.0) / 50.0
            p3 = (obs.precipitation_3h_mm or 0.0) / 100.0
            temp = ((obs.temperature_c or 20.0) + 10.0) / 50.0
            rh = (obs.relative_humidity or 50.0) / 100.0
            stn_p = ((obs.station_pressure_hpa or 900.0) - 750.0) / 300.0
            wspd = (obs.wind_speed_mps or 3.0) / 30.0
            tabular_seq.append([
                round(p1, 4),
                round(p3, 4),
                round(temp, 4),
                round(rh, 4),
                round(stn_p, 4),
                round(wspd, 4),
            ])
            precip_missing_mask.append(int(obs.precip_missing))

        # Pad tabular sequence to length 6 if needed
        while len(tabular_seq) < 6:
            tabular_seq.insert(0, [0.0, 0.0, 0.5, 0.5, 0.5, 0.1])
            # Padding has no observed precipitation value, so it is masked.
            precip_missing_mask.insert(0, 1)

        # 2. Terrain features
        terr_vec = terrain.to_feature_vector()

        # 3. Target label definition (if not explicitly provided):
        # Flash flood event ground truth occurs if:
        # High rainfall (> 30mm/h or > 45mm/3h) on steep elevated terrain (> 2000m)
        if flood_label is None:
            p1h = observation.precipitation_1h_mm or 0.0
            p3h = observation.precipitation_3h_mm or 0.0
            is_steep = terrain.slope_degrees > 8.0
            if (p1h >= 30.0 or p3h >= 45.0) and terrain.is_elevated_terrain and is_steep:
                flood_label = 1
            elif (p1h >= 40.0) and terrain.is_elevated_terrain:
                flood_label = 1
            else:
                flood_label = 0

        sample_id = f"samp_{station_meta.station_id}_{observation.timestamp[:13]}"

        return MultimodalEventSample(
            sample_id=sample_id,
            station_id=station_meta.station_id,
            timestamp_utc=observation.timestamp,
            trajectory_patches=trajectory.frames_data,
            trajectory_mask=trajectory.valid_mask,
            tabular_features=tabular_seq,
            precipitation_missing_mask=precip_missing_mask,
            terrain_features=terr_vec,
            target_flood_label=flood_label,
            rainfall_1h_mm=observation.precipitation_1h_mm or 0.0,
            rainfall_3h_mm=observation.precipitation_3h_mm or 0.0,
            rainfall_6h_mm=observation.precipitation_6h_mm or 0.0,
            elevation_m=station_meta.elevation_m,
            is_elevated_terrain=terrain.is_elevated_terrain,
        )

    def split_train_val_test(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[StationAnchoredFloodDataset, StationAnchoredFloodDataset, StationAnchoredFloodDataset]:
        """Chronological or partitioned dataset split."""
        n = len(self.samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set = StationAnchoredFloodDataset(self.samples[:n_train])
        val_set = StationAnchoredFloodDataset(self.samples[n_train : n_train + n_val])
        test_set = StationAnchoredFloodDataset(self.samples[n_train + n_val :])

        return train_set, val_set, test_set
