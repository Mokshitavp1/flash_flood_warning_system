"""Test Suite for Non-Functional Requirement 1 (NFR1): Data Integrity & Missing Frames.

HARD REQUIREMENT:
- Missing satellite frames must NEVER silently degrade into mislabeled or forward-filled data.
- Manifest must record status = 'missing' for gaps.
- Frames marked 'missing' must NEVER reuse images, timestamps, or labels from prior timesteps.
- Neural network temporal attention must explicitly mask out missing frames (mask=0).
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.ingestion.ghcnh_parser import GHCNhObservation, StationMetadata
from src.model.dataset import MultimodalEventSample
from src.model.multimodal_model import (
    MultimodalFlashFloodNet,
    PurePythonTemporalEncoderWithMasking,
)
from src.trajectory.gee_client import GEEClient
from src.trajectory.manifest import MissingFrameManifest, TrajectoryFrameManifestItem
from src.trajectory.trajectory_builder import SatelliteTrajectoryBuilder


class MockGEEWithForcedDropouts(GEEClient):
    """Controlled GEE mock that returns observations for select timesteps and gaps for others."""

    def __init__(self, missing_indices: List[int], patch_size: int = 32):
        super().__init__(patch_size=patch_size, simulation=True)
        self.missing_indices = set(missing_indices)
        self.call_count = 0

    def query_nearest_image(
        self,
        latitude: float,
        longitude: float,
        target_time: datetime,
        tolerance_seconds: float = 450.0,
        storm_intensity: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        idx = self.call_count
        self.call_count += 1

        if idx in self.missing_indices:
            # Force missing frame: e.g. satellite telemetry drop, eclipse, or beyond tolerance
            return None

        # Valid frame
        return {
            "satellite_id": "GOES-16",
            "collection": self.collection_id,
            "bands": self.bands,
            "matched_time_utc": target_time.isoformat(),
            "time_delta_seconds": 12.0,
            "data": self._synthesize_patch(storm_intensity=storm_intensity),
            "patch_mean_ir": 220.5,
            "cloud_fraction": 0.85,
        }


class TestNFR1MissingFramesIntegrity(unittest.TestCase):
    """Rigorous verification of NFR1 missing frame handling."""

    def setUp(self):
        self.station_meta = StationMetadata(
            station_id="USW00093037",
            station_name="SANDIA MOUNTAIN PEAK",
            latitude=35.2103,
            longitude=-106.4497,
            elevation_m=3255.0,
        )
        self.observation = GHCNhObservation(
            station_id="USW00093037",
            timestamp="2023-08-15T16:00:00+00:00",
            year=2023,
            month=8,
            day=15,
            hour=16,
            minute=0,
            latitude=35.2103,
            longitude=-106.4497,
            elevation_m=3255.0,
            temperature_c=18.5,
            dew_point_c=14.0,
            relative_humidity=75.0,
            wind_direction_deg=220,
            wind_speed_mps=8.5,
            station_pressure_hpa=710.0,
            sea_level_pressure_hpa=1010.0,
            precipitation_mm=42.0,
            precipitation_1h_mm=42.0,
            precipitation_3h_mm=55.0,
            precipitation_6h_mm=60.0,
            present_weather="TSRA",
            quality_flag="V",
        )

    def test_missing_frames_never_forward_filled(self):
        """Verify that when frames 2, 4, 7 are missing, they are marked 'missing' and NOT forward-filled."""
        # Force timesteps 2, 4, 7 to be missing out of 9 total timesteps
        forced_missing = [2, 4, 7]
        mock_gee = MockGEEWithForcedDropouts(missing_indices=forced_missing, patch_size=16)
        manifest = MissingFrameManifest()

        builder = SatelliteTrajectoryBuilder(
            window_hours=2.0,
            step_minutes=30,
            tolerance_minutes=7.5,
            patch_size=16,
            gee_client=mock_gee,
        )

        trajectory = builder.build_trajectory(
            station_meta=self.station_meta,
            observation=self.observation,
            manifest=manifest,
        )

        # 1. Total timesteps check
        self.assertEqual(len(trajectory.frames_data), 9)
        self.assertEqual(len(trajectory.valid_mask), 9)
        self.assertEqual(len(trajectory.frame_statuses), 9)

        # 2. Check each timestep
        for idx in range(9):
            if idx in forced_missing:
                # Must be strictly marked 'missing'
                self.assertEqual(
                    trajectory.frame_statuses[idx],
                    "missing",
                    f"Frame {idx} should have status='missing'!",
                )
                self.assertEqual(
                    trajectory.valid_mask[idx],
                    0,
                    f"Frame {idx} mask must be 0 (masked out)!",
                )
            else:
                self.assertEqual(
                    trajectory.frame_statuses[idx],
                    "ok",
                    f"Frame {idx} should have status='ok'!",
                )
                self.assertEqual(
                    trajectory.valid_mask[idx],
                    1,
                    f"Frame {idx} mask must be 1!",
                )

        # 3. Check Manifest records: NO timestamp or data reuse
        manifest_items = manifest.records
        self.assertEqual(len(manifest_items), 9)

        for idx in range(9):
            item = manifest_items[idx]
            if idx in forced_missing:
                self.assertEqual(item.status, "missing")
                self.assertIsNone(
                    item.matched_satellite_time_utc,
                    f"Missing frame {idx} must not have a matched timestamp!",
                )
            else:
                self.assertEqual(item.status, "ok")
                self.assertIsNotNone(item.matched_satellite_time_utc)
                # Ensure it's not copying from another frame
                if idx > 0 and manifest_items[idx - 1].status == "ok":
                    self.assertNotEqual(
                        item.matched_satellite_time_utc,
                        manifest_items[idx - 1].matched_satellite_time_utc,
                        "Timestamps between consecutive observations must differ!",
                    )

        # 4. Strict NFR1 audit check passes
        is_valid, violations = manifest.verify_nfr1_integrity()
        self.assertTrue(is_valid, f"NFR1 audit failed with violations: {violations}")

    def test_nfr1_audit_detects_illegal_forward_fill(self):
        """Verify that manifest.verify_nfr1_integrity() catches and rejects an illegal forward fill."""
        manifest = MissingFrameManifest()

        # Legitimate frame 0
        manifest.add_record(
            TrajectoryFrameManifestItem(
                station_id="USW00093037",
                event_id="test_event_1",
                target_time_utc="2023-08-15T14:00:00Z",
                frame_index=0,
                offset_minutes=-120,
                matched_satellite_time_utc="2023-08-15T13:58:00Z",
                time_delta_seconds=120.0,
                status="ok",
                satellite_id="GOES-16",
            )
        )

        # Corrupted frame 1: forward-filled by reusing frame 0's matched time!
        manifest.add_record(
            TrajectoryFrameManifestItem(
                station_id="USW00093037",
                event_id="test_event_1",
                target_time_utc="2023-08-15T14:30:00Z",
                frame_index=1,
                offset_minutes=-90,
                matched_satellite_time_utc="2023-08-15T13:58:00Z",  # ILLEGAL REUSE!
                time_delta_seconds=1920.0,
                status="ok",
                satellite_id="GOES-16",
            )
        )

        is_valid, violations = manifest.verify_nfr1_integrity()
        self.assertFalse(is_valid, "verify_nfr1_integrity MUST detect forward-fill corruption!")
        self.assertTrue(any("FORWARD-FILL DETECTED" in v for v in violations))

    def test_model_temporal_attention_masks_out_missing_frames(self):
        """
        Verify that in the neural network, missing frames (mask=0) are masked out
        and do not corrupt the temporal representation with zero-pixel placeholders (FR4.5).
        """
        encoder = PurePythonTemporalEncoderWithMasking(in_dim=16, hidden_dim=16, seed=42)

        # 4 timesteps with random embeddings
        t0 = [0.5 for _ in range(16)]
        t1 = [0.8 for _ in range(16)]
        t2_placeholder = [0.0 for _ in range(16)]  # Missing frame placeholder
        t3 = [0.6 for _ in range(16)]

        # Mask: t2 is missing!
        mask_with_missing = [1, 1, 0, 1]
        embeddings = [t0, t1, t2_placeholder, t3]

        out_with_missing = encoder.forward(embeddings, mask_with_missing)

        # If t2 was NOT masked, putting large bogus numbers in t2 would alter the output.
        # But because t2 is masked (mask=0), changing t2_placeholder should NOT change the output at all!
        t2_bogus = [999.0 for _ in range(16)]
        embeddings_bogus = [t0, t1, t2_bogus, t3]
        out_bogus = encoder.forward(embeddings_bogus, mask_with_missing)

        for d in range(16):
            self.assertAlmostEqual(
                out_with_missing[d],
                out_bogus[d],
                places=5,
                msg=f"Masking failed: dim {d} changed when missing frame placeholder changed!",
            )


if __name__ == "__main__":
    unittest.main()
