"""Test Suite for Multimodal Model Architecture and Forward Passes.

Tests:
- Forward pass across Satellite Trajectory, Tabular Series, and Terrain branches (FR4.1–FR4.4).
- Missing-frame masking behavior (FR4.5).
- Model ablation modes (tabular_only, tabular_plus_terrain, full).
- Score bounding [0.0, 1.0].
"""

from __future__ import annotations

import unittest

from src.model.dataset import MultimodalEventSample
from src.model.multimodal_model import MultimodalFlashFloodNet


class TestMultimodalModel(unittest.TestCase):

    def setUp(self):
        self.model = MultimodalFlashFloodNet(seed=42)

        # Create synthetic test sample
        # 9 timesteps of 3x16x16 patches
        t, c, h, w = 9, 3, 16, 16
        patches = [[[[0.1 for _ in range(w)] for _ in range(h)] for _ in range(c)] for _ in range(t)]
        mask = [1, 1, 1, 0, 1, 1, 0, 1, 1]  # 2 missing frames
        tabular = [[0.1, 0.2, 0.5, 0.4, 0.6, 0.2] for _ in range(6)]
        terrain = [0.8, 0.6, 0.4, 0.7, 1.0]

        self.sample = MultimodalEventSample(
            sample_id="test_samp_1",
            station_id="USW00093037",
            timestamp_utc="2023-08-15T16:00:00Z",
            trajectory_patches=patches,
            trajectory_mask=mask,
            tabular_features=tabular,
            terrain_features=terrain,
            target_flood_label=1,
            rainfall_1h_mm=38.0,
            rainfall_3h_mm=45.0,
            elevation_m=2800.0,
            is_elevated_terrain=True,
        )

    def test_forward_pass_returns_valid_risk_probability(self):
        score = self.model.predict_sample(self.sample)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_ablation_modes(self):
        """FR6.4: Test each ablation mode executes correctly."""
        score_tab = self.model.predict_sample(self.sample, ablation_mode="tabular_only")
        score_tab_terr = self.model.predict_sample(self.sample, ablation_mode="tabular_plus_terrain")
        score_full = self.model.predict_sample(self.sample, ablation_mode=None)

        self.assertTrue(0.0 <= score_tab <= 1.0)
        self.assertTrue(0.0 <= score_tab_terr <= 1.0)
        self.assertTrue(0.0 <= score_full <= 1.0)

    def test_all_missing_trajectory_graceful_handling(self):
        """Verify model handles entirely missing satellite trajectory gracefully (FR4.5)."""
        all_missing_mask = [0 for _ in range(9)]
        score = self.model.forward(
            trajectory_patches=self.sample.trajectory_patches,
            trajectory_mask=all_missing_mask,
            tabular_features=self.sample.tabular_features,
            terrain_features=self.sample.terrain_features,
        )
        self.assertIsInstance(score, float)
        self.assertTrue(0.0 <= score <= 1.0)


if __name__ == "__main__":
    unittest.main()
