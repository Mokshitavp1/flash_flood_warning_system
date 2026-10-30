"""Tests for the learned risk backend and checkpoint contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.model.dataset import MultimodalEventSample, StationAnchoredFloodDataset
from src.model.training import LearnedRiskModel, split_by_event


def _sample(day: str, label: int) -> MultimodalEventSample:
    return MultimodalEventSample(
        sample_id=f"{day}-{label}",
        station_id="station",
        timestamp_utc=f"{day}T12:00:00Z",
        trajectory_patches=[[[[float(label)]]]],
        trajectory_mask=[1],
        tabular_features=[[float(label)] * 6],
        terrain_features=[float(label)] * 5,
        target_flood_label=label,
        rainfall_1h_mm=float(label),
        rainfall_3h_mm=float(label),
        elevation_m=2000.0,
        is_elevated_terrain=bool(label),
    )


class TestLearnedRiskModel(unittest.TestCase):
    def test_fit_saves_and_reloads_checkpoint_with_held_out_metrics(self):
        samples = [
            _sample("2025-07-01", 0),
            _sample("2025-07-01", 1),
            _sample("2025-07-02", 0),
            _sample("2025-07-02", 1),
            _sample("2025-07-03", 0),
            _sample("2025-07-03", 1),
        ]
        split = split_by_event(samples)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "risk-model.json"
            report = LearnedRiskModel(learning_rate=0.1).fit(
                split, epochs=5, checkpoint_path=checkpoint
            )
            loaded = LearnedRiskModel.load_checkpoint(checkpoint)

            self.assertTrue(checkpoint.exists())
            self.assertEqual(report["backend"], "pure_python_logistic_sgd")
            self.assertIn("held_out_test", report)
            self.assertEqual(
                loaded.evaluate(split.test).to_dict(),
                report["held_out_test"],
            )


if __name__ == "__main__":
    unittest.main()
