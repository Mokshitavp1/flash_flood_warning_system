"""End-to-End Pipeline Integration Test.

Runs all 6 stages of the HydroAlert pipeline:
1. Ingestion on synthetic GHCNh data
2. Satellite trajectory construction & manifest tracking
3. DEM flow routing and terrain features
4. Multimodal prediction
5. Alert propagation (circular buffer vs DEM flow routing)
6. Baseline comparisons and 4-stage ablation
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ingestion.mock_ghcnh_generator import MockGHCNhGenerator
from src.pipeline_runner import FlashFloodPipelineRunner
from src.terrain.dem_pipeline import DEMPipeline
from src.trajectory.gee_client import GEEClient


class TestPipelineEndToEnd(unittest.TestCase):

    def test_full_pipeline_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_data_dir = temp_path / "mock_ghcnh"

            # Generate synthetic GHCNh files
            mock_gen = MockGHCNhGenerator(seed=123)
            mock_gen.write_mock_dataset(mock_data_dir, num_hours=24)

            # Initialize and run pipeline
            runner = FlashFloodPipelineRunner(
                satellite_client=GEEClient(seed=123),
                dem_pipeline=DEMPipeline(
                    dem_loader=lambda _station: (
                        [
                            [120.0, 110.0, 40.0],
                            [110.0, 100.0, 30.0],
                            [40.0, 30.0, 0.0],
                        ],
                        1,
                        1,
                    )
                ),
            )
            # Overwrite output targets to temp directory
            runner.manifest_path = temp_path / "manifest.csv"
            runner.alerts_dir = temp_path / "alerts"
            runner.evaluation_dir = temp_path / "evaluation"

            results = runner.run_all(raw_data_dir=str(mock_data_dir))

            # 1. Pipeline finished successfully
            self.assertIn("elapsed_seconds", results)
            self.assertGreater(results["num_stations"], 0)
            self.assertGreater(results["num_samples"], 0)

            # 2. Manifest created and verified
            self.assertTrue(runner.manifest_path.exists())
            self.assertGreater(results["manifest_coverage"]["total_frames"], 0)

            # 3. Alert outputs created
            self.assertTrue((runner.alerts_dir / "alert_event_audit_log.json").exists())
            self.assertTrue((runner.alerts_dir / "alert_event_audit_log.csv").exists())
            self.assertIn("v0_circular_buffer", results["alert_summary"])
            self.assertIn("v1_dem_flow_routing", results["alert_summary"])

            # 4. Evaluation and ablation benchmark results created
            self.assertTrue((runner.evaluation_dir / "ablation_benchmark_results.json").exists())
            ablation = results["ablation_results"]
            self.assertIn("stage_1_tabular", ablation)
            self.assertIn("stage_2_tabular_terrain", ablation)
            self.assertIn("stage_3_multimodal", ablation)
            self.assertIn("stage_4_flow_routing", ablation)
            self.assertIn("baseline_rainfall", ablation)
            self.assertIn("baseline_noaa_ffg", ablation)

            # Stage 4 must not manufacture sample-level hits. Downstream
            # routing is audited separately in the alert outputs.
            self.assertEqual(
                ablation["stage_4_flow_routing"]["csi"],
                ablation["stage_3_multimodal"]["csi"],
            )


if __name__ == "__main__":
    unittest.main()
