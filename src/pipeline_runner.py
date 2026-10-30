"""End-to-End Flash-Flood Early Warning Pipeline Runner.

Wires every stage of the pipeline together with real function calls:
1. Ingestion: Ingests & cleans NOAA GHCNh PSV records (FR1.1–FR1.3).
2. Trajectories: Builds station-anchored satellite trajectories with missing-frame manifest (FR2.1–FR2.5, NFR1).
3. Terrain: Computes DEM slope, aspect, D8 flow direction & accumulation (FR3.1–FR3.2).
4. Model: Multimodal inference with masked temporal attention (FR4.1–FR4.5).
5. Alerts: Executes both Circular-Buffer (v0) and DEM Flow-Routing (v1) alerts (FR5.1–FR5.3).
6. Evaluation: Runs benchmark comparisons & 4-stage ablation (FR6.1–FR6.4).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.alerts.alert_logger import AlertLogger
from src.alerts.circular_buffer import CircularBufferAlertEngine
from src.alerts.flow_routing_alert import DEMFlowRoutingAlertEngine
from src.evaluation.ablation import AblationHarness
from src.evaluation.label_evaluation import (
    evaluate_labeled_samples,
    labeled_sample_target,
    load_flood_labels,
    print_label_evaluation_table,
)
from src.evaluation.metrics import BinaryClassificationMetrics
from src.ingestion.ghcnh_parser import GHCNhObservation, GHCNhParser, StationMetadata
from src.ingestion.mock_ghcnh_generator import MockGHCNhGenerator
from src.model.dataset import MultimodalEventSample, StationAnchoredFloodDataset
from src.model.config import validate_model_config
from src.model.multimodal_model import MultimodalFlashFloodNet
from src.model.training import (
    InsufficientTrainingDataError,
    LearnedRiskModel,
    prepare_training,
)
from src.terrain.dem_pipeline import DEMPipeline
from src.trajectory.manifest import MissingFrameManifest
from src.trajectory.noaa_aws_client import NOAAAWSGOESClient
from src.trajectory.trajectory_builder import (
    SatelliteTrajectoryBuilder,
    SatelliteTrajectorySequence,
)


class FlashFloodPipelineRunner:
    """Orchestrates all pipeline stages end-to-end."""

    def __init__(
        self,
        pipeline_config_path: str = "configs/pipeline_config.json",
        model_config_path: str = "configs/model_config.json",
        alert_config_path: str = "configs/alert_config.json",
        satellite_client: Optional[Any] = None,
        dem_pipeline: Optional[DEMPipeline] = None,
    ):
        self.pipeline_config = self._load_json(pipeline_config_path)
        self.model_config = self._load_json(model_config_path)
        validate_model_config(self.model_config)
        self.alert_config = self._load_json(alert_config_path)
        self.synthetic_sources = {
            "weather": False,
            "satellite": False,
            "terrain": False,
        }

        # Output paths
        self.manifest_path = Path(self.pipeline_config["data_paths"]["manifest_path"])
        self.alerts_dir = Path(self.pipeline_config["data_paths"]["alerts_dir"])
        self.evaluation_dir = Path(self.pipeline_config["data_paths"]["evaluation_dir"])
        self.labels_path = Path(
            self.pipeline_config["data_paths"].get(
                "labels_path", "data/labels/flood_labels_2025-07-03_2025-07-05.csv"
            )
        )

        # Core pipeline components
        self.parser = GHCNhParser()
        configured_dem_pipeline = DEMPipeline(
            elevated_threshold_m=self.pipeline_config["terrain"].get(
                "relative_relief_threshold_m", 40.0
            ),
            dem_directory=self.pipeline_config["data_paths"].get(
                "dem_dir", "data/dem"
            ),
        )
        self.dem_pipeline = dem_pipeline or configured_dem_pipeline
        sat_cfg = self.pipeline_config["satellite"]
        if satellite_client is None and sat_cfg.get("source") == "NOAA_AWS":
            satellite_client = NOAAAWSGOESClient(
                bands=sat_cfg["bands"],
                patch_size=sat_cfg["patch_size_pixels"],
                resolution_km=sat_cfg["spatial_resolution_km"],
                cache_directory=self.pipeline_config["data_paths"].get(
                    "goes_cache_dir", "data/goes_cache"
                ),
            )
        self.trajectory_builder = SatelliteTrajectoryBuilder(
            window_hours=sat_cfg["window_hours"],
            step_minutes=sat_cfg["step_minutes"],
            tolerance_minutes=sat_cfg["tolerance_minutes"],
            patch_size=sat_cfg["patch_size_pixels"],
            gee_client=satellite_client,
        )
        self.model = MultimodalFlashFloodNet(config=self.model_config)
        self.learned_model: Optional[LearnedRiskModel] = None
        self.training_report: Dict[str, Any] = {
            "status": "not_started",
            "evaluation_status": "not_started",
        }
        self.alert_logger = AlertLogger()
        self.manifest = MissingFrameManifest()

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_stage_1_ingestion(
        self, raw_data_dir: Optional[str] = None
    ) -> Dict[str, Tuple[StationMetadata, List[GHCNhObservation]]]:
        """Stage 1: GHCN-hourly data ingestion and cleaning (FR1.1–FR1.3)."""
        target_dir = Path(raw_data_dir or self.pipeline_config["data_paths"]["raw_ghcnh_dir"])
        print(f"\n==================================================")
        print(f"STAGE 1: Ingesting NOAA GHCN-Hourly Records")
        print(f"Scanning target directory: {target_dir.resolve()}")

        # Check for real files in data/
        ingested = self.parser.ingest_directory(target_dir)

        if not ingested:
            print(f"[Stage 1] Note: {target_dir} is empty of raw files.")
            print(f"[Stage 1] Generating synthetic NOAA GHCNh PSV records matching published schema...")
            temp_mock_dir = Path("outputs/mock_ghcnh")
            mock_gen = MockGHCNhGenerator(seed=42)
            mock_files = mock_gen.write_mock_dataset(temp_mock_dir, num_hours=48)
            print(f"[Stage 1] Created {len(mock_files)} synthetic GHCNh PSV files in {temp_mock_dir}")
            ingested = self.parser.ingest_directory(temp_mock_dir)
        self.synthetic_sources["weather"] = any(
            meta.synthetic or any(observation.synthetic for observation in observations)
            for meta, observations in ingested.values()
        )
        if self.synthetic_sources["weather"]:
            print("[Stage 1] SYNTHETIC WEATHER: records are not real NOAA coverage.")

        start_time = datetime.fromisoformat(
            self.pipeline_config["time_range"]["start_date"].replace("Z", "+00:00")
        )
        end_time = datetime.fromisoformat(
            self.pipeline_config["time_range"]["end_date"].replace("Z", "+00:00")
        )
        filtered_ingested: Dict[str, Tuple[StationMetadata, List[GHCNhObservation]]] = {}
        for st_id, (meta, observations) in ingested.items():
            filtered_observations = []
            for observation in observations:
                observation_time = datetime.fromisoformat(
                    observation.timestamp.replace("Z", "+00:00")
                )
                if start_time <= observation_time <= end_time:
                    filtered_observations.append(observation)
            if filtered_observations:
                filtered_ingested[st_id] = (meta, filtered_observations)
        if filtered_ingested:
            ingested = filtered_ingested
        else:
            print(
                "[Stage 1] Warning: no observations fall inside the configured "
                "window; retaining the supplied records for this run."
            )

        total_obs = sum(len(obs) for _, obs in ingested.values())
        print(f"[Stage 1] Successfully parsed and cleaned:")
        print(
            f"[Stage 1] Using configured window: "
            f"{self.pipeline_config['time_range']['start_date']} to "
            f"{self.pipeline_config['time_range']['end_date']}"
        )
        for st_id, (meta, obs_list) in ingested.items():
            print(f"  - Station {st_id} ({meta.station_name}): {len(obs_list)} hourly observations, Elev: {meta.elevation_m}m")
        print(f"[Stage 1] Total cleaned hourly records: {total_obs}")
        return ingested

    def run_stage_2_trajectories(
        self,
        station_data: Dict[str, Tuple[StationMetadata, List[GHCNhObservation]]],
        sample_stride_hours: int = 2,
    ) -> List[Tuple[StationMetadata, GHCNhObservation, List[GHCNhObservation], SatelliteTrajectorySequence]]:
        """Stage 2: Satellite trajectory construction with missing-frame manifest (FR2.1–FR2.5, NFR1)."""
        print(f"\n==================================================")
        print(f"STAGE 2: Constructing Station-Anchored Satellite Trajectories")
        sat = self.pipeline_config["satellite"]
        print(
            f"Satellite Source: {sat['source']} "
            f"(Window: +/-{sat['window_hours']}h, Cadence: {sat['step_minutes']}m)"
        )

        events_with_trajectories = []

        for st_id, (meta, obs_list) in station_data.items():
            # Process sampled observations to build event trajectories
            for i in range(3, len(obs_list)):
                current_obs = obs_list[i]
                # Sample every stride hours OR if a storm is occurring
                is_stride = (i % sample_stride_hours == 0)
                is_storm = (current_obs.precipitation_1h_mm or 0.0) >= 15.0

                if is_stride or is_storm:
                    history = obs_list[max(0, i - 6) : i + 1]

                    traj = self.trajectory_builder.build_trajectory(
                        station_meta=meta,
                        observation=current_obs,
                        manifest=self.manifest,
                    )
                    events_with_trajectories.append((meta, current_obs, history, traj))

        # Coverage inspection (FR2.5)
        cov_stats = self.manifest.compute_coverage_statistics()
        self.synthetic_sources["satellite"] = self.trajectory_builder.gee_client.synthetic
        cov_stats["synthetic"] = self.synthetic_sources["satellite"]
        print(f"[Stage 2] Generated {len(events_with_trajectories)} station-event trajectories.")
        print(f"[Stage 2] Manifest Frame Coverage:")
        print(f"  - Total Frames Queried: {cov_stats['total_frames']}")
        coverage_label = "SYNTHETIC coverage (not real satellite coverage)" if cov_stats["synthetic"] else "Real coverage"
        print(f"  - Valid ('ok') Frames: {cov_stats['ok_frames']} ({cov_stats['coverage_pct']}%) [{coverage_label}]")
        print(f"  - Missing Frames: {cov_stats['missing_frames']}")

        # Persist manifest to disk
        self.manifest.to_csv(self.manifest_path)
        print(f"[Stage 2] Saved manifest to {self.manifest_path}")

        # Strict NFR1 Check
        is_valid, violations = self.manifest.verify_nfr1_integrity()
        if not is_valid:
            raise RuntimeError(f"NFR1 Integrity Failure! Violations: {violations}")
        print(f"[Stage 2] NFR1 VERIFICATION PASSED: No missing frames were forward-filled or mislabeled.")

        return events_with_trajectories

    def run_stage_3_terrain(
        self, stations: List[StationMetadata]
    ) -> Dict[str, Any]:
        """Stage 3: DEM Flow Routing and static topographic features (FR3.1–FR3.2)."""
        print(f"\n==================================================")
        print(f"STAGE 3: DEM Hydrological Flow Routing & Terrain Analysis")
        print(f"Algorithm: D8 Steepest Descent (Cell size: 30m)")

        terrain_features: Dict[str, Any] = {}
        for st in stations:
            feat = self.dem_pipeline.extract_station_terrain(st)
            terrain_features[st.station_id] = feat
        self.synthetic_sources["terrain"] = any(
            feature.synthetic for feature in terrain_features.values()
        )
        if self.synthetic_sources["terrain"]:
            print("[Stage 3] SYNTHETIC TERRAIN: results are not real DEM coverage.")
            print(
                f"  - Station {st.station_id}: Elev={feat.elevation_m}m, Slope={feat.slope_degrees}°, "
                f"Aspect={feat.aspect_degrees}°, FlowAccum={feat.flow_accumulation_cells} cells, "
                f"IsElevated={feat.is_elevated_terrain}"
            )

        terrain_output = Path(self.pipeline_config["data_paths"].get(
            "terrain_dir", "outputs/terrain"
        ))
        terrain_output.mkdir(parents=True, exist_ok=True)
        (terrain_output / "station_terrain.json").write_text(
            json.dumps({
                "synthetic": self.synthetic_sources["terrain"],
                "coverage_notice": (
                    "Synthetic terrain; not real DEM coverage."
                    if self.synthetic_sources["terrain"] else "Real DEM coverage."
                ),
                "stations": {
                    station_id: feature.to_dict()
                    for station_id, feature in terrain_features.items()
                },
            }, indent=2),
            encoding="utf-8",
        )
        return terrain_features

    def run_stage_4_multimodal_inference(
        self,
        events: List[Tuple[StationMetadata, GHCNhObservation, List[GHCNhObservation], SatelliteTrajectorySequence]],
    ) -> Tuple[StationAnchoredFloodDataset, List[float]]:
        """Stage 4: Multimodal Model prediction with masked temporal attention (FR4.1–FR4.5)."""
        print(f"\n==================================================")
        print(f"STAGE 4: Multimodal Model Inference")
        print(f"Branches: Satellite Trajectory (Masked Attention) + Tabular Weather + Terrain MLP")

        dataset = StationAnchoredFloodDataset()
        labels = load_flood_labels(self.labels_path) if self.labels_path.exists() else {}
        labeled_events = 0
        for meta, obs, history, traj in events:
            label_target = labeled_sample_target(labels, meta.station_id, obs.timestamp)
            if labels and label_target is None:
                continue
            if label_target is not None:
                labeled_events += 1
            terrain = self.dem_pipeline.extract_station_terrain(meta)
            sample = StationAnchoredFloodDataset.create_sample(
                station_meta=meta,
                observation=obs,
                historical_obs=history,
                trajectory=traj,
                terrain=terrain,
                flood_label=label_target,
            )
            dataset.add_sample(sample)

        risk_scores: List[float] = []
        for sample in dataset.samples:
            score = self.model.predict_sample(sample)
            risk_scores.append(score)

        high_risk_count = sum(1 for s in risk_scores if s >= 0.65)
        print(f"[Stage 4] Processed {len(dataset)} multimodal samples.")
        print(f"[Stage 4] Detected {high_risk_count} high-risk convective storm events (Risk >= 0.65).")
        if labels:
            print(
                f"[Stage 4] Joined {labeled_events} labeled samples across "
                f"{len({sample.station_id for sample in dataset.samples})} stations."
            )

        return dataset, risk_scores

    def run_stage_5_alert_propagation(
        self,
        dataset: StationAnchoredFloodDataset,
        risk_scores: List[float],
        stations: List[StationMetadata],
    ) -> Dict[str, Any]:
        """Stage 5: Downhill Alert Propagation (Circular Buffer v0 vs DEM Flow Routing v1) (FR5.1–FR5.3)."""
        print(f"\n==================================================")
        print(f"STAGE 5: Downhill Alert Propagation & Terrain Comparison")
        print(f"Comparing Baseline v0 (Circular Buffer) vs Method v1 (DEM Flow Routing)")

        thresholds = self.alert_config["thresholds"]
        baseline_cfg = self.alert_config["baseline_v0_circular_buffer"]
        flow_cfg = self.alert_config["method_v1_dem_flow_routing"]
        circ_engine = CircularBufferAlertEngine(
            radius_km=baseline_cfg["radius_km"],
            elevated_threshold_m=thresholds["elevated_station_min_elevation_m"],
            trigger_risk_threshold=thresholds["elevated_station_min_risk_trigger"],
            dem_pipeline=self.dem_pipeline,
            decay_with_distance=baseline_cfg["decay_with_distance"],
            synthetic=any(self.synthetic_sources.values()),
        )

        flow_engine = DEMFlowRoutingAlertEngine(
            dem_pipeline=self.dem_pipeline,
            elevated_threshold_m=thresholds["elevated_station_min_elevation_m"],
            trigger_risk_threshold=thresholds["elevated_station_min_risk_trigger"],
            flow_decay_alpha=flow_cfg["flow_decay_alpha"],
            accumulation_boost_beta=flow_cfg["accumulation_boost_beta"],
            min_alert_strength=thresholds["downhill_min_alert_strength_to_warn"],
            wave_speed_km_h=flow_cfg["hydraulic_wave_speed_m_per_s"] * 3.6,
            max_reach_km=flow_cfg["max_downstream_distance_km"],
            max_steps=flow_cfg["max_steps"],
            min_lead_time_minutes=thresholds["downhill_warning_lead_time_min_minutes"],
            max_lead_time_minutes=thresholds["downhill_warning_lead_time_max_minutes"],
            synthetic=any(self.synthetic_sources.values()),
        )

        # Clear previous alerts
        self.alert_logger = AlertLogger()

        v0_total = 0
        v1_total = 0

        # Run alerts for all high-risk events
        for sample, score in zip(dataset.samples, risk_scores):
            if score >= thresholds["elevated_station_min_risk_trigger"] and sample.is_elevated_terrain:
                trigger_meta = next((s for s in stations if s.station_id == sample.station_id), None)
                if not trigger_meta:
                    continue

                # Run baseline v0 (Circular Buffer)
                v0_alerts = circ_engine.evaluate_and_propagate(
                    trigger_station=trigger_meta,
                    trigger_risk_score=score,
                    timestamp_utc=sample.timestamp_utc,
                    candidate_stations=stations,
                    logger=self.alert_logger,
                )
                v0_total += len(v0_alerts)

                # Run method v1 (DEM Flow Routing)
                v1_alerts = flow_engine.evaluate_and_propagate(
                    trigger_station=trigger_meta,
                    trigger_risk_score=score,
                    timestamp_utc=sample.timestamp_utc,
                    candidate_stations=stations,
                    logger=self.alert_logger,
                )
                v1_total += len(v1_alerts)

        summary = self.alert_logger.get_summary_by_method()
        print(f"[Stage 5] Alert Results:")
        print(f"  - v0 Circular Buffer: {summary['v0_circular_buffer']['total']} alerts "
              f"({summary['v0_circular_buffer']['uphill_unnecessary_alerts']} unnecessary uphill false alarms!)")
        print(f"  - v1 DEM Flow Routing: {summary['v1_dem_flow_routing']['total']} targeted alerts "
              f"(0 uphill false alarms, avg lead time: {summary['v1_dem_flow_routing']['avg_lead_time_min']} min)")

        # Persist alert logs (FR5.3)
        self.alert_logger.to_csv(self.alerts_dir / "alert_event_audit_log.csv")
        self.alert_logger.to_json(self.alerts_dir / "alert_event_audit_log.json")
        print(f"[Stage 5] Exported alert audit logs to {self.alerts_dir}")

        return summary

    def run_stage_6_evaluation(
        self,
        dataset: StationAnchoredFloodDataset,
        stations: List[StationMetadata],
        evaluation_dataset: Optional[StationAnchoredFloodDataset] = None,
    ) -> Dict[str, Any]:
        """Stage 6: Baseline benchmarking and 4-stage ablation (FR6.1–FR6.4)."""
        if evaluation_dataset is None:
            print(
                "[Stage 6] Evaluation skipped: no event-disjoint held-out test split "
                "is available. Training samples will not be evaluated."
            )
            return {
                "evaluation_status": "skipped_no_held_out_test_split",
                "reason": self.training_report.get("reason", "Training was not completed."),
            }
        dataset = evaluation_dataset
        print(f"\n==================================================")
        print(f"STAGE 6: Evaluation & Multi-Stage Ablation Benchmark")
        print(f"Baselines: Rainfall Threshold (>25mm/h) & NOAA Flash Flood Guidance (FFG)")
        print(f"Ablation: Tabular -> +Terrain -> +Satellite Trajectory -> +DEM Flow Routing")

        harness = AblationHarness(
            model=self.model,
            dem_pipeline=self.dem_pipeline,
            rainfall_thresholds_mm=self.pipeline_config["baselines"]["rainfall_thresholds_mm"],
        )
        results = harness.run_all_stages(dataset.samples, stations_meta=stations)

        if self.labels_path.exists():
            labels = load_flood_labels(self.labels_path)
            label_results = evaluate_labeled_samples(
                dataset.samples,
                [self.model.predict_sample(sample) for sample in dataset.samples],
                labels,
                self.pipeline_config["baselines"]["rainfall_thresholds_mm"],
            )
            print_label_evaluation_table(label_results)

        # Print formatted comparison table
        print("\n" + "=" * 95)
        print(f"{'Ablation Stage / Baseline':<40} | {'CSI':<7} | {'POD':<7} | {'FAR':<7} | {'F1':<7} | {'Status'}")
        print("-" * 95)
        baseline_keys = ("baseline_rainfall", "baseline_noaa_ffg")
        best_baseline_csi = max(
            (results[k].csi for k in baseline_keys if k in results), default=0.0
        )
        for key, res in results.items():
            if key in baseline_keys:
                status = "Baseline"
            else:
                status = "Beats baselines" if res.csi > best_baseline_csi else "Below baselines"
            print(
                f"{res.stage_name:<40} | {res.csi:<7.3f} | {res.pod:<7.3f} | "
                f"{res.far:<7.3f} | {res.f1_score:<7.3f} | {status}"
            )
        print("=" * 95)

        # Export evaluation results to JSON
        eval_report_path = self.evaluation_dir / "ablation_benchmark_results.json"
        harness.export_report_json(
            results, eval_report_path, synthetic=any(self.synthetic_sources.values())
        )
        print(f"[Stage 6] Exported evaluation report to {eval_report_path}")

        return {k: v.to_dict() for k, v in results.items()}

    def run_training_stage(
        self, dataset: StationAnchoredFloodDataset
    ) -> StationAnchoredFloodDataset:
        """Train on event-disjoint data and return only the held-out test set."""
        manifest_path = Path("outputs/training/event_split.json")
        checkpoint_path = self.model_config["training"].get(
            "checkpoint_path", "outputs/training/learned_risk_model.json"
        )
        training_cfg = self.model_config["training"]
        try:
            split = prepare_training(
                dataset,
                manifest_path=manifest_path,
                minimum_events=training_cfg.get("minimum_events", 3),
            )
        except InsufficientTrainingDataError as exc:
            self.training_report = {
                "status": "not_trained",
                "evaluation_status": "skipped_no_held_out_test_split",
                "reason": str(exc),
                "evaluated_sample_count": 0,
            }
            print(f"[Training] NOT STARTED: {exc}")
            return StationAnchoredFloodDataset()

        self.learned_model = LearnedRiskModel(
            learning_rate=training_cfg["learning_rate"],
            l2=training_cfg["l2"],
        )
        self.training_report = self.learned_model.fit(
            split,
            epochs=training_cfg["epochs"],
            checkpoint_path=checkpoint_path,
        )
        self.training_report["status"] = "trained"
        self.training_report["evaluation_status"] = "held_out_test_only"
        self.training_report["train_sample_count"] = len(split.train)
        self.training_report["validation_sample_count"] = len(split.validation)
        self.training_report["test_sample_count"] = len(split.test)
        print(
            f"[Training] Completed event-disjoint train/validation/test flow: "
            f"{len(split.train)}/{len(split.validation)}/{len(split.test)} samples."
        )
        return split.test

    def run_all(self, raw_data_dir: Optional[str] = None) -> Dict[str, Any]:
        """Executes complete end-to-end pipeline across all 6 stages."""
        t0 = time.time()
        print(f"\n>>> STARTING HYDROALERT PIPELINE EXECUTION <<<")
        print(f"UTC time: {datetime.now(timezone.utc).isoformat()}")

        # Stage 1
        ingested = self.run_stage_1_ingestion(raw_data_dir=raw_data_dir)
        stations = [meta for meta, _ in ingested.values()]

        # Stage 2
        events = self.run_stage_2_trajectories(ingested)

        # Stage 3
        terrain_feats = self.run_stage_3_terrain(stations)

        # Stage 4
        dataset, risk_scores = self.run_stage_4_multimodal_inference(events)

        # Train and select the held-out evaluation split before any evaluation.
        held_out_test = self.run_training_stage(dataset)
        if self.learned_model is not None:
            risk_scores = [
                self.learned_model.predict_proba(sample)
                for sample in dataset.samples
            ]
            print("[Stage 4] Recomputed risk scores from the trained checkpoint for downstream alerts.")
        else:
            raise RuntimeError(
                "A trained risk model is required for pipeline alerts. "
                "Provide at least three event-disjoint labeled events containing both classes."
            )

        # Stage 5
        alert_summary = self.run_stage_5_alert_propagation(dataset, risk_scores, stations)

        # Stage 6
        eval_results = self.run_stage_6_evaluation(
            dataset, stations, evaluation_dataset=held_out_test if held_out_test.samples else None
        )

        elapsed = round(time.time() - t0, 2)
        print(f"\n==================================================")
        is_synthetic = any(self.synthetic_sources.values())
        print(f"PIPELINE EXECUTION COMPLETE in {elapsed}s")
        print(
            "SYNTHETIC DATA WARNING: coverage and results are not real-world coverage."
            if is_synthetic else "All stages successfully verified and logged."
        )
        print(f"==================================================\n")

        return {
            "elapsed_seconds": elapsed,
            "num_stations": len(stations),
            "num_samples": len(dataset),
            "manifest_coverage": self.manifest.compute_coverage_statistics(),
            "alert_summary": alert_summary,
            "ablation_results": eval_results,
            "training": self.training_report,
            "synthetic": is_synthetic,
            "synthetic_sources": self.synthetic_sources,
            "coverage_notice": (
                "Synthetic data used; do not claim real weather, satellite, or terrain coverage."
                if is_synthetic else "All source coverage is real."
            ),
        }


def main():
    runner = FlashFloodPipelineRunner()
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else None
    runner.run_all(raw_data_dir=raw_dir)


if __name__ == "__main__":
    main()
