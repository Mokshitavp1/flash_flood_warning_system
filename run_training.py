#!/usr/bin/env python3
"""Train and evaluate the dependency-free learned risk backend."""

import sys

from src.pipeline_runner import FlashFloodPipelineRunner
from src.model.training import InsufficientTrainingDataError, LearnedRiskModel, prepare_training


def main() -> int:
    try:
        runner = FlashFloodPipelineRunner()
        station_data = runner.run_stage_1_ingestion("data")
        events = runner.run_stage_2_trajectories(station_data)
        runner.run_stage_3_terrain([meta for meta, _ in station_data.values()])
        dataset, _ = runner.run_stage_4_multimodal_inference(events)
        split = prepare_training(dataset)
        checkpoint_path = "outputs/training/learned_risk_model.json"
        report = LearnedRiskModel().fit(split, checkpoint_path=checkpoint_path)
        reloaded = LearnedRiskModel.load_checkpoint(checkpoint_path)
        print(f"TRAINING COMPLETE: {report['event_names']}")
        print(f"HELD-OUT TEST: {reloaded.evaluate(split.test).to_dict()}")
    except InsufficientTrainingDataError as exc:
        print(f"TRAINING NOT STARTED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
