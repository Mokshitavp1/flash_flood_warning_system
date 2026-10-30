#!/usr/bin/env python3
"""Validate the event-disjoint training contract for a prepared dataset."""

import sys

from src.pipeline_runner import FlashFloodPipelineRunner
from src.model.training import InsufficientTrainingDataError, prepare_training


def main() -> int:
    try:
        runner = FlashFloodPipelineRunner()
        station_data = runner.run_stage_1_ingestion("data")
        events = runner.run_stage_2_trajectories(station_data)
        runner.run_stage_3_terrain([meta for meta, _ in station_data.values()])
        dataset, _ = runner.run_stage_4_multimodal_inference(events)
        split = prepare_training(dataset)
        print(f"TRAINING DATA READY: {split.event_names}")
    except InsufficientTrainingDataError as exc:
        print(f"TRAINING NOT STARTED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
