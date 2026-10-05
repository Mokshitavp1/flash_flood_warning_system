#!/usr/bin/env python3
"""Root CLI entrypoint for HydroAlert pipeline execution."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline_runner import FlashFloodPipelineRunner


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else None
    runner = FlashFloodPipelineRunner()
    runner.run_all(raw_data_dir=raw_dir)


if __name__ == "__main__":
    main()
