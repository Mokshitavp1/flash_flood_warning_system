"""Regression tests for station/timestamp label joining."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.evaluation.label_evaluation import labeled_sample_target, load_flood_labels


class TestLabelJoining(unittest.TestCase):
    def test_labels_are_keyed_by_station_and_normalized_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "station_id",
                        "timestamp_utc",
                        "flood_label",
                        "hours_to_onset",
                        "nws_warning_active",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "station_id": "STATION_A",
                            "timestamp_utc": "2025-07-04T01:00:00Z",
                            "flood_label": "0",
                            "hours_to_onset": "2",
                            "nws_warning_active": "1",
                        },
                        {
                            "station_id": "STATION_B",
                            "timestamp_utc": "2025-07-04T01:00:00Z",
                            "flood_label": "0",
                            "hours_to_onset": "3",
                            "nws_warning_active": "1",
                        },
                    ]
                )

            labels = load_flood_labels(path)
            self.assertEqual(
                labeled_sample_target(labels, "STATION_A", "2025-07-04T01:00:00+00:00"),
                1,
            )
            self.assertEqual(
                labeled_sample_target(labels, "STATION_B", "2025-07-04T01:00:00+00:00"),
                1,
            )
            self.assertIsNone(
                labeled_sample_target(labels, "STATION_C", "2025-07-04T01:00:00+00:00")
            )


if __name__ == "__main__":
    unittest.main()
