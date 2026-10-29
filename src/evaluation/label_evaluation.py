"""Evaluation against station flood labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.evaluation.baselines import BaselineRainfallThreshold
from src.evaluation.metrics import BinaryClassificationMetrics, SkillScoreEvaluator
from src.model.dataset import MultimodalEventSample


@dataclass(frozen=True)
class FloodLabel:
    flood_label: Optional[int]
    hours_to_onset: Optional[int]
    nws_warning_active: Optional[int]


def _parse_optional_int(value: str) -> Optional[int]:
    return int(value) if value != "" else None


def load_flood_labels(path: str | Path) -> Dict[tuple[str, str], FloodLabel]:
    """Load labels keyed by station and normalized UTC sample timestamp."""
    labels: Dict[tuple[str, str], FloodLabel] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
            timestamp = timestamp.astimezone(timezone.utc).isoformat()
            labels[(row["station_id"], timestamp)] = FloodLabel(
                flood_label=_parse_optional_int(row["flood_label"]),
                hours_to_onset=_parse_optional_int(row["hours_to_onset"]),
                nws_warning_active=_parse_optional_int(row["nws_warning_active"]),
            )
    return labels


def target_for_label(label: FloodLabel) -> Optional[int]:
    """Return the forecast target, excluding flooding, post-flood, and blank hours."""
    if label.flood_label != 0 or label.hours_to_onset is None:
        return None
    return int(1 <= label.hours_to_onset <= 3)


def labeled_sample_target(
    labels: Dict[tuple[str, str], FloodLabel],
    station_id: str,
    timestamp_utc: str,
) -> Optional[int]:
    label = labels.get((station_id, timestamp_utc))
    return None if label is None else target_for_label(label)


def _metric_row(
    name: str,
    predictions: Sequence[int],
    targets: Sequence[int],
    timestamps: Sequence[str],
    onset_timestamp: str,
) -> Dict[str, object]:
    metrics: BinaryClassificationMetrics = SkillScoreEvaluator.evaluate(
        list(predictions), list(targets)
    )
    alert_times = [
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        for ts, prediction in zip(timestamps, predictions)
        if prediction
    ]
    onset = datetime.fromisoformat(onset_timestamp.replace("Z", "+00:00"))
    lead_time = None
    if alert_times:
        lead_time = round((onset - min(alert_times)).total_seconds() / 3600.0, 2)
    return {
        "method": name,
        "pod": metrics.pod,
        "far": metrics.far,
        "csi": metrics.csi,
        "lead_time_hours": lead_time,
    }


def evaluate_labeled_samples(
    samples: Sequence[MultimodalEventSample],
    risk_scores: Sequence[float],
    labels: Dict[tuple[str, str], FloodLabel],
    rainfall_thresholds_mm: Dict[str, float],
    model_threshold: float = 0.65,
    onset_timestamp: str = "2025-07-04T08:00:00+00:00",
) -> List[Dict[str, object]]:
    """Evaluate warnings on samples with valid Kerrville pre-onset labels."""
    evaluated = []
    for sample, score in zip(samples, risk_scores):
        label = labels.get((sample.station_id, sample.timestamp_utc))
        if label is None or target_for_label(label) is None:
            continue
        evaluated.append((sample, score, label))

    targets = [target_for_label(label) for _, _, label in evaluated]
    timestamps = [sample.timestamp_utc for sample, _, _ in evaluated]
    results = [
        _metric_row(
            "NWS warning",
            [label.nws_warning_active or 0 for _, _, label in evaluated],
            targets,
            timestamps,
            onset_timestamp,
        )
    ]
    for key, threshold in rainfall_thresholds_mm.items():
        rainfall_attr = {
            "one_hour": "rainfall_1h_mm",
            "three_hour": "rainfall_3h_mm",
            "six_hour": "rainfall_6h_mm",
        }[key]
        results.append(
            _metric_row(
                f"Rainfall {key.replace('_', '-')}",
                [
                    int(getattr(sample, rainfall_attr) >= threshold)
                    for sample, _, _ in evaluated
                ],
                targets,
                timestamps,
                onset_timestamp,
            )
        )
    results.append(
        _metric_row(
            f"Model (threshold {model_threshold:.2f})",
            [int(score >= model_threshold) for _, score, _ in evaluated],
            targets,
            timestamps,
            onset_timestamp,
        )
    )
    return results


def print_label_evaluation_table(results: Iterable[Dict[str, object]]) -> None:
    print("\nLabel evaluation (Kerrville, July 3-5, 2025)")
    print("| Method | POD | FAR | CSI | Lead time (h) |")
    print("|---|---:|---:|---:|---:|")
    for result in results:
        lead = "N/A" if result["lead_time_hours"] is None else f'{result["lead_time_hours"]:.2f}'
        print(
            f'| {result["method"]} | {result["pod"]:.3f} | '
            f'{result["far"]:.3f} | {result["csi"]:.3f} | {lead} |'
        )
