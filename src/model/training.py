"""Event-disjoint training contract for HydroAlert.

The current checkout contains one labeled storm episode. This module prevents
that episode from being mislabeled as a trained general-purpose model. It
validates a multi-event corpus before a learned checkpoint may be created.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.model.dataset import MultimodalEventSample, StationAnchoredFloodDataset


class InsufficientTrainingDataError(ValueError):
    """Raised when event-disjoint training cannot be scientifically supported."""


@dataclass(frozen=True)
class EventSplit:
    train: StationAnchoredFloodDataset
    validation: StationAnchoredFloodDataset
    test: StationAnchoredFloodDataset
    event_names: Dict[str, List[str]]


def _event_name(sample: MultimodalEventSample) -> str:
    return sample.timestamp_utc[:10]


def split_by_event(
    samples: Sequence[MultimodalEventSample],
    minimum_events: int = 3,
) -> EventSplit:
    """Split complete storm dates so no event appears in multiple partitions."""
    grouped: Dict[str, List[MultimodalEventSample]] = {}
    for sample in samples:
        grouped.setdefault(_event_name(sample), []).append(sample)
    event_names = sorted(grouped)
    if len(event_names) < minimum_events:
        raise InsufficientTrainingDataError(
            f"Need at least {minimum_events} distinct labeled storm events; "
            f"found {len(event_names)} ({', '.join(event_names) or 'none'}). "
            "Add additional GHCNh, GOES, and gauge-label event windows before training."
        )
    labels = [sample.target_flood_label for sample in samples]
    if 1 not in labels or 0 not in labels:
        raise InsufficientTrainingDataError(
            "Training data must contain both flood and non-flood labels."
        )
    train_events = event_names[: max(1, len(event_names) - 2)]
    validation_events = [event_names[-2]]
    test_events = [event_names[-1]]
    by_name = lambda names: StationAnchoredFloodDataset(
        [sample for name in names for sample in grouped[name]]
    )
    split = EventSplit(
        train=by_name(train_events),
        validation=by_name(validation_events),
        test=by_name(test_events),
        event_names={
            "train": train_events,
            "validation": validation_events,
            "test": test_events,
        },
    )
    for name, dataset in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if not dataset.samples:
            raise InsufficientTrainingDataError(f"{name} split is empty.")
    return split


def save_training_manifest(
    split: EventSplit,
    destination: str | Path,
) -> None:
    """Persist the event split without serializing large raster/image tensors."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_names": split.event_names,
        "counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "status": "ready_for_model_training",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_training(
    dataset: StationAnchoredFloodDataset,
    manifest_path: str | Path = "outputs/training/event_split.json",
) -> EventSplit:
    """Validate and record a scientifically defensible event split."""
    split = split_by_event(dataset.samples)
    save_training_manifest(split, manifest_path)
    return split
