"""Event-disjoint training and learned checkpoint support for HydroAlert.

The pure-Python multimodal network is an inference prototype with deterministic
weights. This module provides a real, dependency-free SGD backend that learns a
regularized logistic risk head from multimodal sample features. It enforces
event-disjoint splits before training and persists learned checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.model.dataset import MultimodalEventSample, StationAnchoredFloodDataset
from src.evaluation.metrics import BinaryClassificationMetrics, SkillScoreEvaluator


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
    """Split complete storm dates without leaking events across partitions.

    With exactly two events, this is an explicitly limited demonstration split:
    the first event is training data and the second is held-out test data.
    """
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
    if len(event_names) == 2:
        train_events = [event_names[0]]
        validation_events = []
        test_events = [event_names[1]]
    else:
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
        if name != "validation" and not dataset.samples:
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
    minimum_events: int = 3,
) -> EventSplit:
    """Validate and record a scientifically defensible event split."""
    split = split_by_event(dataset.samples, minimum_events=minimum_events)
    save_training_manifest(split, manifest_path)
    return split


FEATURE_NAMES = [
    "tabular_mean_precipitation_1h",
    "tabular_mean_precipitation_3h",
    "tabular_mean_temperature",
    "tabular_mean_relative_humidity",
    "tabular_mean_station_pressure",
    "tabular_mean_wind_speed",
    "terrain_relative_height",
    "terrain_slope",
    "terrain_aspect",
    "terrain_flow_accumulation",
    "terrain_elevated_indicator",
    "satellite_valid_frame_mean",
]


def _sample_features(sample: MultimodalEventSample) -> List[float]:
    """Create a fixed-size feature vector from the canonical sample schema."""
    tabular = sample.tabular_features
    tabular_means = [
        sum(row[index] for row in tabular) / max(1, len(tabular))
        for index in range(6)
    ] if tabular else [0.0] * 6
    valid_frames = [
        frame
        for frame, valid in zip(sample.trajectory_patches, sample.trajectory_mask)
        if valid
    ]
    satellite_mean = (
        sum(
            sum(pixel for row in channel for pixel in row)
            / max(1, sum(len(row) for row in channel))
            for frame in valid_frames
            for channel in frame
        )
        / max(1, len(valid_frames) * len(valid_frames[0]))
        if valid_frames
        else 0.0
    )
    return [round(value, 6) for value in tabular_means + sample.terrain_features + [satellite_mean]]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


@dataclass
class HeldOutEvaluation:
    """Probabilistic and thresholded metrics for a held-out event split."""

    loss: float
    threshold: float
    metrics: BinaryClassificationMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": round(self.loss, 6),
            "threshold": self.threshold,
            "metrics": self.metrics.to_dict(),
        }


class LearnedRiskModel:
    """Trainable logistic risk head with deterministic JSON checkpoints."""

    backend_name = "pure_python_logistic_sgd"
    checkpoint_version = 1

    def __init__(
        self,
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
        learning_rate: float = 0.05,
        l2: float = 1e-4,
    ):
        self.weights = list(weights) if weights is not None else [0.0] * len(FEATURE_NAMES)
        if len(self.weights) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} checkpoint weights")
        self.bias = bias
        self.learning_rate = learning_rate
        self.l2 = l2

    def predict_proba(self, sample: MultimodalEventSample) -> float:
        features = _sample_features(sample)
        return round(_sigmoid(self.bias + sum(w * x for w, x in zip(self.weights, features))), 6)

    def _loss(self, dataset: StationAnchoredFloodDataset) -> float:
        if not dataset.samples:
            return 0.0
        losses = []
        for sample in dataset.samples:
            probability = max(1e-7, min(1.0 - 1e-7, self.predict_proba(sample)))
            target = sample.target_flood_label
            losses.append(-target * math.log(probability) - (1 - target) * math.log(1 - probability))
        return sum(losses) / len(losses)

    def evaluate(
        self,
        dataset: StationAnchoredFloodDataset,
        threshold: float = 0.5,
    ) -> HeldOutEvaluation:
        probabilities = [self.predict_proba(sample) for sample in dataset.samples]
        targets = [sample.target_flood_label for sample in dataset.samples]
        predictions = [int(probability >= threshold) for probability in probabilities]
        return HeldOutEvaluation(
            loss=round(self._loss(dataset), 6),
            threshold=threshold,
            metrics=SkillScoreEvaluator.evaluate(predictions, targets),
        )

    def fit(
        self,
        split: EventSplit,
        epochs: int = 100,
        checkpoint_path: str | Path = "outputs/training/learned_risk_model.json",
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Fit with SGD, select the best validation epoch, save, and test once."""
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if not split.train.samples or not split.test.samples:
            raise InsufficientTrainingDataError("Train and test splits must both be non-empty.")

        best_validation_loss = float("inf")
        best_state = (self.weights[:], self.bias)
        history: List[Dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            for sample in split.train.samples:
                features = _sample_features(sample)
                probability = _sigmoid(self.bias + sum(w * x for w, x in zip(self.weights, features)))
                error = probability - sample.target_flood_label
                for index, feature in enumerate(features):
                    self.weights[index] -= self.learning_rate * (error * feature + self.l2 * self.weights[index])
                self.bias -= self.learning_rate * error
            validation = self.evaluate(split.validation, threshold) if split.validation.samples else None
            history_entry = {"epoch": epoch, "train_loss": self._loss(split.train)}
            if validation is not None:
                history_entry["validation_loss"] = validation.loss
                if validation.loss < best_validation_loss:
                    best_validation_loss = validation.loss
                    best_state = (self.weights[:], self.bias)
            else:
                best_state = (self.weights[:], self.bias)
            history.append(history_entry)

        self.weights, self.bias = best_state
        self.save_checkpoint(checkpoint_path)
        return {
            "backend": self.backend_name,
            "epochs": epochs,
            "history": history,
            "validation": self.evaluate(split.validation, threshold).to_dict() if split.validation.samples else None,
            "held_out_test": self.evaluate(split.test, threshold).to_dict(),
            "checkpoint": str(checkpoint_path),
            "event_names": split.event_names,
        }

    def save_checkpoint(self, destination: str | Path) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": self.checkpoint_version,
            "backend": self.backend_name,
            "feature_names": FEATURE_NAMES,
            "weights": self.weights,
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, source: str | Path) -> "LearnedRiskModel":
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        if payload.get("checkpoint_version") != cls.checkpoint_version:
            raise ValueError("Unsupported learned risk model checkpoint version")
        if payload.get("backend") != cls.backend_name or payload.get("feature_names") != FEATURE_NAMES:
            raise ValueError("Checkpoint backend or feature schema does not match the training backend")
        return cls(
            weights=payload["weights"],
            bias=payload["bias"],
            learning_rate=payload.get("learning_rate", 0.05),
            l2=payload.get("l2", 1e-4),
        )
