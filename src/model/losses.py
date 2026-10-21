"""Loss functions for class-imbalanced flash-flood early warning."""

from __future__ import annotations

import math
from typing import List


class FocalLoss:
    """
    Focal Loss for binary classification on rare event datasets (FR4.4).
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    Down-weights easy negatives and focuses training on rare flash-flood events.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, eps: float = 1e-7):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def compute(self, predictions: List[float], targets: List[float]) -> float:
        total_loss = 0.0
        n = max(1, len(predictions))

        for p_raw, y in zip(predictions, targets):
            p = max(self.eps, min(1.0 - self.eps, p_raw))
            if y > 0.5:
                # Positive class (flash flood)
                weight = self.alpha * ((1.0 - p) ** self.gamma)
                loss = -weight * math.log(p)
            else:
                # Negative class (no flood)
                weight = (1.0 - self.alpha) * (p ** self.gamma)
                loss = -weight * math.log(1.0 - p)
            total_loss += loss

        return round(total_loss / n, 6)


class WeightedBCELoss:
    """Weighted Binary Cross-Entropy with positive class weighting."""

    def __init__(self, pos_weight: float = 8.0, eps: float = 1e-7):
        self.pos_weight = pos_weight
        self.eps = eps

    def compute(self, predictions: List[float], targets: List[float]) -> float:
        total_loss = 0.0
        n = max(1, len(predictions))

        for p_raw, y in zip(predictions, targets):
            p = max(self.eps, min(1.0 - self.eps, p_raw))
            if y > 0.5:
                loss = -self.pos_weight * math.log(p)
            else:
                loss = -math.log(1.0 - p)
            total_loss += loss

        return round(total_loss / n, 6)
