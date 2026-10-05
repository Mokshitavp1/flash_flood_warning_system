"""Operational Benchmark Baselines for Flash-Flood Warning Comparison.

Implements Functional Requirements:
- FR6.1: Rainfall-threshold-only baseline (e.g. >25 mm/h or >45 mm/3h).
- FR6.2: NOAA Flash Flood Guidance (FFG) baseline.
"""

from __future__ import annotations

from typing import List

from src.evaluation.metrics import BinaryClassificationMetrics, SkillScoreEvaluator
from src.model.dataset import MultimodalEventSample


class BaselineRainfallThreshold:
    """
    FR6.1: Rainfall-threshold-only baseline.
    Triggers an alert solely when local ground station rainfall exceeds fixed thresholds:
    - 1-hour rainfall >= 25.0 mm OR
    - 3-hour rainfall >= 45.0 mm
    
    LIMITATION: Completely blindsided by flash floods in downhill communities
    where local rainfall is 0.0 mm but floodwaters rush down from uphill peaks!
    """

    def __init__(self, threshold_1h_mm: float = 25.0, threshold_3h_mm: float = 45.0):
        self.threshold_1h_mm = threshold_1h_mm
        self.threshold_3h_mm = threshold_3h_mm

    def predict(self, sample: MultimodalEventSample) -> int:
        if sample.rainfall_1h_mm >= self.threshold_1h_mm:
            return 1
        if sample.rainfall_3h_mm >= self.threshold_3h_mm:
            return 1
        return 0

    def evaluate(self, samples: List[MultimodalEventSample]) -> BinaryClassificationMetrics:
        preds = [self.predict(s) for s in samples]
        targets = [s.target_flood_label for s in samples]
        return SkillScoreEvaluator.evaluate(preds, targets)


class NOAAFlashFloodGuidanceBaseline:
    """
    FR6.2: NOAA Flash Flood Guidance (FFG) Operational Baseline.
    Simulates the NWS FFG threshold system:
    - Compares rainfall to static hydrological basin threshold (typically ~35mm/1h or 50mm/3h).
    - FFG operates on coarse regional scale and does not track fine-grained storm cloud
      convective trajectories or dynamic downhill flow paths.
    - Documented skill scores: CSI typically 0.15–0.38, high FAR (0.45–0.70).
    """

    def __init__(self, ffg_1h_threshold_mm: float = 32.0, ffg_3h_threshold_mm: float = 48.0):
        self.ffg_1h_threshold_mm = ffg_1h_threshold_mm
        self.ffg_3h_threshold_mm = ffg_3h_threshold_mm

    def predict(self, sample: MultimodalEventSample) -> int:
        # FFG triggers if either 1h or 3h exceeds county/basin FFG value
        if sample.rainfall_1h_mm >= self.ffg_1h_threshold_mm:
            return 1
        if sample.rainfall_3h_mm >= self.ffg_3h_threshold_mm:
            return 1
        return 0

    def evaluate(self, samples: List[MultimodalEventSample]) -> BinaryClassificationMetrics:
        preds = [self.predict(s) for s in samples]
        targets = [s.target_flood_label for s in samples]
        return SkillScoreEvaluator.evaluate(preds, targets)
