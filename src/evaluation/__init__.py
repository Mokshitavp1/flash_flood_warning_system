"""Evaluation metrics, baseline comparisons, and multi-stage ablation harness."""

from src.evaluation.ablation import AblationHarness, AblationStageResult
from src.evaluation.baselines import (
    BaselineRainfallThreshold,
    NOAAFlashFloodGuidanceBaseline,
)
from src.evaluation.metrics import BinaryClassificationMetrics, SkillScoreEvaluator

__all__ = [
    "AblationHarness",
    "AblationStageResult",
    "BaselineRainfallThreshold",
    "NOAAFlashFloodGuidanceBaseline",
    "BinaryClassificationMetrics",
    "SkillScoreEvaluator",
]
