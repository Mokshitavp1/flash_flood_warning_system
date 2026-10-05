"""Hydrometeorological Skill Metrics and Verification Statistics.

Implements Functional Requirement FR6.3:
- Critical Success Index (CSI / Threat Score)
- Probability of Detection (POD / Hit Rate)
- False Alarm Ratio (FAR)
- Frequency Bias (FBIAS)
- Heidke Skill Score (HSS)
- F1-Score, Precision, and Recall
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class BinaryClassificationMetrics:
    """Standard 2x2 contingency table metrics for flood warning events."""
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int
    total: int
    pod: float  # Probability of Detection = Hits / (Hits + Misses)
    far: float  # False Alarm Ratio = FA / (Hits + FA)
    csi: float  # Critical Success Index = Hits / (Hits + Misses + FA)
    fbias: float  # Frequency Bias = (Hits + FA) / (Hits + Misses)
    hss: float  # Heidke Skill Score
    f1_score: float
    precision: float
    recall: float
    accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SkillScoreEvaluator:
    """Calculates verification metrics on predicted vs target labels."""

    @staticmethod
    def evaluate(
        predictions_binary: List[int], ground_truth: List[int]
    ) -> BinaryClassificationMetrics:
        """Computes full contingency table and skill metrics."""
        hits = 0
        misses = 0
        false_alarms = 0
        correct_negatives = 0

        for pred, true in zip(predictions_binary, ground_truth):
            p = 1 if pred >= 1 else 0
            t = 1 if true >= 1 else 0

            if p == 1 and t == 1:
                hits += 1
            elif p == 0 and t == 1:
                misses += 1
            elif p == 1 and t == 0:
                false_alarms += 1
            else:
                correct_negatives += 1

        total = hits + misses + false_alarms + correct_negatives

        # 1. Probability of Detection (POD / Hit Rate)
        pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0

        # 2. False Alarm Ratio (FAR)
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0

        # 3. Critical Success Index (CSI / Threat Score)
        csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0

        # 4. Frequency Bias (FBIAS)
        fbias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else 0.0

        # 5. Heidke Skill Score (HSS)
        # Expected correct by chance: E = ((H+M)*(H+FA) + (CN+M)*(CN+FA)) / Total
        e_numer = (hits + misses) * (hits + false_alarms) + (correct_negatives + misses) * (correct_negatives + false_alarms)
        denom_hss = (total * total - e_numer)
        numer_hss = (total * (hits + correct_negatives) - e_numer)
        hss = (numer_hss / denom_hss) if denom_hss > 0 else 0.0

        # 6. Precision, Recall, F1
        precision = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
        recall = pod
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (hits + correct_negatives) / total if total > 0 else 0.0

        return BinaryClassificationMetrics(
            hits=hits,
            misses=misses,
            false_alarms=false_alarms,
            correct_negatives=correct_negatives,
            total=total,
            pod=round(pod, 4),
            far=round(far, 4),
            csi=round(csi, 4),
            fbias=round(fbias, 4),
            hss=round(hss, 4),
            f1_score=round(f1, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            accuracy=round(accuracy, 4),
        )
