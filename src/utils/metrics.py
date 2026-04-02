from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class PredictionRecord:
    video_id: str
    is_positive: bool
    ground_truth_start_s: float | None
    predicted_start_s: float | None


@dataclass(slots=True)
class ContestMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


def compute_contest_metrics(
    records: Iterable[PredictionRecord],
    early_tolerance_s: float = 1.0,
    late_tolerance_s: float = 30.0,
) -> ContestMetrics:
    tp = 0
    fp = 0
    fn = 0

    for record in records:
        prediction = record.predicted_start_s
        target = record.ground_truth_start_s
        if record.is_positive:
            if prediction is None:
                fn += 1
                continue
            assert target is not None
            if (target - early_tolerance_s) <= prediction <= (target + late_tolerance_s):
                tp += 1
            else:
                fp += 1
        else:
            if prediction is not None:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return ContestMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )
