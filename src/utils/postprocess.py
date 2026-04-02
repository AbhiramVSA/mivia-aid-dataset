from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class PostprocessResult:
    predicted_start_s: float | None
    max_score: float


def median_filter_1d(values: Iterable[float], kernel_size: int = 3) -> list[float]:
    sequence = list(values)
    if kernel_size <= 1 or len(sequence) <= 1:
        return sequence
    radius = kernel_size // 2
    filtered: list[float] = []
    for index in range(len(sequence)):
        start = max(0, index - radius)
        end = min(len(sequence), index + radius + 1)
        window = sorted(sequence[start:end])
        filtered.append(window[len(window) // 2])
    return filtered


def predict_start_time(
    step_scores: list[float],
    timestamps_s: list[float],
    tau_empty: float,
    tau_start: float,
    median_kernel_size: int = 3,
) -> PostprocessResult:
    if len(step_scores) != len(timestamps_s):
        raise ValueError("step_scores and timestamps_s must have the same length")
    if not step_scores:
        return PostprocessResult(predicted_start_s=None, max_score=0.0)

    smoothed_scores = median_filter_1d(step_scores, kernel_size=median_kernel_size)
    max_score = max(smoothed_scores)
    if max_score < tau_empty:
        return PostprocessResult(predicted_start_s=None, max_score=max_score)
    for score, timestamp in zip(smoothed_scores, timestamps_s):
        if score >= tau_start:
            return PostprocessResult(predicted_start_s=timestamp, max_score=max_score)
    return PostprocessResult(predicted_start_s=None, max_score=max_score)
