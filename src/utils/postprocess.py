from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class PostprocessResult:
    predicted_start_s: float | None
    max_score: float
    video_score: float


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
    tau_keep: float | None = None,
    tau_video: float = 0.0,
    video_score: float | None = None,
    median_kernel_size: int = 3,
    min_consecutive_steps: int = 1,
    mode: str = "cumulative",
) -> PostprocessResult:
    if len(step_scores) != len(timestamps_s):
        raise ValueError("step_scores and timestamps_s must have the same length")
    effective_video_score = 1.0 if video_score is None else float(video_score)
    if not step_scores:
        return PostprocessResult(predicted_start_s=None, max_score=0.0, video_score=effective_video_score)

    smoothed_scores = median_filter_1d(step_scores, kernel_size=median_kernel_size)
    max_score = max(smoothed_scores)
    if effective_video_score < tau_video or max_score < tau_empty:
        return PostprocessResult(predicted_start_s=None, max_score=max_score, video_score=effective_video_score)

    if mode == "peak":
        if max_score < tau_start:
            return PostprocessResult(predicted_start_s=None, max_score=max_score, video_score=effective_video_score)
        peak_index = max(range(len(smoothed_scores)), key=lambda idx: smoothed_scores[idx])
        return PostprocessResult(
            predicted_start_s=timestamps_s[peak_index],
            max_score=max_score,
            video_score=effective_video_score,
        )

    active_count = 0
    continuation_threshold = tau_start if tau_keep is None else min(tau_start, tau_keep)
    for index, (score, timestamp) in enumerate(zip(smoothed_scores, timestamps_s)):
        threshold = tau_start if active_count == 0 else continuation_threshold
        if score >= threshold:
            active_count += 1
            if active_count >= max(1, min_consecutive_steps):
                onset_index = max(0, index - active_count + 1)
                return PostprocessResult(
                    predicted_start_s=timestamps_s[onset_index],
                    max_score=max_score,
                    video_score=effective_video_score,
                )
        else:
            active_count = 0
    return PostprocessResult(predicted_start_s=None, max_score=max_score, video_score=effective_video_score)
