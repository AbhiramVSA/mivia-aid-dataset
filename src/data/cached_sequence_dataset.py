from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class CachedSequenceSample:
    video_id: str
    features: torch.Tensor
    timestamps_s: torch.Tensor
    step_targets: torch.Tensor
    onset_targets: torch.Tensor
    step_mask: torch.Tensor
    video_target: torch.Tensor
    source_positive: torch.Tensor


@dataclass(slots=True)
class CachedSequenceBatch:
    video_ids: list[str]
    features: torch.Tensor
    timestamps_s: torch.Tensor
    step_targets: torch.Tensor
    onset_targets: torch.Tensor
    step_mask: torch.Tensor
    video_target: torch.Tensor
    source_positive: torch.Tensor


@dataclass(slots=True)
class CachedWindowRecord:
    cache_path: Path
    step_start: int
    step_end: int


@dataclass(slots=True)
class CachedAnnotationMeta:
    video_id: str
    is_positive: bool
    start_s: float | None


def load_feature_cache(cache_path: Path) -> dict:
    return torch.load(cache_path, map_location="cpu", weights_only=False)


class CachedSequenceDataset(Dataset[CachedSequenceSample]):
    def __init__(self, cache_dir: Path, max_steps: int | None = None, window_stride: int | None = None) -> None:
        self.cache_dir = cache_dir
        self.max_steps = max_steps
        self.window_stride = window_stride or max_steps
        self.cache_paths = sorted(cache_dir.glob("*.pt"))
        self.annotation_metas: list[CachedAnnotationMeta] = []
        self.records = self._build_window_records()
        self.total_window_steps = sum(record.step_end - record.step_start for record in self.records)
        self.max_window_steps = max((record.step_end - record.step_start for record in self.records), default=0)
        self.sample_weights = self._build_sample_weights()

    def _build_window_records(self) -> list[CachedWindowRecord]:
        records: list[CachedWindowRecord] = []
        for cache_path in self.cache_paths:
            bundle = load_feature_cache(cache_path)
            self.annotation_metas.append(
                CachedAnnotationMeta(
                    video_id=str(bundle["video_id"]),
                    is_positive=bool(bundle["source_positive"].item() > 0.5),
                    start_s=float(bundle["ground_truth_start_s"]) if bundle["ground_truth_start_s"] is not None else None,
                )
            )
            total_steps = int(bundle["features"].shape[0])
            if total_steps == 0:
                continue
            if self.max_steps is None or total_steps <= self.max_steps:
                records.append(CachedWindowRecord(cache_path=cache_path, step_start=0, step_end=total_steps))
                continue
            assert self.window_stride is not None
            start = 0
            while start < total_steps:
                end = min(total_steps, start + self.max_steps)
                records.append(CachedWindowRecord(cache_path=cache_path, step_start=start, step_end=end))
                if end >= total_steps:
                    break
                start += self.window_stride
        return records

    def _build_sample_weights(self) -> torch.Tensor:
        if not self.records:
            return torch.zeros((0,), dtype=torch.float32)
        windows_per_video: dict[Path, int] = {}
        for record in self.records:
            windows_per_video[record.cache_path] = windows_per_video.get(record.cache_path, 0) + 1
        return torch.tensor(
            [1.0 / float(windows_per_video[record.cache_path]) for record in self.records],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def num_videos(self) -> int:
        return len(self.cache_paths)

    @property
    def avg_window_steps(self) -> float:
        if not self.records:
            return 0.0
        return self.total_window_steps / float(len(self.records))

    def __getitem__(self, index: int) -> CachedSequenceSample:
        record = self.records[index]
        bundle = load_feature_cache(record.cache_path)
        sl = slice(record.step_start, record.step_end)
        timestamps_s = bundle["timestamps_s"][sl]
        features = bundle["features"][sl]
        step_targets = bundle["step_targets"][sl]
        onset_targets = bundle["onset_targets"][sl]
        step_mask = torch.ones(features.shape[0], dtype=torch.bool)
        return CachedSequenceSample(
            video_id=str(bundle["video_id"]),
            features=features,
            timestamps_s=timestamps_s,
            step_targets=step_targets,
            onset_targets=onset_targets,
            step_mask=step_mask,
            video_target=bundle["video_target"].clone(),
            source_positive=bundle["source_positive"].clone(),
        )


def collate_cached_sequence_batch(samples: list[CachedSequenceSample]) -> CachedSequenceBatch:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(samples)
    max_steps = max(sample.features.shape[0] for sample in samples)
    hidden_size = samples[0].features.shape[-1]

    features = torch.zeros((batch_size, max_steps, hidden_size), dtype=samples[0].features.dtype)
    timestamps_s = torch.zeros((batch_size, max_steps), dtype=torch.float32)
    step_targets = torch.zeros((batch_size, max_steps), dtype=torch.float32)
    onset_targets = torch.zeros((batch_size, max_steps), dtype=torch.float32)
    step_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    video_target = torch.stack([sample.video_target for sample in samples], dim=0)
    source_positive = torch.stack([sample.source_positive for sample in samples], dim=0)
    video_ids: list[str] = []

    for batch_index, sample in enumerate(samples):
        num_steps = sample.features.shape[0]
        features[batch_index, :num_steps] = sample.features
        timestamps_s[batch_index, :num_steps] = sample.timestamps_s
        step_targets[batch_index, :num_steps] = sample.step_targets
        onset_targets[batch_index, :num_steps] = sample.onset_targets
        step_mask[batch_index, :num_steps] = sample.step_mask
        video_ids.append(sample.video_id)

    return CachedSequenceBatch(
        video_ids=video_ids,
        features=features,
        timestamps_s=timestamps_s,
        step_targets=step_targets,
        onset_targets=onset_targets,
        step_mask=step_mask,
        video_target=video_target,
        source_positive=source_positive,
    )
