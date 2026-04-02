from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.config import VideoSamplingConfig
from src.data.annotations import VideoAnnotation
from src.data.video_decode import (
    build_causal_clip_indices,
    decode_sampled_frame_window,
    extract_causal_clip,
    infer_sampled_frame_count,
    preprocess_clip_batch,
)


@dataclass(slots=True)
class SequenceSample:
    video_id: str
    clip_tensor: torch.Tensor
    timestamps_s: torch.Tensor
    step_targets: torch.Tensor
    step_mask: torch.Tensor
    video_target: torch.Tensor
    source_positive: torch.Tensor


@dataclass(slots=True)
class SequenceBatch:
    video_ids: list[str]
    clip_tensor: torch.Tensor
    timestamps_s: torch.Tensor
    step_targets: torch.Tensor
    step_mask: torch.Tensor
    video_target: torch.Tensor
    source_positive: torch.Tensor


@dataclass(slots=True)
class SequenceWindowRecord:
    annotation_index: int
    step_start: int
    step_end: int


class Stage2SequenceDataset(Dataset[SequenceSample]):
    """Per-video sequence dataset for the temporal fine-tuning stage."""

    def __init__(
        self,
        annotations: list[VideoAnnotation],
        sampling: VideoSamplingConfig,
        backbone_name: str,
        max_steps: int | None = None,
        window_stride: int | None = None,
    ) -> None:
        self.annotations = annotations
        self.sampling = sampling
        self.backbone_name = backbone_name
        self.max_steps = max_steps
        self.window_stride = window_stride or max_steps
        self.records = self._build_window_records()

    def _build_window_records(self) -> list[SequenceWindowRecord]:
        records: list[SequenceWindowRecord] = []
        for annotation_index, annotation in enumerate(self.annotations):
            num_frames = infer_sampled_frame_count(annotation.video_path, self.sampling.sample_fps)
            clip_spans = build_causal_clip_indices(num_frames, self.sampling)
            total_steps = len(clip_spans)
            if total_steps == 0:
                continue
            if self.max_steps is None or total_steps <= self.max_steps:
                records.append(SequenceWindowRecord(annotation_index=annotation_index, step_start=0, step_end=total_steps))
                continue
            assert self.window_stride is not None
            start = 0
            while start < total_steps:
                end = min(total_steps, start + self.max_steps)
                records.append(SequenceWindowRecord(annotation_index=annotation_index, step_start=start, step_end=end))
                if end >= total_steps:
                    break
                start += self.window_stride
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SequenceSample:
        record = self.records[index]
        annotation = self.annotations[record.annotation_index]
        total_sampled_frames = infer_sampled_frame_count(annotation.video_path, self.sampling.sample_fps)
        all_clip_spans = build_causal_clip_indices(total_sampled_frames, self.sampling)
        clip_spans = all_clip_spans[record.step_start:record.step_end]
        if not clip_spans:
            raise RuntimeError(f"No clip spans generated for {annotation.video_id} window {record}")

        first_end_idx = clip_spans[0][1]
        last_end_idx = clip_spans[-1][1]
        sample_start_idx = max(0, first_end_idx - self.sampling.clip_num_frames)
        sample_end_idx = last_end_idx
        frames, _ = decode_sampled_frame_window(
            annotation.video_path,
            self.sampling.sample_fps,
            start_sample_idx=sample_start_idx,
            end_sample_idx=sample_end_idx,
        )
        clips = [
            extract_causal_clip(
                frames,
                end_idx_exclusive=end_idx - sample_start_idx,
                clip_len=self.sampling.clip_num_frames,
            )
            for _, end_idx in clip_spans
        ]
        clip_tensor = preprocess_clip_batch(clips, backbone_name=self.backbone_name)
        timestamps_s = torch.tensor(
            [(end_idx - 1) / float(self.sampling.sample_fps) for _, end_idx in clip_spans],
            dtype=torch.float32,
        )
        if annotation.is_positive and annotation.start_s is not None:
            step_targets = (timestamps_s >= float(annotation.start_s)).to(dtype=torch.float32)
            video_target = torch.tensor(float(step_targets.max().item() > 0.0), dtype=torch.float32)
        else:
            step_targets = torch.zeros_like(timestamps_s)
            video_target = torch.tensor(0.0, dtype=torch.float32)
        step_mask = torch.ones_like(timestamps_s, dtype=torch.bool)
        return SequenceSample(
            video_id=annotation.video_id,
            clip_tensor=clip_tensor,
            timestamps_s=timestamps_s,
            step_targets=step_targets,
            step_mask=step_mask,
            video_target=video_target,
            source_positive=torch.tensor(float(annotation.is_positive), dtype=torch.float32),
        )


def collate_sequence_batch(samples: list[SequenceSample]) -> SequenceBatch:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(samples)
    max_steps = max(sample.clip_tensor.shape[0] for sample in samples)
    channels, frames_per_clip, height, width = samples[0].clip_tensor.shape[1:]

    clip_tensor = torch.zeros(
        (batch_size, max_steps, channels, frames_per_clip, height, width),
        dtype=samples[0].clip_tensor.dtype,
    )
    timestamps_s = torch.zeros((batch_size, max_steps), dtype=torch.float32)
    step_targets = torch.zeros((batch_size, max_steps), dtype=torch.float32)
    step_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    video_target = torch.stack([sample.video_target for sample in samples], dim=0)
    source_positive = torch.stack([sample.source_positive for sample in samples], dim=0)
    video_ids: list[str] = []

    for batch_index, sample in enumerate(samples):
        num_steps = sample.clip_tensor.shape[0]
        clip_tensor[batch_index, :num_steps] = sample.clip_tensor
        timestamps_s[batch_index, :num_steps] = sample.timestamps_s
        step_targets[batch_index, :num_steps] = sample.step_targets
        step_mask[batch_index, :num_steps] = sample.step_mask
        video_ids.append(sample.video_id)

    return SequenceBatch(
        video_ids=video_ids,
        clip_tensor=clip_tensor,
        timestamps_s=timestamps_s,
        step_targets=step_targets,
        step_mask=step_mask,
        video_target=video_target,
        source_positive=source_positive,
    )
