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
    preprocess_clip_frames,
)


@dataclass(slots=True)
class ClipSample:
    video_id: str
    clip: torch.Tensor
    clip_end_s: float
    label: int


@dataclass(slots=True)
class ClipRecord:
    annotation_index: int
    clip_end_s: float
    label: int
    group: str


class Stage1ClipDataset(Dataset[ClipSample]):
    """Clip-level dataset for the onset-aware pretraining stage."""

    def __init__(self, annotations: list[VideoAnnotation], sampling: VideoSamplingConfig, backbone_name: str) -> None:
        self.annotations = annotations
        self.sampling = sampling
        self.backbone_name = backbone_name
        self.records = self._build_records()

    def _build_records(self) -> list[ClipRecord]:
        records: list[ClipRecord] = []
        for annotation_index, annotation in enumerate(self.annotations):
            num_frames = infer_sampled_frame_count(annotation.video_path, self.sampling.sample_fps)
            clip_spans = build_causal_clip_indices(num_frames, self.sampling)
            for _, end_idx in clip_spans:
                clip_end_s = (end_idx - 1) / float(self.sampling.sample_fps)
                if not annotation.is_positive:
                    label = 0
                    group = "negative_video"
                elif annotation.start_s is not None and clip_end_s >= float(annotation.start_s):
                    label = 1
                    group = "post_onset"
                else:
                    label = 0
                    group = "pre_onset"
                records.append(
                    ClipRecord(
                        annotation_index=annotation_index,
                        clip_end_s=clip_end_s,
                        label=label,
                        group=group,
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ClipSample:
        record = self.records[index]
        annotation = self.annotations[record.annotation_index]
        clip_end_idx = max(1, int(round(record.clip_end_s * self.sampling.sample_fps)) + 1)
        sample_start_idx = max(0, clip_end_idx - self.sampling.clip_num_frames)
        frames, _ = decode_sampled_frame_window(
            annotation.video_path,
            self.sampling.sample_fps,
            start_sample_idx=sample_start_idx,
            end_sample_idx=clip_end_idx,
        )
        clip = extract_causal_clip(
            frames,
            end_idx_exclusive=clip_end_idx - sample_start_idx,
            clip_len=self.sampling.clip_num_frames,
        )
        processed_clip = preprocess_clip_frames(clip, backbone_name=self.backbone_name)
        return ClipSample(
            video_id=annotation.video_id,
            clip=processed_clip,
            clip_end_s=record.clip_end_s,
            label=record.label,
        )

    def sample_weights(self) -> torch.Tensor:
        group_counts: dict[str, int] = {}
        for record in self.records:
            group_counts[record.group] = group_counts.get(record.group, 0) + 1
        weights = [1.0 / max(1, group_counts[record.group]) for record in self.records]
        return torch.tensor(weights, dtype=torch.float32)
