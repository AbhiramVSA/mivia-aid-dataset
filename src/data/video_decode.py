from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from src.config import PathsConfig, VideoSamplingConfig


@dataclass(slots=True)
class VideoProbe:
    frame_count: int
    fps: float
    duration_s: float


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("OpenCV is required for video decoding. Install opencv-python-headless.") from exc
    return cv2


@lru_cache(maxsize=4)
def _get_videomae_processor(backbone_name: str):
    try:
        from transformers import AutoImageProcessor  # type: ignore
    except ImportError as exc:
        raise ImportError("transformers is required for VideoMAE preprocessing.") from exc
    project_root = PathsConfig().resolve().project_root
    local_processor_dir = project_root / "submission" / "processor" / "videomae-base"
    if local_processor_dir.exists():
        return AutoImageProcessor.from_pretrained(local_processor_dir)
    backbone_path = Path(backbone_name)
    if backbone_path.exists():
        return AutoImageProcessor.from_pretrained(backbone_path)
    return AutoImageProcessor.from_pretrained(backbone_name)


def _standardize_processor_output(pixel_values: torch.Tensor) -> torch.Tensor:
    if pixel_values.ndim != 5:
        raise ValueError(f"Expected 5D processor output, got shape {tuple(pixel_values.shape)}")
    if pixel_values.shape[1] == 3:
        return pixel_values
    if pixel_values.shape[2] == 3:
        return pixel_values.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(f"Cannot infer channel dimension from shape {tuple(pixel_values.shape)}")


def _resampled_source_indices(frame_count: int, source_fps: float, sample_fps: int) -> list[int]:
    if frame_count <= 0:
        return []
    if source_fps <= 0:
        source_fps = float(sample_fps)
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")

    step = source_fps / float(sample_fps)
    target = 0.0
    indices: list[int] = []
    while True:
        source_index = int(round(target))
        if source_index >= frame_count:
            break
        if not indices or source_index > indices[-1]:
            indices.append(source_index)
        target += step
    if not indices:
        indices.append(0)
    return indices


def probe_video(video_path: Path) -> VideoProbe:
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    duration_s = (frame_count / fps) if frame_count > 0 and fps > 0 else 0.0
    capture.release()
    return VideoProbe(frame_count=frame_count, fps=fps, duration_s=duration_s)


def decode_video_frames(video_path: Path, sample_fps: int) -> tuple[torch.Tensor, float]:
    """Decode a video into RGB frames sampled at `sample_fps`.

    Returns:
        frames: Tensor with shape [T, C, H, W]
        source_fps: The original average fps reported by the decoder
    """
    cv2 = _require_cv2()
    return decode_sampled_frame_window(video_path, sample_fps)


def decode_sampled_frame_window(
    video_path: Path,
    sample_fps: int,
    start_sample_idx: int = 0,
    end_sample_idx: int | None = None,
) -> tuple[torch.Tensor, float]:
    def _decode_target_indices(target_source_indices: list[int], use_seek: bool) -> list[torch.Tensor]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        try:
            if use_seek:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target_source_indices[0])
                current_source_idx = target_source_indices[0]
            else:
                current_source_idx = 0

            frames_local: list[torch.Tensor] = []
            current_target_ptr = 0
            final_source_idx = target_source_indices[-1]

            while current_source_idx <= final_source_idx and current_target_ptr < len(target_source_indices):
                success, frame = capture.read()
                if not success:
                    break
                if current_source_idx == target_source_indices[current_target_ptr]:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_local.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
                    current_target_ptr += 1
                current_source_idx += 1
            return frames_local
        finally:
            capture.release()

    cv2 = _require_cv2()
    probe = probe_video(video_path)
    sampled_source_indices = _resampled_source_indices(probe.frame_count, probe.fps, sample_fps)
    if end_sample_idx is None:
        end_sample_idx = len(sampled_source_indices)
    start_sample_idx = max(0, start_sample_idx)
    end_sample_idx = min(len(sampled_source_indices), end_sample_idx)
    if start_sample_idx >= end_sample_idx:
        raise ValueError(
            f"Invalid sampled-frame window [{start_sample_idx}, {end_sample_idx}) for {video_path}"
        )

    target_source_indices = sampled_source_indices[start_sample_idx:end_sample_idx]
    frames = _decode_target_indices(target_source_indices, use_seek=True)

    if len(frames) == len(target_source_indices):
        return torch.stack(frames, dim=0), probe.fps

    # Some codecs report slightly optimistic metadata or land imprecisely after
    # random seeking. Retry with a deterministic scan from the start before
    # falling back to tail padding.
    fallback_frames = _decode_target_indices(target_source_indices, use_seek=False)
    if len(fallback_frames) > len(frames):
        frames = fallback_frames

    if frames and len(frames) < len(target_source_indices):
        shortfall = len(target_source_indices) - len(frames)
        pad_frame = frames[-1].clone()
        frames.extend([pad_frame.clone() for _ in range(shortfall)])
        return torch.stack(frames, dim=0), probe.fps

    if not frames:
        raise RuntimeError(f"No frames decoded from window in {video_path}")

    if len(frames) != len(target_source_indices):
        raise RuntimeError(
            f"Decoded {len(frames)} sampled frames but expected {len(target_source_indices)} from {video_path}"
        )
    return torch.stack(frames, dim=0), probe.fps


def extract_causal_clip(frames: torch.Tensor, end_idx_exclusive: int, clip_len: int) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected frames [T, C, H, W], got {tuple(frames.shape)}")
    end_idx_exclusive = max(1, min(end_idx_exclusive, frames.shape[0]))
    start_idx = max(0, end_idx_exclusive - clip_len)
    clip = frames[start_idx:end_idx_exclusive]
    if clip.shape[0] < clip_len:
        pad_count = clip_len - clip.shape[0]
        pad_frame = clip[:1].expand(pad_count, -1, -1, -1)
        clip = torch.cat([pad_frame, clip], dim=0)
    return clip


def preprocess_clip_frames(clip_frames: torch.Tensor, backbone_name: str) -> torch.Tensor:
    """Convert raw uint8 frames [F, C, H, W] into model-ready [C, F, H, W]."""

    processor = _get_videomae_processor(backbone_name)
    frame_list = [frame.permute(1, 2, 0).cpu().numpy() for frame in clip_frames]
    inputs = processor(frame_list, return_tensors="pt")
    pixel_values = _standardize_processor_output(inputs["pixel_values"])
    return pixel_values[0]


def preprocess_clip_batch(clips: Sequence[torch.Tensor], backbone_name: str) -> torch.Tensor:
    processor = _get_videomae_processor(backbone_name)
    nested_frames = [[frame.permute(1, 2, 0).cpu().numpy() for frame in clip] for clip in clips]
    inputs = processor(nested_frames, return_tensors="pt")
    return _standardize_processor_output(inputs["pixel_values"])


def pad_clip_batch(clips: list[torch.Tensor]) -> torch.Tensor:
    if not clips:
        raise ValueError("Cannot pad an empty clip batch")
    max_h = max(clip.shape[-2] for clip in clips)
    max_w = max(clip.shape[-1] for clip in clips)
    padded: list[torch.Tensor] = []
    for clip in clips:
        pad_h = max_h - clip.shape[-2]
        pad_w = max_w - clip.shape[-1]
        if pad_h == 0 and pad_w == 0:
            padded.append(clip)
            continue
        padded.append(F.pad(clip, (0, pad_w, 0, pad_h)))
    return torch.stack(padded, dim=0)


def infer_sampled_frame_count(video_path: Path, sample_fps: int) -> int:
    probe = probe_video(video_path)
    return len(_resampled_source_indices(probe.frame_count, probe.fps, sample_fps))


def build_causal_clip_indices(num_frames: int, sampling: VideoSamplingConfig) -> list[tuple[int, int]]:
    clip_spans: list[tuple[int, int]] = []
    clip_len = sampling.clip_num_frames
    stride = sampling.clip_stride_frames
    if num_frames <= 0:
        return clip_spans
    for end_idx in range(clip_len - 1, num_frames, stride):
        start_idx = end_idx - clip_len + 1
        clip_spans.append((start_idx, end_idx + 1))
    if not clip_spans and num_frames > 0:
        clip_spans.append((0, min(num_frames, clip_len)))
    return clip_spans
