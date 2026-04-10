from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def compute_clip_motion_features(clip_frames: torch.Tensor) -> torch.Tensor:
    """Build a lightweight motion descriptor from raw clip frames.

    Args:
        clip_frames: uint8/float tensor with shape [F, C, H, W]

    Returns:
        Tensor with shape [24]:
        - 15 resampled stepwise mean absolute frame differences
        - 3 channelwise mean magnitudes
        - 3 channelwise max magnitudes
        - 3 global stats (mean/std/max of stepwise motion)
    """

    if clip_frames.ndim != 4:
        raise ValueError(f"Expected clip_frames [F, C, H, W], got {tuple(clip_frames.shape)}")
    if clip_frames.shape[0] < 2:
        return torch.zeros((24,), dtype=torch.float32)

    clip = clip_frames.to(dtype=torch.float32) / 255.0
    diffs = clip[1:] - clip[:-1]
    abs_diffs = diffs.abs()

    step_motion = abs_diffs.mean(dim=(1, 2, 3))
    step_motion = F.interpolate(
        step_motion.view(1, 1, -1),
        size=15,
        mode="linear",
        align_corners=False,
    ).view(-1)
    channel_mean = abs_diffs.mean(dim=(0, 2, 3))
    channel_max = abs_diffs.amax(dim=(0, 2, 3))
    global_stats = torch.tensor(
        [
            float(step_motion.mean().item()),
            float(step_motion.std(unbiased=False).item()),
            float(step_motion.max().item()),
        ],
        dtype=torch.float32,
    )
    return torch.cat([step_motion, channel_mean, channel_max, global_stats], dim=0)


def compute_motion_feature_batch(clips: Sequence[torch.Tensor]) -> torch.Tensor:
    if not clips:
        return torch.zeros((0, 24), dtype=torch.float32)
    return torch.stack([compute_clip_motion_features(clip) for clip in clips], dim=0)
