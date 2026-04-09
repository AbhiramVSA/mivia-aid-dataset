from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from src.config import ExperimentConfig
from src.data.video_decode import (
    build_causal_clip_indices,
    decode_sampled_frame_window,
    extract_causal_clip,
    infer_sampled_frame_count,
    preprocess_clip_batch,
)
from src.models.aid_model import AIDTemporalModel
from src.utils.checkpoint import load_checkpoint
from src.utils.postprocess import predict_start_time


DEFAULT_CHECKPOINT_PATH = Path("submission/weights/model.pt")


def _config_value(config_dict: dict, *keys: str, default):
    current = config_dict
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


@lru_cache(maxsize=1)
def _load_bundle(checkpoint_path: str) -> tuple[AIDTemporalModel, torch.device, ExperimentConfig, dict]:
    payload = load_checkpoint(Path(checkpoint_path), map_location="cpu")
    config_dict = payload.get("config", {})
    config = ExperimentConfig().resolved()
    backbone_name = _config_value(config_dict, "model", "backbone_name", default=config.model.backbone_name)
    hidden_size = _config_value(config_dict, "model", "hidden_size", default=config.model.hidden_size)
    temporal_model = _config_value(config_dict, "model", "temporal_model", default=config.model.temporal_model)
    temporal_channels = tuple(
        _config_value(config_dict, "model", "temporal_channels", default=config.model.temporal_channels)
    )
    dropout = _config_value(config_dict, "model", "dropout", default=config.model.dropout)
    transformer_layers = _config_value(
        config_dict, "model", "transformer_layers", default=config.model.transformer_layers
    )
    transformer_heads = _config_value(
        config_dict, "model", "transformer_heads", default=config.model.transformer_heads
    )
    transformer_ffn_dim = _config_value(
        config_dict, "model", "transformer_ffn_dim", default=config.model.transformer_ffn_dim
    )
    config.model.backbone_name = backbone_name
    config.model.hidden_size = hidden_size
    config.model.temporal_model = temporal_model
    config.model.temporal_channels = temporal_channels
    config.model.dropout = dropout
    config.model.transformer_layers = transformer_layers
    config.model.transformer_heads = transformer_heads
    config.model.transformer_ffn_dim = transformer_ffn_dim
    config.video.sample_fps = _config_value(config_dict, "video", "sample_fps", default=config.video.sample_fps)
    config.video.clip_num_frames = _config_value(
        config_dict, "video", "clip_num_frames", default=config.video.clip_num_frames
    )
    config.video.clip_stride_frames = _config_value(
        config_dict, "video", "clip_stride_frames", default=config.video.clip_stride_frames
    )
    config.stage2.max_steps_per_sample = _config_value(
        config_dict, "stage2", "max_steps_per_sample", default=config.stage2.max_steps_per_sample
    )
    config.stage2.window_stride_steps = _config_value(
        config_dict, "stage2", "window_stride_steps", default=config.stage2.window_stride_steps
    )
    config.postprocess.median_kernel_size = _config_value(
        config_dict, "postprocess", "median_kernel_size", default=config.postprocess.median_kernel_size
    )
    config.postprocess.default_tau_video = _config_value(
        config_dict, "postprocess", "default_tau_video", default=config.postprocess.default_tau_video
    )
    config.postprocess.default_tau_keep = _config_value(
        config_dict, "postprocess", "default_tau_keep", default=config.postprocess.default_tau_keep
    )
    config.postprocess.default_min_consecutive_steps = _config_value(
        config_dict,
        "postprocess",
        "default_min_consecutive_steps",
        default=config.postprocess.default_min_consecutive_steps,
    )
    config.postprocess.prediction_mode = _config_value(
        config_dict, "postprocess", "prediction_mode", default=config.postprocess.prediction_mode
    )
    model = AIDTemporalModel(
        backbone_name=backbone_name,
        hidden_size=hidden_size,
        temporal_model=temporal_model,
        temporal_channels=temporal_channels,
        dropout=dropout,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_ffn_dim=transformer_ffn_dim,
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device, config, payload.get("extra", {})


@torch.no_grad()
def infer_single_video(video_path: Path, checkpoint_path: Path | None = None) -> float | None:
    """Return the predicted onset time in seconds or None for no incident."""
    resolved_checkpoint = checkpoint_path or DEFAULT_CHECKPOINT_PATH
    model, device, config, extra = _load_bundle(str(resolved_checkpoint))
    total_sampled_frames = infer_sampled_frame_count(video_path, config.video.sample_fps)
    clip_spans = build_causal_clip_indices(total_sampled_frames, config.video)
    if not clip_spans:
        return None

    max_steps = config.stage2.max_steps_per_sample
    window_stride = config.stage2.window_stride_steps
    aggregated: dict[float, list[float]] = {}
    video_scores: list[float] = []

    start = 0
    while start < len(clip_spans):
        end = min(len(clip_spans), start + max_steps)
        window_spans = clip_spans[start:end]
        first_end_idx = window_spans[0][1]
        last_end_idx = window_spans[-1][1]
        sample_start_idx = max(0, first_end_idx - config.video.clip_num_frames)
        sample_end_idx = last_end_idx
        frames, _ = decode_sampled_frame_window(
            video_path,
            config.video.sample_fps,
            start_sample_idx=sample_start_idx,
            end_sample_idx=sample_end_idx,
        )
        clips = [
            extract_causal_clip(
                frames,
                end_idx_exclusive=end_idx - sample_start_idx,
                clip_len=config.video.clip_num_frames,
            )
            for _, end_idx in window_spans
        ]
        clip_tensor = preprocess_clip_batch(clips, backbone_name=config.model.backbone_name).unsqueeze(0).to(device)
        step_logits, video_logits, _ = model(clip_tensor)
        step_scores = torch.sigmoid(step_logits[0]).cpu().tolist()
        video_scores.append(float(torch.sigmoid(video_logits[0]).item()))
        timestamps = [(end_idx - 1) / float(config.video.sample_fps) for _, end_idx in window_spans]
        for timestamp, score in zip(timestamps, step_scores):
            aggregated.setdefault(timestamp, []).append(float(score))
        if end >= len(clip_spans):
            break
        start += window_stride

    timestamps = sorted(aggregated)
    scores = [sum(aggregated[t]) / len(aggregated[t]) for t in timestamps]
    video_score = sum(video_scores) / max(1, len(video_scores))
    result = predict_start_time(
        scores,
        timestamps,
        tau_empty=float(extra.get("tau_empty", config.postprocess.default_tau_empty)),
        tau_start=float(extra.get("tau_start", config.postprocess.default_tau_start)),
        tau_keep=float(extra.get("tau_keep", config.postprocess.default_tau_keep)),
        tau_video=float(extra.get("tau_video", config.postprocess.default_tau_video)),
        video_score=video_score,
        median_kernel_size=config.postprocess.median_kernel_size,
        min_consecutive_steps=int(
            extra.get("min_consecutive_steps", config.postprocess.default_min_consecutive_steps)
        ),
        mode=str(extra.get("prediction_mode", config.postprocess.prediction_mode)),
    )
    return result.predicted_start_s
