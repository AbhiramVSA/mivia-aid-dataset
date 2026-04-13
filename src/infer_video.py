from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from src.config import ExperimentConfig
from src.data.motion_features import compute_motion_feature_batch
from src.data.video_decode import (
    build_causal_clip_indices,
    decode_sampled_frame_window,
    extract_causal_clip,
    infer_sampled_frame_count,
    preprocess_clip_batch,
)
from src.models.aid_model import AIDTemporalModel
from src.models.temporal_only_model import TemporalOnlyModel
from src.models.videomae_encoder import VideoMAEClipEncoder
from src.utils.checkpoint import load_checkpoint
from src.utils.postprocess import predict_start_time


DEFAULT_CHECKPOINT_PATH = Path("submission/weights/model.pt")
DEFAULT_ENCODER_CHECKPOINT_PATH = Path("artifacts/checkpoints/stage1_best.pt")


@dataclass(frozen=True)
class InferenceBundle:
    mode: str
    raw_model: AIDTemporalModel | None
    encoder: VideoMAEClipEncoder | None
    temporal_model: TemporalOnlyModel | None
    device: torch.device
    config: ExperimentConfig
    extra: dict


def _config_value(config_dict: dict, *keys: str, default):
    current = config_dict
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


@lru_cache(maxsize=1)
def _load_bundle(checkpoint_path: str) -> InferenceBundle:
    payload = load_checkpoint(Path(checkpoint_path), map_location="cpu")
    config_dict = payload.get("config", {})
    extra = payload.get("extra", {})
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
    use_motion_branch = _config_value(
        config_dict, "model", "use_motion_branch", default=config.model.use_motion_branch
    )
    motion_feature_dim = _config_value(
        config_dict, "model", "motion_feature_dim", default=config.model.motion_feature_dim
    )
    config.model.backbone_name = backbone_name
    config.model.hidden_size = hidden_size
    config.model.temporal_model = temporal_model
    config.model.temporal_channels = temporal_channels
    config.model.dropout = dropout
    config.model.transformer_layers = transformer_layers
    config.model.transformer_heads = transformer_heads
    config.model.transformer_ffn_dim = transformer_ffn_dim
    config.model.use_motion_branch = use_motion_branch
    config.model.motion_feature_dim = motion_feature_dim
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = payload["model_state_dict"]
    has_encoder_weights = any(key.startswith("encoder.") for key in state_dict)
    is_cached_stage2 = str(extra.get("stage", "")) == "stage2_cached" or not has_encoder_weights
    if is_cached_stage2:
        encoder = VideoMAEClipEncoder(backbone_name=backbone_name)
        encoder_checkpoint = Path(str(extra.get("encoder_checkpoint", DEFAULT_ENCODER_CHECKPOINT_PATH)))
        if not encoder_checkpoint.is_absolute():
            encoder_checkpoint = (config.paths.project_root / encoder_checkpoint).resolve()
        if encoder_checkpoint.exists():
            encoder_payload = load_checkpoint(encoder_checkpoint, map_location="cpu")
            encoder.load_state_dict(encoder_payload["model_state_dict"], strict=False)
        encoder.to(device)
        encoder.eval()

        temporal_head = TemporalOnlyModel(
            hidden_size=hidden_size,
            temporal_model=temporal_model,
            temporal_channels=temporal_channels,
            dropout=dropout,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ffn_dim=transformer_ffn_dim,
            use_motion_branch=use_motion_branch,
            motion_feature_dim=motion_feature_dim,
        )
        temporal_head.load_state_dict(state_dict, strict=False)
        temporal_head.to(device)
        temporal_head.eval()
        return InferenceBundle(
            mode="cached_stage2",
            raw_model=None,
            encoder=encoder,
            temporal_model=temporal_head,
            device=device,
            config=config,
            extra=extra,
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
        use_motion_branch=use_motion_branch,
        motion_feature_dim=motion_feature_dim,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return InferenceBundle(
        mode="raw_stage2",
        raw_model=model,
        encoder=None,
        temporal_model=None,
        device=device,
        config=config,
        extra=extra,
    )


@torch.no_grad()
def infer_single_video(video_path: Path, checkpoint_path: Path | None = None) -> float | None:
    """Return the predicted onset time in seconds or None for no incident."""
    resolved_checkpoint = checkpoint_path or DEFAULT_CHECKPOINT_PATH
    bundle = _load_bundle(str(resolved_checkpoint))
    device = bundle.device
    config = bundle.config
    extra = bundle.extra
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
        motion_features = None
        if config.model.use_motion_branch:
            motion_features = compute_motion_feature_batch(clips).unsqueeze(0).to(device=device, dtype=torch.float32)
        clip_tensor = preprocess_clip_batch(clips, backbone_name=config.model.backbone_name).unsqueeze(0).to(device)
        if bundle.mode == "cached_stage2":
            assert bundle.encoder is not None
            assert bundle.temporal_model is not None
            features = bundle.encoder(clip_tensor).unsqueeze(0)
            step_logits, video_logits, _ = bundle.temporal_model(features, motion_features=motion_features)
        else:
            assert bundle.raw_model is not None
            step_logits, video_logits, _ = bundle.raw_model(clip_tensor, motion_features=motion_features)
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
