from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.cached_sequence_dataset import CachedSequenceDataset, collate_cached_sequence_batch
from src.models.temporal_only_model import TemporalOnlyModel
from src.train_stage2 import sweep_postprocess_thresholds
from src.utils.checkpoint import load_checkpoint
from src.utils.postprocess import predict_start_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine false-positive negative videos from cached checkpoints")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", type=str, choices=("train", "val"), default="train")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-recall-for-selection", type=float, default=None)
    return parser.parse_args()


def _config_value(config_dict: dict, *keys: str, default):
    current = config_dict
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def main() -> None:
    args = parse_args()
    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    config_dict = payload.get("config", {})
    extra = payload.get("extra", {})
    config = ExperimentConfig().resolved()
    config.model.temporal_model = _config_value(config_dict, "model", "temporal_model", default=config.model.temporal_model)
    config.model.temporal_channels = tuple(
        _config_value(config_dict, "model", "temporal_channels", default=config.model.temporal_channels)
    )
    config.model.hidden_size = _config_value(config_dict, "model", "hidden_size", default=config.model.hidden_size)
    config.model.dropout = _config_value(config_dict, "model", "dropout", default=config.model.dropout)
    config.model.transformer_layers = _config_value(
        config_dict, "model", "transformer_layers", default=config.model.transformer_layers
    )
    config.model.transformer_heads = _config_value(
        config_dict, "model", "transformer_heads", default=config.model.transformer_heads
    )
    config.model.transformer_ffn_dim = _config_value(
        config_dict, "model", "transformer_ffn_dim", default=config.model.transformer_ffn_dim
    )
    config.model.use_motion_branch = _config_value(
        config_dict, "model", "use_motion_branch", default=config.model.use_motion_branch
    )
    config.model.motion_feature_dim = _config_value(
        config_dict, "model", "motion_feature_dim", default=config.model.motion_feature_dim
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
    config.postprocess.prediction_mode = _config_value(
        config_dict, "postprocess", "prediction_mode", default=config.postprocess.prediction_mode
    )
    config.postprocess.selection_min_recall = _config_value(
        config_dict, "postprocess", "selection_min_recall", default=config.postprocess.selection_min_recall
    )
    if args.min_recall_for_selection is not None:
        config.postprocess.selection_min_recall = args.min_recall_for_selection

    dataset = CachedSequenceDataset(
        cache_dir=args.cache_root / args.split,
        max_steps=config.stage2.max_steps_per_sample,
        window_stride=config.stage2.window_stride_steps,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_cached_sequence_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalOnlyModel(
        hidden_size=config.model.hidden_size,
        temporal_model=config.model.temporal_model,
        temporal_channels=config.model.temporal_channels,
        dropout=config.model.dropout,
        transformer_layers=config.model.transformer_layers,
        transformer_heads=config.model.transformer_heads,
        transformer_ffn_dim=config.model.transformer_ffn_dim,
        use_motion_branch=config.model.use_motion_branch,
        motion_feature_dim=config.model.motion_feature_dim,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()

    per_video_scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_video_probs: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            features = batch.features.to(device=device, dtype=torch.float32, non_blocking=True)
            motion_features = batch.motion_features.to(device=device, dtype=torch.float32, non_blocking=True)
            step_mask = batch.step_mask.to(device, non_blocking=True)
            step_logits, video_logits, _ = model(features, step_mask=step_mask, motion_features=motion_features)
            probs = torch.sigmoid(step_logits).cpu()
            video_probs = torch.sigmoid(video_logits).cpu()
            for sample_index, video_id in enumerate(batch.video_ids):
                valid_mask = batch.step_mask[sample_index]
                timestamps = batch.timestamps_s[sample_index][valid_mask].tolist()
                scores = probs[sample_index][valid_mask].tolist()
                per_video_scores[video_id].extend(zip(timestamps, scores))
                per_video_probs[video_id].append(float(video_probs[sample_index].item()))

    metrics = sweep_postprocess_thresholds(
        annotations=dataset.annotation_metas,
        config=config,
        per_video_scores=per_video_scores,
        per_video_probs=per_video_probs,
    )
    tau_empty = float(extra.get("tau_empty", metrics["tau_empty"]))
    tau_start = float(extra.get("tau_start", metrics["tau_start"]))
    tau_keep = float(extra.get("tau_keep", metrics["tau_keep"]))
    tau_video = float(extra.get("tau_video", metrics["tau_video"]))
    min_consecutive = int(extra.get("min_consecutive_steps", metrics["min_consecutive_steps"]))
    prediction_mode = str(extra.get("prediction_mode", config.postprocess.prediction_mode))

    hard_negative_ids: list[str] = []
    for annotation in dataset.annotation_metas:
        merged: dict[float, list[float]] = defaultdict(list)
        for timestamp, score in per_video_scores.get(annotation.video_id, []):
            merged[float(timestamp)].append(float(score))
        timestamps = sorted(merged)
        scores = [sum(merged[t]) / len(merged[t]) for t in timestamps]
        video_prob_list = per_video_probs.get(annotation.video_id, [])
        video_score = sum(video_prob_list) / len(video_prob_list) if video_prob_list else 0.0
        prediction = predict_start_time(
            scores,
            timestamps,
            tau_empty=tau_empty,
            tau_start=tau_start,
            tau_keep=tau_keep,
            tau_video=tau_video,
            video_score=video_score,
            median_kernel_size=config.postprocess.median_kernel_size,
            min_consecutive_steps=min_consecutive,
            mode=prediction_mode,
        )
        if (not annotation.is_positive) and prediction.predicted_start_s is not None:
            hard_negative_ids.append(annotation.video_id)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("\n".join(sorted(hard_negative_ids)) + ("\n" if hard_negative_ids else ""), encoding="utf-8")
    print(
        f"split={args.split} negatives={sum(1 for a in dataset.annotation_metas if not a.is_positive)} "
        f"hard_negatives={len(hard_negative_ids)} output={args.output_path}"
    )


if __name__ == "__main__":
    main()
