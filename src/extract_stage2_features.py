from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import ExperimentConfig
from src.data.annotations import VideoAnnotation, load_annotations
from src.data.temporal_targets import TEMPORAL_BIN_IGNORE_INDEX, build_temporal_distance_bins
from src.data.video_decode import (
    build_causal_clip_indices,
    decode_sampled_frame_window,
    extract_causal_clip,
    infer_sampled_frame_count,
    preprocess_clip_batch,
)
from src.models.videomae_encoder import VideoMAEClipEncoder
from src.utils.checkpoint import load_checkpoint

CACHE_SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract per-video Stage 2 clip embeddings")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--split", type=str, choices=("train", "val", "all"), default="all")
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_split(config: ExperimentConfig, split: str, limit: int | None = None) -> list[VideoAnnotation]:
    if split == "train":
        records = load_annotations(config.paths.train_csv, config.paths.train_videos_dir, split="train")
    elif split == "val":
        records = load_annotations(config.paths.val_csv, config.paths.val_videos_dir, split="val")
    else:
        records = load_annotations(config.paths.train_csv, config.paths.train_videos_dir, split="train")
        if config.paths.val_videos_dir.exists():
            records.extend(load_annotations(config.paths.val_csv, config.paths.val_videos_dir, split="val"))
    if limit is not None:
        records = records[:limit]
    return records


def load_encoder(config: ExperimentConfig, checkpoint_path: Path | None, device: torch.device) -> VideoMAEClipEncoder:
    encoder = VideoMAEClipEncoder(backbone_name=config.model.backbone_name).to(device)
    if checkpoint_path is not None:
        payload = load_checkpoint(checkpoint_path, map_location="cpu")
        encoder.load_state_dict(payload["model_state_dict"], strict=False)
    encoder.eval()
    return encoder


def update_summary_from_bundle(
    bundle: dict,
    *,
    temporal_bin_hist: torch.Tensor,
) -> tuple[int, int]:
    is_positive = bool(bundle["source_positive"].item() > 0.5)
    if is_positive:
        valid_bins = bundle["temporal_bin_targets"]
        valid_bins = valid_bins[valid_bins >= 0]
        if valid_bins.numel() > 0:
            temporal_bin_hist += torch.bincount(valid_bins, minlength=4).to(dtype=torch.long)
        return 1, 0
    return 0, 1


@torch.no_grad()
def encode_video(
    annotation: VideoAnnotation,
    *,
    config: ExperimentConfig,
    encoder: VideoMAEClipEncoder,
    device: torch.device,
    chunk_steps: int,
    clip_batch_size: int,
) -> dict:
    total_sampled_frames = infer_sampled_frame_count(annotation.video_path, config.video.sample_fps)
    clip_spans = build_causal_clip_indices(total_sampled_frames, config.video)
    if not clip_spans:
        empty = torch.zeros((0, config.model.hidden_size), dtype=torch.float16)
        empty_time = torch.zeros((0,), dtype=torch.float32)
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "video_id": annotation.video_id,
            "features": empty,
            "timestamps_s": empty_time,
            "step_targets": empty_time.clone(),
            "onset_targets": empty_time.clone(),
            "temporal_bin_targets": torch.empty((0,), dtype=torch.long),
            "ground_truth_start_s": None,
            "video_target": torch.tensor(0.0, dtype=torch.float32),
            "source_positive": torch.tensor(float(annotation.is_positive), dtype=torch.float32),
        }

    all_features: list[torch.Tensor] = []
    all_timestamps: list[float] = []

    for start in range(0, len(clip_spans), chunk_steps):
        end = min(len(clip_spans), start + chunk_steps)
        window_spans = clip_spans[start:end]
        first_end_idx = window_spans[0][1]
        last_end_idx = window_spans[-1][1]
        sample_start_idx = max(0, first_end_idx - config.video.clip_num_frames)
        sample_end_idx = last_end_idx
        frames, _ = decode_sampled_frame_window(
            annotation.video_path,
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
        for batch_start in range(0, len(clips), clip_batch_size):
            batch_clips = clips[batch_start : batch_start + clip_batch_size]
            clip_tensor = preprocess_clip_batch(batch_clips, backbone_name=config.model.backbone_name).to(device)
            features = encoder(clip_tensor).to(dtype=torch.float16).cpu()
            all_features.append(features)
        all_timestamps.extend([(end_idx - 1) / float(config.video.sample_fps) for _, end_idx in window_spans])

    features_tensor = torch.cat(all_features, dim=0)
    timestamps_s = torch.tensor(all_timestamps, dtype=torch.float32)
    if annotation.is_positive and annotation.start_s is not None:
        step_targets = (timestamps_s >= float(annotation.start_s)).to(dtype=torch.float32)
        distance = timestamps_s - float(annotation.start_s)
        onset_targets = torch.exp(-0.5 * (distance / float(config.stage2.onset_sigma_seconds)) ** 2)
        temporal_bin_targets = build_temporal_distance_bins(
            timestamps_s,
            float(annotation.start_s),
            bin_edges_s=config.stage2.temporal_distance_bin_edges_s,
        )
        video_target = torch.tensor(1.0, dtype=torch.float32)
    else:
        step_targets = torch.zeros_like(timestamps_s)
        onset_targets = torch.zeros_like(timestamps_s)
        temporal_bin_targets = torch.full_like(timestamps_s, TEMPORAL_BIN_IGNORE_INDEX, dtype=torch.long)
        video_target = torch.tensor(0.0, dtype=torch.float32)
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "video_id": annotation.video_id,
        "features": features_tensor,
        "timestamps_s": timestamps_s,
        "step_targets": step_targets,
        "onset_targets": onset_targets,
        "temporal_bin_targets": temporal_bin_targets,
        "ground_truth_start_s": annotation.start_s,
        "video_target": video_target,
        "source_positive": torch.tensor(float(annotation.is_positive), dtype=torch.float32),
    }


def main() -> None:
    args = parse_args()
    config = ExperimentConfig().resolved()
    records = build_split(config, args.split, limit=args.limit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(config, args.encoder_checkpoint, device)
    output_root = config.paths.features_dir / "stage2_cache"
    if args.encoder_checkpoint is not None:
        output_root = output_root / args.encoder_checkpoint.stem
    else:
        output_root = output_root / "pretrained"

    temporal_bin_hist = torch.zeros(4, dtype=torch.long)
    positive_videos = 0
    negative_videos = 0
    extracted = 0
    skipped = 0

    for index, annotation in enumerate(records, start=1):
        split_dir = output_root / annotation.split
        split_dir.mkdir(parents=True, exist_ok=True)
        output_path = split_dir / f"{annotation.video_id}.pt"
        if output_path.exists() and not args.force:
            existing = torch.load(output_path, map_location="cpu", weights_only=False)
            if existing.get("cache_schema_version") == CACHE_SCHEMA_VERSION and "temporal_bin_targets" in existing:
                skipped += 1
                positive_delta, negative_delta = update_summary_from_bundle(
                    existing,
                    temporal_bin_hist=temporal_bin_hist,
                )
                positive_videos += positive_delta
                negative_videos += negative_delta
                print(f"[cache] skip {index}/{len(records)} {annotation.split}/{annotation.video_id}")
                continue
            print(f"[cache] refresh {index}/{len(records)} {annotation.split}/{annotation.video_id} reason=schema")
        bundle = encode_video(
            annotation,
            config=config,
            encoder=encoder,
            device=device,
            chunk_steps=args.chunk_steps,
            clip_batch_size=args.clip_batch_size,
        )
        torch.save(bundle, output_path)
        extracted += 1
        positive_delta, negative_delta = update_summary_from_bundle(
            bundle,
            temporal_bin_hist=temporal_bin_hist,
        )
        positive_videos += positive_delta
        negative_videos += negative_delta
        print(
            f"[cache] done {index}/{len(records)} {annotation.split}/{annotation.video_id} "
            f"steps={bundle['features'].shape[0]}"
        )
    print(
        " ".join(
            [
                f"[cache] schema={CACHE_SCHEMA_VERSION}",
                f"output_root={output_root}",
                f"extracted={extracted}",
                f"skipped={skipped}",
                f"positive_videos={positive_videos}",
                f"negative_videos={negative_videos}",
                f"temporal_bin_hist={temporal_bin_hist.tolist()}",
            ]
        )
    )


if __name__ == "__main__":
    main()
