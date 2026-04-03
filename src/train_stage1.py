from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.annotations import VideoAnnotation, load_annotations
from src.data.sequence_dataset import Stage2SequenceDataset, collate_sequence_batch
from src.models.clip_onset_model import ClipOnsetModel
from src.utils.checkpoint import checkpoint_payload, save_checkpoint
from src.utils.metrics import PredictionRecord, compute_contest_metrics
from src.utils.postprocess import predict_start_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 clip-level training with per-video windows")
    parser.add_argument("--output-name", type=str, default="stage1_best.pt")
    parser.add_argument("--max-train-videos", type=int, default=None)
    parser.add_argument("--max-val-videos", type=int, default=None)
    return parser.parse_args()


def print_device_diagnostics() -> None:
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_version={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu_count={torch.cuda.device_count()}")
        print(f"gpu_name={torch.cuda.get_device_name(0)}")
    else:
        print("gpu_name=None")


def build_split(
    *,
    csv_path: Path,
    videos_dir: Path,
    split: str,
    limit: int | None = None,
) -> list[VideoAnnotation]:
    records = load_annotations(csv_path, videos_dir, split=split)
    if limit is not None:
        records = records[:limit]
    return records


def make_dataloaders(config: ExperimentConfig, args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None]:
    train_records = build_split(
        csv_path=config.paths.train_csv,
        videos_dir=config.paths.train_videos_dir,
        split="train",
        limit=args.max_train_videos,
    )
    train_dataset = Stage2SequenceDataset(
        train_records,
        sampling=config.video,
        backbone_name=config.model.backbone_name,
        max_steps=config.stage1.max_steps_per_sample,
        window_stride=config.stage1.window_stride_steps,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.stage1.batch_size,
        shuffle=True,
        num_workers=config.stage1.num_workers,
        pin_memory=True,
        collate_fn=collate_sequence_batch,
    )

    val_loader: DataLoader | None = None
    if config.paths.val_videos_dir.exists():
        val_records = build_split(
            csv_path=config.paths.val_csv,
            videos_dir=config.paths.val_videos_dir,
            split="val",
            limit=args.max_val_videos,
        )
        val_dataset = Stage2SequenceDataset(
            val_records,
            sampling=config.video,
            backbone_name=config.model.backbone_name,
            max_steps=config.stage1.max_steps_per_sample,
            window_stride=config.stage1.window_stride_steps,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.stage1.batch_size,
            shuffle=False,
            num_workers=config.stage1.num_workers,
            pin_memory=True,
            collate_fn=collate_sequence_batch,
        )
    return train_loader, val_loader


def build_optimizer(model: ClipOnsetModel, config: ExperimentConfig) -> torch.optim.Optimizer:
    model.build()
    assert model.classifier is not None
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.classifier.parameters())
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.stage1.backbone_lr},
            {"params": head_params, "lr": config.stage1.head_lr},
        ],
        weight_decay=config.stage1.weight_decay,
    )


def compute_balanced_step_loss(
    logits: torch.Tensor,
    step_targets: torch.Tensor,
    step_mask: torch.Tensor,
    source_positive: torch.Tensor,
) -> torch.Tensor:
    valid = step_mask.bool()
    loss = F.binary_cross_entropy_with_logits(logits, step_targets, reduction="none")
    weights = torch.zeros_like(loss)
    negative_video = valid & (source_positive.unsqueeze(1) < 0.5)
    positive_pre = valid & (source_positive.unsqueeze(1) >= 0.5) & (step_targets < 0.5)
    positive_post = valid & (step_targets >= 0.5)

    for mask in (negative_video, positive_pre, positive_post):
        count = int(mask.sum().item())
        if count > 0:
            weights[mask] = 1.0 / count

    normalizer = weights[valid].sum().clamp_min(1e-8)
    weights = weights * (valid.sum().clamp_min(1).to(dtype=weights.dtype) / normalizer)
    return (loss * weights * valid).sum() / valid.sum().clamp_min(1)


def train_one_epoch(
    model: ClipOnsetModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0
    for batch in loader:
        clip_tensor = batch.clip_tensor.to(device, non_blocking=True)
        step_targets = batch.step_targets.to(device, non_blocking=True)
        step_mask = batch.step_mask.to(device, non_blocking=True)
        source_positive = batch.source_positive.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=True) if amp and device.type == "cuda" else nullcontext()
        )
        with autocast_context:
            logits = model(clip_tensor)
            loss = compute_balanced_step_loss(logits, step_targets, step_mask, source_positive)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1
    return running_loss / max(1, num_batches)


@torch.no_grad()
def validate(
    model: ClipOnsetModel,
    loader: DataLoader,
    annotations: list[VideoAnnotation],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    annotation_by_id = {annotation.video_id: annotation for annotation in annotations}
    per_video_scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        clip_tensor = batch.clip_tensor.to(device, non_blocking=True)
        step_targets = batch.step_targets.to(device, non_blocking=True)
        step_mask = batch.step_mask.to(device, non_blocking=True)
        source_positive = batch.source_positive.to(device, non_blocking=True)

        logits = model(clip_tensor)
        loss = compute_balanced_step_loss(logits, step_targets, step_mask, source_positive)
        total_loss += float(loss.item())
        num_batches += 1

        probs = torch.sigmoid(logits).cpu()
        for batch_index, video_id in enumerate(batch.video_ids):
            valid_mask = batch.step_mask[batch_index]
            timestamps = batch.timestamps_s[batch_index][valid_mask].tolist()
            scores = probs[batch_index][valid_mask].tolist()
            per_video_scores[video_id].extend(zip(timestamps, scores))

    prediction_records: list[PredictionRecord] = []
    for video_id, annotation in annotation_by_id.items():
        merged: dict[float, list[float]] = defaultdict(list)
        for timestamp, score in per_video_scores.get(video_id, []):
            merged[float(timestamp)].append(float(score))
        timestamps = sorted(merged)
        scores = [sum(merged[t]) / len(merged[t]) for t in timestamps]
        post = predict_start_time(
            scores,
            timestamps,
            tau_empty=config.postprocess.default_tau_empty,
            tau_start=config.postprocess.default_tau_start,
            median_kernel_size=config.postprocess.median_kernel_size,
        )
        prediction_records.append(
            PredictionRecord(
                video_id=video_id,
                is_positive=annotation.is_positive,
                ground_truth_start_s=float(annotation.start_s) if annotation.start_s is not None else None,
                predicted_start_s=post.predicted_start_s,
            )
        )

    metrics = compute_contest_metrics(prediction_records)
    return {
        "loss": total_loss / max(1, num_batches),
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
    }


def main() -> None:
    args = parse_args()
    config = ExperimentConfig().resolved()
    print_device_diagnostics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_dataloaders(config, args)

    model = ClipOnsetModel(backbone_name=config.model.backbone_name).to(device)
    optimizer = build_optimizer(model, config)

    best_f1 = -1.0
    best_metrics: dict[str, float] = {}
    val_annotations: list[VideoAnnotation] = []
    if config.paths.val_videos_dir.exists():
        val_annotations = build_split(
            csv_path=config.paths.val_csv,
            videos_dir=config.paths.val_videos_dir,
            split="val",
            limit=args.max_val_videos,
        )

    for epoch in range(1, config.stage1.num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp=config.stage1.amp,
        )
        print(f"epoch={epoch} train_loss={train_loss:.4f}")

        if val_loader is None:
            continue
        metrics = validate(
            model=model,
            loader=val_loader,
            annotations=val_annotations,
            config=config,
            device=device,
        )
        print(
            " ".join(
                [
                    f"epoch={epoch}",
                    f"val_loss={metrics['loss']:.4f}",
                    f"precision={metrics['precision']:.4f}",
                    f"recall={metrics['recall']:.4f}",
                    f"f1={metrics['f1_score']:.4f}",
                ]
            )
        )
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_metrics = metrics
            payload = checkpoint_payload(
                model_state_dict=model.state_dict(),
                config=config,
                epoch=epoch,
                metrics=metrics,
                extra={
                    "stage": 1,
                    "tau_empty": config.postprocess.default_tau_empty,
                    "tau_start": config.postprocess.default_tau_start,
                },
            )
            save_checkpoint(config.paths.checkpoints_dir / args.output_name, payload)

    if best_metrics:
        print(f"best_f1={best_metrics['f1_score']:.4f}")
    else:
        payload = checkpoint_payload(
            model_state_dict=model.state_dict(),
            config=config,
            epoch=config.stage1.num_epochs,
            metrics={},
            extra={"stage": 1},
        )
        save_checkpoint(config.paths.checkpoints_dir / args.output_name, payload)


if __name__ == "__main__":
    main()
