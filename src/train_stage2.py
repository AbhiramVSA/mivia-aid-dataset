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
from src.models.aid_model import AIDTemporalModel
from src.utils.checkpoint import checkpoint_payload, load_checkpoint, save_checkpoint
from src.utils.metrics import PredictionRecord, compute_contest_metrics
from src.utils.postprocess import predict_start_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 temporal fine-tuning")
    parser.add_argument("--output-name", type=str, default="stage2_best.pt")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--lambda-video", type=float, default=0.5)
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
        max_steps=config.stage2.max_steps_per_sample,
        window_stride=config.stage2.window_stride_steps,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.stage2.batch_size,
        shuffle=True,
        num_workers=config.stage2.num_workers,
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
            max_steps=config.stage2.max_steps_per_sample,
            window_stride=config.stage2.window_stride_steps,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.stage2.batch_size,
            shuffle=False,
            num_workers=config.stage2.num_workers,
            pin_memory=True,
            collate_fn=collate_sequence_batch,
        )
    return train_loader, val_loader


def build_optimizer(model: AIDTemporalModel, config: ExperimentConfig) -> torch.optim.Optimizer:
    _ = model.encoder.hidden_size
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.temporal_head.parameters())
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.stage2.backbone_lr},
            {"params": head_params, "lr": config.stage2.head_lr},
        ],
        weight_decay=config.stage2.weight_decay,
    )


def maybe_load_init_checkpoint(model: AIDTemporalModel, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    payload = load_checkpoint(checkpoint_path)
    state_dict = payload["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)


def compute_stage2_loss(
    step_logits: torch.Tensor,
    video_logits: torch.Tensor,
    step_targets: torch.Tensor,
    step_mask: torch.Tensor,
    video_target: torch.Tensor,
    lambda_video: float,
) -> torch.Tensor:
    step_loss = F.binary_cross_entropy_with_logits(step_logits, step_targets, reduction="none")
    valid = step_mask.bool()
    step_loss = (step_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    step_loss_mean = step_loss.mean()
    video_loss = F.binary_cross_entropy_with_logits(video_logits, video_target)
    return step_loss_mean + lambda_video * video_loss


def train_one_epoch(
    model: AIDTemporalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    lambda_video: float,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0
    for batch in loader:
        clip_tensor = batch.clip_tensor.to(device, non_blocking=True)
        step_targets = batch.step_targets.to(device, non_blocking=True)
        step_mask = batch.step_mask.to(device, non_blocking=True)
        video_target = batch.video_target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=True) if amp and device.type == "cuda" else nullcontext()
        )
        with autocast_context:
            step_logits, video_logits = model(clip_tensor)
            loss = compute_stage2_loss(
                step_logits=step_logits,
                video_logits=video_logits,
                step_targets=step_targets,
                step_mask=step_mask,
                video_target=video_target,
                lambda_video=lambda_video,
            )
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1
    return running_loss / max(1, num_batches)


@torch.no_grad()
def validate(
    model: AIDTemporalModel,
    loader: DataLoader,
    annotations: list[VideoAnnotation],
    config: ExperimentConfig,
    device: torch.device,
    lambda_video: float,
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
        video_target = batch.video_target.to(device, non_blocking=True)

        step_logits, video_logits = model(clip_tensor)
        loss = compute_stage2_loss(
            step_logits=step_logits,
            video_logits=video_logits,
            step_targets=step_targets,
            step_mask=step_mask,
            video_target=video_target,
            lambda_video=lambda_video,
        )
        total_loss += float(loss.item())
        num_batches += 1

        probs = torch.sigmoid(step_logits).cpu()
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

    model = AIDTemporalModel(
        backbone_name=config.model.backbone_name,
        hidden_size=config.model.hidden_size,
        temporal_channels=config.model.temporal_channels,
        dropout=config.model.dropout,
    ).to(device)
    maybe_load_init_checkpoint(model, args.init_checkpoint)
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

    for epoch in range(1, config.stage2.num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp=config.stage2.amp,
            lambda_video=args.lambda_video,
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
            lambda_video=args.lambda_video,
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
                    "stage": 2,
                    "lambda_video": args.lambda_video,
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
            epoch=config.stage2.num_epochs,
            metrics={},
            extra={"stage": 2, "lambda_video": args.lambda_video},
        )
        save_checkpoint(config.paths.checkpoints_dir / args.output_name, payload)


if __name__ == "__main__":
    main()
