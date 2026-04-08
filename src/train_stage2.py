from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

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
    parser.add_argument("--run-name", type=str, default="stage2")
    parser.add_argument("--output-name", type=str, default="stage2_best.pt")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--lambda-video", type=float, default=0.5)
    parser.add_argument("--temporal-model", type=str, choices=("conv", "transformer"), default=None)
    parser.add_argument("--target-mode", type=str, choices=("cumulative", "onset"), default=None)
    parser.add_argument("--max-train-videos", type=int, default=None)
    parser.add_argument("--max-val-videos", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=1)
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


def log_prefix(run_name: str) -> str:
    return f"[{run_name}]"


def format_duration(seconds: float) -> str:
    total = int(max(0.0, round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def current_gpu_memory_gb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024**3)


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


def make_dataloaders(
    config: ExperimentConfig, args: argparse.Namespace
) -> tuple[Stage2SequenceDataset, DataLoader, Stage2SequenceDataset | None, DataLoader | None]:
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
        onset_sigma_s=config.stage2.onset_sigma_seconds if config.stage2.target_mode == "onset" else None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.stage2.batch_size,
        shuffle=True,
        num_workers=config.stage2.num_workers,
        pin_memory=True,
        persistent_workers=config.stage2.num_workers > 0,
        collate_fn=collate_sequence_batch,
    )

    val_dataset: Stage2SequenceDataset | None = None
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
            onset_sigma_s=config.stage2.onset_sigma_seconds if config.stage2.target_mode == "onset" else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.stage2.batch_size,
            shuffle=False,
            num_workers=config.stage2.num_workers,
            pin_memory=True,
            persistent_workers=config.stage2.num_workers > 0,
            collate_fn=collate_sequence_batch,
        )
    return train_dataset, train_loader, val_dataset, val_loader


def print_dataset_summary(name: str, dataset: Stage2SequenceDataset, loader: DataLoader) -> None:
    print(
        " ".join(
            [
                f"{name}_videos={dataset.num_videos}",
                f"{name}_windows={len(dataset)}",
                f"{name}_batches={len(loader)}",
                f"{name}_avg_window_steps={dataset.avg_window_steps:.2f}",
                f"{name}_max_window_steps={dataset.max_window_steps}",
            ]
        )
    )


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
    target_mode: str,
    monotonic_loss_weight: float,
) -> torch.Tensor:
    step_loss = F.binary_cross_entropy_with_logits(step_logits, step_targets, reduction="none")
    valid = step_mask.bool()
    step_loss = (step_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    step_loss_mean = step_loss.mean()
    video_loss = F.binary_cross_entropy_with_logits(video_logits, video_target)
    monotonic_loss = step_logits.new_tensor(0.0)
    if target_mode == "cumulative" and monotonic_loss_weight > 0:
        step_probs = torch.sigmoid(step_logits)
        pair_valid = valid[:, :-1] & valid[:, 1:]
        if pair_valid.any():
            downward = torch.relu(step_probs[:, :-1] - step_probs[:, 1:])
            monotonic_loss = downward[pair_valid].mean()
    return step_loss_mean + lambda_video * video_loss + monotonic_loss_weight * monotonic_loss


def sweep_postprocess_thresholds(
    *,
    annotations: list[VideoAnnotation],
    config: ExperimentConfig,
    per_video_scores: dict[str, list[tuple[float, float]]],
    per_video_probs: dict[str, list[float]],
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for tau_empty in config.postprocess.tau_empty_grid:
        for tau_start in config.postprocess.tau_start_grid:
            for tau_keep in config.postprocess.tau_keep_grid:
                if tau_keep > tau_start:
                    continue
                for tau_video in config.postprocess.tau_video_grid:
                    for min_consecutive_steps in config.postprocess.consecutive_hits_grid:
                        prediction_records: list[PredictionRecord] = []
                        for annotation in annotations:
                            merged: dict[float, list[float]] = defaultdict(list)
                            for timestamp, score in per_video_scores.get(annotation.video_id, []):
                                merged[float(timestamp)].append(float(score))
                            timestamps = sorted(merged)
                            scores = [sum(merged[t]) / len(merged[t]) for t in timestamps]
                            video_prob_list = per_video_probs.get(annotation.video_id, [])
                            video_score = (
                                sum(video_prob_list) / len(video_prob_list) if video_prob_list else 0.0
                            )
                            post = predict_start_time(
                                scores,
                                timestamps,
                                tau_empty=float(tau_empty),
                                tau_start=float(tau_start),
                                tau_keep=float(tau_keep),
                                tau_video=float(tau_video),
                                video_score=video_score,
                                median_kernel_size=config.postprocess.median_kernel_size,
                                min_consecutive_steps=int(min_consecutive_steps),
                                mode=config.postprocess.prediction_mode,
                            )
                            prediction_records.append(
                                PredictionRecord(
                                    video_id=annotation.video_id,
                                    is_positive=annotation.is_positive,
                                    ground_truth_start_s=float(annotation.start_s)
                                    if annotation.start_s is not None
                                    else None,
                                    predicted_start_s=post.predicted_start_s,
                                )
                            )
                        contest_metrics = compute_contest_metrics(prediction_records)
                        candidate = {
                            "precision": contest_metrics.precision,
                            "recall": contest_metrics.recall,
                            "f1_score": contest_metrics.f1_score,
                            "tp": float(contest_metrics.true_positives),
                            "fp": float(contest_metrics.false_positives),
                            "fn": float(contest_metrics.false_negatives),
                            "tau_empty": float(tau_empty),
                            "tau_start": float(tau_start),
                            "tau_keep": float(tau_keep),
                            "tau_video": float(tau_video),
                            "min_consecutive_steps": float(min_consecutive_steps),
                        }
                        if best is None:
                            best = candidate
                            continue
                        if candidate["f1_score"] > best["f1_score"]:
                            best = candidate
                            continue
                        if candidate["f1_score"] == best["f1_score"] and candidate["precision"] > best["precision"]:
                            best = candidate
                            continue
                        if (
                            candidate["f1_score"] == best["f1_score"]
                            and candidate["precision"] == best["precision"]
                            and candidate["recall"] > best["recall"]
                        ):
                            best = candidate
    assert best is not None
    return best


def train_one_epoch(
    model: AIDTemporalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    lambda_video: float,
    epoch: int,
    total_epochs: int,
    log_every: int,
    run_name: str,
    target_mode: str,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0
    epoch_start = perf_counter()
    batch_timer = perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        data_ready_time = perf_counter()
        clip_tensor = batch.clip_tensor.to(device, non_blocking=True)
        selected_targets = (
            batch.onset_targets.to(device, non_blocking=True)
            if target_mode == "onset"
            else batch.step_targets.to(device, non_blocking=True)
        )
        step_mask = batch.step_mask.to(device, non_blocking=True)
        video_target = batch.video_target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=True) if amp and device.type == "cuda" else nullcontext()
        )
        with autocast_context:
            step_logits, video_logits = model(clip_tensor, step_mask=step_mask)
            loss = compute_stage2_loss(
                step_logits=step_logits,
                video_logits=video_logits,
                step_targets=selected_targets,
                step_mask=step_mask,
                video_target=video_target,
                lambda_video=lambda_video,
                target_mode=target_mode,
                monotonic_loss_weight=config.stage2.monotonic_loss_weight,
            )
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1
        if log_every > 0 and (batch_index % log_every == 0 or batch_index == len(loader)):
            now = perf_counter()
            data_time = data_ready_time - batch_timer
            step_time = now - data_ready_time
            elapsed = now - epoch_start
            avg_loss = running_loss / num_batches
            valid_steps = int(step_mask.sum().item())
            print(
                " ".join(
                    [
                        log_prefix(run_name),
                        "phase=train",
                        f"epoch={epoch}/{total_epochs}",
                        f"batch={batch_index}/{len(loader)}",
                        f"avg_loss={avg_loss:.4f}",
                        f"batch_loss={loss.item():.4f}",
                        f"valid_steps={valid_steps}",
                        f"data_time_s={data_time:.2f}",
                        f"step_time_s={step_time:.2f}",
                        f"elapsed={format_duration(elapsed)}",
                        f"gpu_mem_gb={current_gpu_memory_gb(device):.2f}",
                    ]
                )
            )
            batch_timer = perf_counter()
    return running_loss / max(1, num_batches)


@torch.no_grad()
def validate(
    model: AIDTemporalModel,
    loader: DataLoader,
    annotations: list[VideoAnnotation],
    config: ExperimentConfig,
    device: torch.device,
    lambda_video: float,
    epoch: int,
    total_epochs: int,
    log_every: int,
    run_name: str,
    target_mode: str,
) -> dict[str, float]:
    model.eval()
    per_video_scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_video_probs: dict[str, list[float]] = defaultdict(list)
    total_loss = 0.0
    num_batches = 0
    val_start = perf_counter()
    batch_timer = perf_counter()

    for batch_index, batch in enumerate(loader, start=1):
        data_ready_time = perf_counter()
        clip_tensor = batch.clip_tensor.to(device, non_blocking=True)
        selected_targets = (
            batch.onset_targets.to(device, non_blocking=True)
            if target_mode == "onset"
            else batch.step_targets.to(device, non_blocking=True)
        )
        step_mask = batch.step_mask.to(device, non_blocking=True)
        video_target = batch.video_target.to(device, non_blocking=True)

        step_logits, video_logits = model(clip_tensor, step_mask=step_mask)
        loss = compute_stage2_loss(
            step_logits=step_logits,
            video_logits=video_logits,
            step_targets=selected_targets,
            step_mask=step_mask,
            video_target=video_target,
            lambda_video=lambda_video,
            target_mode=target_mode,
            monotonic_loss_weight=config.stage2.monotonic_loss_weight,
        )
        total_loss += float(loss.item())
        num_batches += 1

        probs = torch.sigmoid(step_logits).cpu()
        video_probs = torch.sigmoid(video_logits).cpu()
        for sample_index, video_id in enumerate(batch.video_ids):
            valid_mask = batch.step_mask[sample_index]
            timestamps = batch.timestamps_s[sample_index][valid_mask].tolist()
            scores = probs[sample_index][valid_mask].tolist()
            per_video_scores[video_id].extend(zip(timestamps, scores))
            per_video_probs[video_id].append(float(video_probs[sample_index].item()))

        if log_every > 0 and (batch_index % log_every == 0 or batch_index == len(loader)):
            now = perf_counter()
            data_time = data_ready_time - batch_timer
            step_time = now - data_ready_time
            elapsed = now - val_start
            print(
                " ".join(
                    [
                        log_prefix(run_name),
                        "phase=val",
                        f"epoch={epoch}/{total_epochs}",
                        f"batch={batch_index}/{len(loader)}",
                        f"avg_loss={total_loss / num_batches:.4f}",
                        f"data_time_s={data_time:.2f}",
                        f"step_time_s={step_time:.2f}",
                        f"elapsed={format_duration(elapsed)}",
                        f"gpu_mem_gb={current_gpu_memory_gb(device):.2f}",
                    ]
                )
            )
            batch_timer = perf_counter()

    sweep_metrics = sweep_postprocess_thresholds(
        annotations=annotations,
        config=config,
        per_video_scores=per_video_scores,
        per_video_probs=per_video_probs,
    )
    return {
        "loss": total_loss / max(1, num_batches),
        "precision": sweep_metrics["precision"],
        "recall": sweep_metrics["recall"],
        "f1_score": sweep_metrics["f1_score"],
        "tau_empty": sweep_metrics["tau_empty"],
        "tau_start": sweep_metrics["tau_start"],
        "tau_keep": sweep_metrics["tau_keep"],
        "tau_video": sweep_metrics["tau_video"],
        "min_consecutive_steps": sweep_metrics["min_consecutive_steps"],
    }


def main() -> None:
    args = parse_args()
    config = ExperimentConfig().resolved()
    if args.batch_size is not None:
        config.stage2.batch_size = args.batch_size
    if args.temporal_model is not None:
        config.model.temporal_model = args.temporal_model
    if args.target_mode is not None:
        config.stage2.target_mode = args.target_mode
    if args.max_steps is not None:
        config.stage2.max_steps_per_sample = args.max_steps
    if args.window_stride is not None:
        config.stage2.window_stride_steps = args.window_stride
    if args.num_workers is not None:
        config.stage2.num_workers = args.num_workers
    if args.num_epochs is not None:
        config.stage2.num_epochs = args.num_epochs
    config.postprocess.prediction_mode = "peak" if config.stage2.target_mode == "onset" else "cumulative"
    print_device_diagnostics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, train_loader, val_dataset, val_loader = make_dataloaders(config, args)
    print(
        " ".join(
            [
                log_prefix(args.run_name),
                f"temporal_model={config.model.temporal_model}",
                f"target_mode={config.stage2.target_mode}",
                f"lambda_video={args.lambda_video}",
                f"monotonic_weight={config.stage2.monotonic_loss_weight}",
                f"batch_size={config.stage2.batch_size}",
                f"max_steps={config.stage2.max_steps_per_sample}",
                f"window_stride={config.stage2.window_stride_steps}",
                f"num_workers={config.stage2.num_workers}",
                f"num_epochs={config.stage2.num_epochs}",
                f"init_checkpoint={'none' if args.init_checkpoint is None else args.init_checkpoint.name}",
            ]
        )
    )
    print(log_prefix(args.run_name), end=" ")
    print_dataset_summary("train", train_dataset, train_loader)
    if val_dataset is not None and val_loader is not None:
        print(log_prefix(args.run_name), end=" ")
        print_dataset_summary("val", val_dataset, val_loader)

    model = AIDTemporalModel(
        backbone_name=config.model.backbone_name,
        hidden_size=config.model.hidden_size,
        temporal_model=config.model.temporal_model,
        temporal_channels=config.model.temporal_channels,
        dropout=config.model.dropout,
        transformer_layers=config.model.transformer_layers,
        transformer_heads=config.model.transformer_heads,
        transformer_ffn_dim=config.model.transformer_ffn_dim,
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
            epoch=epoch,
            total_epochs=config.stage2.num_epochs,
            log_every=args.log_every,
            run_name=args.run_name,
            target_mode=config.stage2.target_mode,
        )
        print(f"{log_prefix(args.run_name)} epoch={epoch} train_loss={train_loss:.4f}")

        if val_loader is None:
            continue
        if args.validate_every > 1 and epoch % args.validate_every != 0:
            print(
                f"{log_prefix(args.run_name)} epoch={epoch} validation=skipped validate_every={args.validate_every}"
            )
            continue
        metrics = validate(
            model=model,
            loader=val_loader,
            annotations=val_annotations,
            config=config,
            device=device,
            lambda_video=args.lambda_video,
            epoch=epoch,
            total_epochs=config.stage2.num_epochs,
            log_every=args.log_every,
            run_name=args.run_name,
            target_mode=config.stage2.target_mode,
        )
        print(
            " ".join(
                [
                    log_prefix(args.run_name),
                    f"epoch={epoch}",
                    f"val_loss={metrics['loss']:.4f}",
                    f"precision={metrics['precision']:.4f}",
                    f"recall={metrics['recall']:.4f}",
                    f"f1={metrics['f1_score']:.4f}",
                    f"tau_empty={metrics['tau_empty']:.2f}",
                    f"tau_start={metrics['tau_start']:.2f}",
                    f"tau_keep={metrics['tau_keep']:.2f}",
                    f"tau_video={metrics['tau_video']:.2f}",
                    f"min_consecutive={int(metrics['min_consecutive_steps'])}",
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
                    "tau_empty": metrics["tau_empty"],
                    "tau_start": metrics["tau_start"],
                    "tau_keep": metrics["tau_keep"],
                    "tau_video": metrics["tau_video"],
                    "min_consecutive_steps": int(metrics["min_consecutive_steps"]),
                    "prediction_mode": config.postprocess.prediction_mode,
                },
            )
            save_checkpoint(config.paths.checkpoints_dir / args.output_name, payload)

    if best_metrics:
        print(f"{log_prefix(args.run_name)} best_f1={best_metrics['f1_score']:.4f}")
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
