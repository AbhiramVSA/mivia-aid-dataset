from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import ExperimentConfig
from src.data.cached_sequence_dataset import CachedSequenceDataset, collate_cached_sequence_batch
from src.models.temporal_only_model import TemporalOnlyModel
from src.train_stage2 import (
    build_scheduler,
    compute_stage2_loss,
    current_gpu_memory_gb,
    format_duration,
    log_prefix,
    print_device_diagnostics,
    sweep_postprocess_thresholds,
)
from src.utils.checkpoint import checkpoint_payload, save_checkpoint
from src.utils.training import make_generator, make_worker_init_fn, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 temporal head from cached embeddings")
    parser.add_argument("--run-name", type=str, default="stage2_cached")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-name", type=str, default="stage2_cached_best.pt")
    parser.add_argument("--lambda-video", type=float, default=0.5)
    parser.add_argument("--temporal-model", type=str, choices=("conv", "transformer"), default="transformer")
    parser.add_argument("--target-mode", type=str, choices=("cumulative", "onset"), default="cumulative")
    parser.add_argument("--use-motion-branch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--window-stride", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--min-recall-for-selection", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scheduler", type=str, choices=("cosine", "none"), default=None)
    parser.add_argument("--monotonic-loss-weight", type=float, default=None)
    parser.add_argument("--temporal-aux-loss-weight", type=float, default=None)
    parser.add_argument("--disable-video-balanced-sampling", action="store_true")
    parser.add_argument("--hard-negative-ids-path", type=Path, default=None)
    parser.add_argument("--hard-negative-multiplier", type=float, default=None)
    return parser.parse_args()


def make_dataloaders(
    config: ExperimentConfig, args: argparse.Namespace
) -> tuple[CachedSequenceDataset, DataLoader, CachedSequenceDataset, DataLoader]:
    generator = make_generator(config.stage2.seed)
    worker_init_fn = make_worker_init_fn(config.stage2.seed)
    hard_negative_video_ids: set[str] = set()
    if args.hard_negative_ids_path is not None and args.hard_negative_ids_path.exists():
        hard_negative_video_ids = {
            line.strip() for line in args.hard_negative_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()
        }
    train_dataset = CachedSequenceDataset(
        cache_dir=args.cache_root / "train",
        max_steps=config.stage2.max_steps_per_sample,
        window_stride=config.stage2.window_stride_steps,
        hard_negative_video_ids=hard_negative_video_ids,
        hard_negative_multiplier=config.stage2.hard_negative_multiplier,
    )
    val_dataset = CachedSequenceDataset(
        cache_dir=args.cache_root / "val",
        max_steps=config.stage2.max_steps_per_sample,
        window_stride=config.stage2.window_stride_steps,
    )
    train_sampler = None
    shuffle = True
    if not args.disable_video_balanced_sampling:
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.stage2.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config.stage2.num_workers,
        pin_memory=True,
        persistent_workers=config.stage2.num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=generator if shuffle else None,
        collate_fn=collate_cached_sequence_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.stage2.batch_size,
        shuffle=False,
        num_workers=config.stage2.num_workers,
        pin_memory=True,
        persistent_workers=config.stage2.num_workers > 0,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_cached_sequence_batch,
    )
    return train_dataset, train_loader, val_dataset, val_loader


def print_dataset_summary(name: str, dataset: CachedSequenceDataset, loader: DataLoader) -> None:
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
    if name == "train":
        total_windows = max(1, len(dataset))
        raw_fraction = dataset.hard_negative_window_count / float(total_windows)
        print(
            " ".join(
                [
                    f"{name}_hard_negative_video_count={len(dataset.hard_negative_video_ids)}",
                    f"{name}_hard_negative_window_count={dataset.hard_negative_window_count}",
                    f"{name}_hard_negative_window_fraction={raw_fraction:.4f}",
                    f"{name}_hard_negative_sampling_mass_fraction={dataset.hard_negative_sampling_mass_fraction:.4f}",
                    f"{name}_temporal_bin_hist={dataset.temporal_bin_hist}",
                ]
            )
        )
    else:
        print(f"{name}_temporal_bin_hist={dataset.temporal_bin_hist}")


def train_one_epoch(
    model: TemporalOnlyModel,
    loader: DataLoader,
    config: ExperimentConfig,
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
        features = batch.features.to(device=device, dtype=torch.float32, non_blocking=True)
        motion_features = batch.motion_features.to(device=device, dtype=torch.float32, non_blocking=True)
        selected_targets = (
            batch.onset_targets.to(device, non_blocking=True)
            if target_mode == "onset"
            else batch.step_targets.to(device, non_blocking=True)
        )
        temporal_bin_targets = batch.temporal_bin_targets.to(device, non_blocking=True)
        step_mask = batch.step_mask.to(device, non_blocking=True)
        video_target = batch.video_target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=True) if amp and device.type == "cuda" else nullcontext()
        )
        with autocast_context:
            step_logits, video_logits, temporal_bin_logits = model(
                features,
                step_mask=step_mask,
                motion_features=motion_features,
            )
            loss = compute_stage2_loss(
                step_logits=step_logits,
                video_logits=video_logits,
                temporal_bin_logits=temporal_bin_logits,
                step_targets=selected_targets,
                temporal_bin_targets=temporal_bin_targets,
                step_mask=step_mask,
                video_target=video_target,
                lambda_video=lambda_video,
                target_mode=target_mode,
                monotonic_loss_weight=config.stage2.monotonic_loss_weight,
                temporal_aux_loss_weight=config.stage2.temporal_aux_loss_weight,
            )
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())
        num_batches += 1
        if log_every > 0 and (batch_index % log_every == 0 or batch_index == len(loader)):
            now = perf_counter()
            print(
                " ".join(
                    [
                        log_prefix(run_name),
                        "phase=train_cached",
                        f"epoch={epoch}/{total_epochs}",
                        f"batch={batch_index}/{len(loader)}",
                        f"avg_loss={running_loss / num_batches:.4f}",
                        f"batch_loss={loss.item():.4f}",
                        f"data_time_s={data_ready_time - batch_timer:.2f}",
                        f"step_time_s={now - data_ready_time:.2f}",
                        f"elapsed={format_duration(now - epoch_start)}",
                        f"gpu_mem_gb={current_gpu_memory_gb(device):.2f}",
                    ]
                )
            )
            batch_timer = perf_counter()
    return running_loss / max(1, num_batches)


@torch.no_grad()
def validate(
    model: TemporalOnlyModel,
    loader: DataLoader,
    dataset: CachedSequenceDataset,
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
    per_video_scores: dict[str, list[tuple[float, float]]] = {}
    per_video_probs: dict[str, list[float]] = {}
    annotations = []
    total_loss = 0.0
    num_batches = 0
    val_start = perf_counter()
    batch_timer = perf_counter()

    annotations.extend(dataset.annotation_metas)

    for batch_index, batch in enumerate(loader, start=1):
        data_ready_time = perf_counter()
        features = batch.features.to(device=device, dtype=torch.float32, non_blocking=True)
        motion_features = batch.motion_features.to(device=device, dtype=torch.float32, non_blocking=True)
        selected_targets = (
            batch.onset_targets.to(device, non_blocking=True)
            if target_mode == "onset"
            else batch.step_targets.to(device, non_blocking=True)
        )
        temporal_bin_targets = batch.temporal_bin_targets.to(device, non_blocking=True)
        step_mask = batch.step_mask.to(device, non_blocking=True)
        video_target = batch.video_target.to(device, non_blocking=True)
        step_logits, video_logits, temporal_bin_logits = model(
            features,
            step_mask=step_mask,
            motion_features=motion_features,
        )
        loss = compute_stage2_loss(
            step_logits=step_logits,
            video_logits=video_logits,
            temporal_bin_logits=temporal_bin_logits,
            step_targets=selected_targets,
            temporal_bin_targets=temporal_bin_targets,
            step_mask=step_mask,
            video_target=video_target,
            lambda_video=lambda_video,
            target_mode=target_mode,
            monotonic_loss_weight=config.stage2.monotonic_loss_weight,
            temporal_aux_loss_weight=config.stage2.temporal_aux_loss_weight,
        )
        total_loss += float(loss.item())
        num_batches += 1
        probs = torch.sigmoid(step_logits).cpu()
        video_probs = torch.sigmoid(video_logits).cpu()
        for sample_index, video_id in enumerate(batch.video_ids):
            valid_mask = batch.step_mask[sample_index]
            timestamps = batch.timestamps_s[sample_index][valid_mask].tolist()
            scores = probs[sample_index][valid_mask].tolist()
            per_video_scores.setdefault(video_id, []).extend(zip(timestamps, scores))
            per_video_probs.setdefault(video_id, []).append(float(video_probs[sample_index].item()))
        if log_every > 0 and (batch_index % log_every == 0 or batch_index == len(loader)):
            now = perf_counter()
            print(
                " ".join(
                    [
                        log_prefix(run_name),
                        "phase=val_cached",
                        f"epoch={epoch}/{total_epochs}",
                        f"batch={batch_index}/{len(loader)}",
                        f"avg_loss={total_loss / num_batches:.4f}",
                        f"data_time_s={data_ready_time - batch_timer:.2f}",
                        f"step_time_s={now - data_ready_time:.2f}",
                        f"elapsed={format_duration(now - val_start)}",
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
        "selection_min_recall": sweep_metrics["selection_min_recall"],
        "selection_recall_floor_met": sweep_metrics["selection_recall_floor_met"],
    }


def main() -> None:
    args = parse_args()
    config = ExperimentConfig().resolved()
    config.stage2.batch_size = args.batch_size
    config.stage2.max_steps_per_sample = args.max_steps
    config.stage2.window_stride_steps = args.window_stride
    config.stage2.num_workers = args.num_workers
    config.stage2.num_epochs = args.num_epochs
    config.stage2.target_mode = args.target_mode
    config.model.temporal_model = args.temporal_model
    config.model.use_motion_branch = args.use_motion_branch
    if args.monotonic_loss_weight is not None:
        config.stage2.monotonic_loss_weight = args.monotonic_loss_weight
    if args.temporal_aux_loss_weight is not None:
        config.stage2.temporal_aux_loss_weight = args.temporal_aux_loss_weight
    if args.hard_negative_multiplier is not None:
        config.stage2.hard_negative_multiplier = args.hard_negative_multiplier
    if args.early_stopping_patience is not None:
        config.stage2.early_stopping_patience = args.early_stopping_patience
    if args.min_recall_for_selection is not None:
        config.postprocess.selection_min_recall = args.min_recall_for_selection
    if args.seed is not None:
        config.stage2.seed = args.seed
    if args.scheduler is not None:
        config.stage2.scheduler_name = args.scheduler
    config.postprocess.prediction_mode = "peak" if args.target_mode == "onset" else "cumulative"
    seed_everything(config.stage2.seed)
    print_device_diagnostics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, train_loader, val_dataset, val_loader = make_dataloaders(config, args)
    print(
        " ".join(
            [
                log_prefix(args.run_name),
                f"cache_root={args.cache_root}",
                f"temporal_model={config.model.temporal_model}",
                f"use_motion_branch={config.model.use_motion_branch}",
                f"target_mode={config.stage2.target_mode}",
                f"lambda_video={args.lambda_video}",
                f"monotonic_weight={config.stage2.monotonic_loss_weight}",
                f"temporal_aux_weight={config.stage2.temporal_aux_loss_weight}",
                f"video_balanced_sampling={not args.disable_video_balanced_sampling}",
                f"hard_negative_multiplier={config.stage2.hard_negative_multiplier}",
                f"early_stopping_patience={config.stage2.early_stopping_patience}",
                f"selection_min_recall={config.postprocess.selection_min_recall}",
                f"seed={config.stage2.seed}",
                f"scheduler={config.stage2.scheduler_name}",
                f"batch_size={config.stage2.batch_size}",
                f"max_steps={config.stage2.max_steps_per_sample}",
                f"window_stride={config.stage2.window_stride_steps}",
                f"num_workers={config.stage2.num_workers}",
            ]
        )
    )
    print(log_prefix(args.run_name), end=" ")
    print_dataset_summary("train", train_dataset, train_loader)
    print(log_prefix(args.run_name), end=" ")
    print_dataset_summary("val", val_dataset, val_loader)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.stage2.head_lr, weight_decay=config.stage2.weight_decay)
    scheduler = build_scheduler(optimizer, config)

    best_f1 = -1.0
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    validations_without_improvement = 0
    selection_floor_ever_met = False
    for epoch in range(1, config.stage2.num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            config=config,
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
        if args.validate_every > 1 and epoch % args.validate_every != 0:
            print(f"{log_prefix(args.run_name)} epoch={epoch} validation=skipped validate_every={args.validate_every}")
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[-1]["lr"]
                print(f"{log_prefix(args.run_name)} epoch={epoch} lr={current_lr:.6e}")
            continue
        metrics = validate(
            model=model,
            loader=val_loader,
            dataset=val_dataset,
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
                    f"selection_recall_floor_met={bool(metrics['selection_recall_floor_met'])}",
                ]
            )
        )
        floor_met = bool(metrics["selection_recall_floor_met"])
        if floor_met:
            selection_floor_ever_met = True
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_metrics = metrics
            best_epoch = epoch
            validations_without_improvement = 0
            payload = checkpoint_payload(
                model_state_dict=model.state_dict(),
                config=config,
                epoch=epoch,
                metrics=metrics,
                extra={
                    "stage": "stage2_cached",
                    "encoder_checkpoint": f"artifacts/checkpoints/{args.cache_root.name}.pt",
                    "lambda_video": args.lambda_video,
                    "tau_empty": metrics["tau_empty"],
                    "tau_start": metrics["tau_start"],
                    "tau_keep": metrics["tau_keep"],
                    "tau_video": metrics["tau_video"],
                    "min_consecutive_steps": int(metrics["min_consecutive_steps"]),
                    "prediction_mode": config.postprocess.prediction_mode,
                    "selection_min_recall": config.postprocess.selection_min_recall,
                    "selection_recall_floor_met": bool(metrics["selection_recall_floor_met"]),
                },
            )
            save_checkpoint(config.paths.checkpoints_dir / args.output_name, payload)
        else:
            if config.postprocess.selection_min_recall is None or selection_floor_ever_met:
                validations_without_improvement += 1
            else:
                print(
                    " ".join(
                        [
                            log_prefix(args.run_name),
                            f"epoch={epoch}",
                            "early_stopping=deferred",
                            f"selection_recall_floor_met={floor_met}",
                            f"selection_min_recall={config.postprocess.selection_min_recall}",
                        ]
                    )
                )

        patience = config.stage2.early_stopping_patience
        can_stop = config.postprocess.selection_min_recall is None or selection_floor_ever_met
        if patience is not None and patience >= 0 and can_stop and validations_without_improvement >= patience:
            print(
                " ".join(
                    [
                        log_prefix(args.run_name),
                        f"early_stopping=triggered",
                        f"epoch={epoch}",
                        f"best_epoch={best_epoch}",
                        f"best_f1={best_f1:.4f}",
                        f"patience={patience}",
                    ]
                )
            )
            break
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[-1]["lr"]
            print(f"{log_prefix(args.run_name)} epoch={epoch} lr={current_lr:.6e}")
    if best_metrics:
        print(f"{log_prefix(args.run_name)} best_epoch={best_epoch} best_f1={best_metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
