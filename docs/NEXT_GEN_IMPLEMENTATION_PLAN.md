# Next-Gen Architecture And Implementation Plan

## Current Best Baseline

- Stage 1 warm-started `VideoMAE-base` encoder
- Sampling:
  - `8 FPS`
  - `16` frames per clip
  - `4` sampled-frame stride
  - clip span `2.0s`
  - step spacing `0.5s`
- Stage 2 winner:
  - temporal model: `conv`
  - target mode: `cumulative`
  - `max_steps=12`
  - `window_stride=6`
  - `lambda_video=0.5`
  - `monotonic_loss_weight=0.05`
- Best cached validation result so far:
  - `F1 = 0.7756`

## Why The Baseline Saturates

The current system is strong but bottlenecked by three issues:

1. RGB-only clip embeddings do not explicitly model motion discontinuities.
2. Cumulative supervision teaches stable post-onset detection, but weakly teaches temporal progression.
3. Precision is still the limiting factor, so false-positive negatives need targeted reweighting.

## Implemented Next-Gen Upgrade

This repo now includes the first practical upgrade path that does not destabilize the current baseline.

### 1. Auxiliary Temporal-Progression Head

The temporal heads now predict:

- step logits
- video logits
- temporal-distance bin logits

Temporal-distance bins are defined relative to the annotated onset:

- `0`: far before (`t - onset < -5s`)
- `1`: near before (`-5s <= t - onset < 0s`)
- `2`: near after (`0s <= t - onset <= 5s`)
- `3`: far after (`t - onset > 5s`)

Negative videos are ignored for this auxiliary loss.

This keeps the cumulative objective as the main contest-aligned target while adding progression supervision.

### 2. Hard-Negative Weighting Hooks

Cached Stage 2 training now supports:

- a text file of hard-negative video IDs
- a configurable multiplier applied on top of video-balanced sampling

This lets us upweight negatives that the current model falsely activates on.

### 3. Hard-Negative Mining Utility

Added a script to mine false-positive negatives from a cached checkpoint:

- `aid-mine-hard-negatives`

This uses a cached Stage 2 checkpoint and the saved decoding thresholds to produce a list of negative video IDs that still trigger falsely.

## New Training Objective

For Stage 2, the total loss is now:

`L = L_step + lambda_video * L_video + lambda_monotonic * L_monotonic + lambda_aux * L_temporal_bin`

Where:

- `L_step`: BCE on cumulative targets
- `L_video`: BCE on video-level incident prediction
- `L_monotonic`: penalty on downward probability transitions
- `L_temporal_bin`: cross-entropy on temporal-distance bins for positive videos

## Recommended Experiment Order

### Wave 1: Auxiliary Supervision Only

Start from the current best recipe and sweep `temporal_aux_loss_weight`:

- `0.0`
- `0.1`
- `0.2`
- `0.3`

Keep fixed:

- `conv`
- `cumulative`
- `ms12`
- `lambda_video=0.5`
- `monotonic_loss_weight=0.05`

### Wave 2: Hard-Negative Mining

1. Train the best auxiliary-supervision model.
2. Mine false-positive negatives:

```bash
uv run aid-mine-hard-negatives --checkpoint artifacts/checkpoints/<best>.pt --cache-root artifacts/features/stage2_cache/stage1_best --split train --output-path artifacts/hard_negatives/train_fp_ids.txt
```

3. Retrain with:

- `--hard-negative-ids-path artifacts/hard_negatives/train_fp_ids.txt`
- `--hard-negative-multiplier 2.0` or `3.0`

### Wave 3: Motion Branch

Only after exhausting the auxiliary + hard-negative path:

- add a separate motion encoder over frame differences or optical flow
- fuse motion embedding with the RGB clip embedding before the temporal head

This is the next likely step-change improvement, but it is intentionally not the first upgrade.

## Why This Plan

This sequence is chosen to maximize ROI while preserving the current working system:

- keep Stage 1 warm-start
- keep cumulative detection
- keep the winning conv head
- add richer supervision before rewriting the visual front-end
- add hard-negative pressure before moving to a larger architectural branch

## Commands Summary

Feature extraction:

```bash
uv run aid-extract-features --encoder-checkpoint artifacts/checkpoints/stage1_best.pt --split all --chunk-steps 64 --clip-batch-size 32
```

Train cached Stage 2 with auxiliary supervision:

```bash
uv run aid-stage2-cached --run-name cached_conv_cum_ms12_m005_aux020 --cache-root artifacts/features/stage2_cache/stage1_best --temporal-model conv --target-mode cumulative --lambda-video 0.5 --monotonic-loss-weight 0.05 --temporal-aux-loss-weight 0.2 --batch-size 8 --max-steps 12 --window-stride 6 --num-workers 2 --num-epochs 20 --log-every 50 --validate-every 1 --output-name cached_conv_cum_ms12_m005_aux020.pt
```

Mine hard negatives:

```bash
uv run aid-mine-hard-negatives --checkpoint artifacts/checkpoints/cached_conv_cum_ms12_m005_aux020.pt --cache-root artifacts/features/stage2_cache/stage1_best --split train --output-path artifacts/hard_negatives/train_fp_ids.txt
```

Retrain with hard negatives:

```bash
uv run aid-stage2-cached --run-name cached_conv_cum_ms12_m005_aux020_hn300 --cache-root artifacts/features/stage2_cache/stage1_best --temporal-model conv --target-mode cumulative --lambda-video 0.5 --monotonic-loss-weight 0.05 --temporal-aux-loss-weight 0.2 --hard-negative-ids-path artifacts/hard_negatives/train_fp_ids.txt --hard-negative-multiplier 3.0 --batch-size 8 --max-steps 12 --window-stride 6 --num-workers 2 --num-epochs 20 --log-every 50 --validate-every 1 --output-name cached_conv_cum_ms12_m005_aux020_hn300.pt
```
