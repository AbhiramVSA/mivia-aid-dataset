# Next-Gen Architecture And Implementation Plan

## Current System

The pipeline is still a two-stage system.

Stage 1:

- backbone: `MCG-NJU/videomae-base`
- input sampling:
  - `8 FPS`
  - `16` frames per clip
  - `4` sampled-frame stride inside each clip
  - clip duration `2.0s`
  - step spacing `0.5s`
- output:
  - a warm-start checkpoint used to initialize the raw Stage 2 encoder
  - a cached feature root used by the fast Stage 2 search loop

Stage 2:

- task formulation: temporal onset detection with cumulative step targets
- current stable temporal family: `conv`
- current stable search loop: cached Stage 2
- stabilizers in place:
  - fixed seeding
  - cosine LR schedule
  - monotonic loss on cumulative outputs
  - recall-constrained threshold selection
  - early stopping

## Current Best Knowledge

What has held up across reruns:

- Stage 1 warm-start is useful and should stay.
- Cumulative targets outperform onset-only targets for this contest metric.
- The conv temporal head is more stable than the small transformer here.
- Cached Stage 2 is the correct experimentation path because raw Stage 2 is dominated by decode and preprocessing cost.

What did not improve the model:

- larger hard-negative multipliers
- stronger auxiliary temporal-bin loss
- replacing conv with the current transformer head

What is now implemented beyond the original baseline:

- cached feature extraction with schema validation and `--force`
- cached motion descriptors
- seeded Stage 2 runs
- recall-constrained model selection
- motion fusion in both cached and raw Stage 2 paths
- motion-aware inference support for submission-time decoding

## Architecture Details

### Stage 1

Stage 1 trains `VideoMAE-base` as a clip-level warm-start model.

- clips are causal windows built from sampled video frames
- each step predicts whether the incident has already started by that time
- the resulting checkpoint is not the final submission model
- its practical value is encoder adaptation before Stage 2

### Raw Stage 2

Raw Stage 2 is end-to-end and now supports both RGB-only and RGB+motion.

Data path:

- decode sampled frames from video
- build causal clips over the decoded frame window
- preprocess clips for `VideoMAE`
- compute lightweight motion descriptors from raw frame differences
- batch clips into `[B, T, C, F, H, W]`
- batch motion descriptors into `[B, T, D_motion]`

Model path:

- `VideoMAEClipEncoder` produces per-step RGB embeddings
- optional `MotionFeatureFusion` maps the motion descriptor into the hidden space and fuses it with a learned gate
- the temporal head predicts:
  - step logits
  - video logits
  - temporal-bin logits

Loss:

`L = BCE_step + lambda_video * BCE_video + lambda_monotonic * monotonic_penalty + lambda_aux * CE_temporal_bins`

### Cached Stage 2

Cached Stage 2 is the primary experimentation loop.

Cached bundle fields:

- `features`
- `motion_features`
- `timestamps_s`
- `step_targets`
- `onset_targets`
- `temporal_bin_targets`
- `video_target`
- `source_positive`
- `ground_truth_start_s`
- `cache_schema_version`

Cached training keeps the same temporal model and postprocessing logic as raw Stage 2, but skips repeated video decoding and encoder forward passes.

## Motion Branch

The current motion branch is deliberately lightweight.

Descriptor:

- frame-difference based
- `24` dimensions
- captures resampled stepwise motion magnitude plus channelwise and global summary statistics

Fusion:

- motion descriptor is encoded through a small MLP bottleneck
- a learned gate modulates how much motion information is injected into each RGB feature
- the fused representation is normalized before the temporal head

This is a probe architecture, not the final motion model. If motion helps, the next step should be a learned motion encoder rather than more threshold tuning around the handcrafted descriptor.

## Postprocessing And Selection

Stage 2 prediction uses threshold search over:

- `tau_empty`
- `tau_start`
- `tau_keep`
- `tau_video`
- `min_consecutive_steps`

The search now supports recall-constrained model selection.

- if `selection_min_recall` is set, the trainer prefers the best threshold set that meets the recall floor
- if no threshold set meets the floor, selection falls back to the unconstrained best candidate
- early stopping is now deferred until the recall floor has been met at least once, so runs are not terminated before they enter a recall-feasible regime

The threshold grids now include higher-precision operating points:

- `tau_start` includes `0.8` and `0.9`
- `tau_video` includes `0.8` and `0.9`
- `min_consecutive_steps` includes `4`

## Cache Management

Feature extraction is schema-versioned.

- stale bundles are rejected by the cached dataset loader
- `--force` rebuilds all bundles
- extraction logs:
  - extracted / skipped counts
  - positive / negative video counts
  - temporal-bin histogram

When any cache field changes, rebuild the cache before training.

## Experiment Guidance

Recommended search order:

1. RGB-only cached conv baseline under recall floor.
2. RGB+motion cached conv under the same seed and recall floor.
3. Repeat with at least two seeds before claiming a gain.
4. Only then promote the winner to a raw Stage 2 confirmation run.

Recommended comparison axis now:

- `--use-motion-branch` off vs on
- same seed
- same `lambda_video`
- same `monotonic-loss-weight`
- same recall floor

## Practical Commands

Rebuild cache after schema or feature changes:

```bash
uv run aid-extract-features --encoder-checkpoint artifacts/checkpoints/stage1_best.pt --split all --chunk-steps 64 --clip-batch-size 32 --force
```

Cached RGB-only baseline:

```bash
uv run aid-stage2-cached --run-name cached_conv_rgb_s1337 --cache-root artifacts/features/stage2_cache/stage1_best --temporal-model conv --target-mode cumulative --lambda-video 0.5 --monotonic-loss-weight 0.05 --batch-size 8 --max-steps 12 --window-stride 6 --num-workers 2 --num-epochs 20 --validate-every 1 --early-stopping-patience 4 --min-recall-for-selection 0.90 --seed 1337 --output-name cached_conv_rgb_s1337.pt
```

Cached RGB+motion comparison:

```bash
uv run aid-stage2-cached --run-name cached_conv_rgb_motion_s1337 --cache-root artifacts/features/stage2_cache/stage1_best --temporal-model conv --target-mode cumulative --use-motion-branch --lambda-video 0.5 --monotonic-loss-weight 0.05 --batch-size 8 --max-steps 12 --window-stride 6 --num-workers 2 --num-epochs 20 --validate-every 1 --early-stopping-patience 4 --min-recall-for-selection 0.90 --seed 1337 --output-name cached_conv_rgb_motion_s1337.pt
```

Raw confirmation run for the winning setup:

```bash
uv run aid-stage2 --run-name raw_confirm_motion --init-checkpoint artifacts/checkpoints/stage1_best.pt --temporal-model conv --target-mode cumulative --use-motion-branch --lambda-video 0.5 --monotonic-loss-weight 0.05 --batch-size 1 --max-steps 12 --window-stride 6 --num-workers 4 --num-epochs 20 --validate-every 1 --early-stopping-patience 4 --min-recall-for-selection 0.90 --output-name raw_confirm_motion.pt
```
