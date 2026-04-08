# AID2026 Model Review Context

## Dataset Setup

- Train annotations: `Train_GT.csv`
- Val annotations: `MIVIA-AID-Dataset/Val_GT.csv`
- Local split in use:
  - train: `1246` videos
  - val: `310` videos
- Labels are second-level onset annotations parsed from `mm:ss`

## Contest Metric

- A positive prediction is correct if `prediction in [g - 1s, g + 30s]`
- This means:
  - recall matters
  - false positives matter a lot
  - exact boundary localization is less important than stable post-onset detection

## Shared Video Front End

- Backbone: `MCG-NJU/videomae-base`
- Sampling:
  - `8 FPS`
  - `16` frames per clip
  - stride `4` sampled frames = `0.5s`
- Clip span = `2.0s`
- Current preprocessing:
  - decode RGB frames with OpenCV
  - Hugging Face `AutoImageProcessor` for VideoMAE

## Stage 1

- Model:
  - VideoMAE encoder
  - independent linear clip classifier
- Target:
  - cumulative step label: `1 if t >= onset else 0`
- Role:
  - warm-start / encoder adaptation
  - not intended as final submitted model

### Best Stage 1 Result

- `precision=0.4974`
- `recall=0.8362`
- `f1=0.6238`

## Previous Stage 2 Family

- Temporal head:
  - 3-layer dilated Conv1D
- Target:
  - cumulative step label
- Video head:
  - auxiliary BCE
- Inference:
  - thresholded postprocessing over step scores

### Best Observed Result Before Later Refactors

- `best_f1=0.6408`

## Refactor That Failed

- Replaced temporal conv with Transformer
- Replaced cumulative label with Gaussian onset target
- Switched to peak-style decoding

### Observed Outcome

- warm-started Transformer onset run:
  - `best_f1=0.5808`
- scratch Transformer onset run:
  - `best_f1=0.5132`

### Interpretation

- Removing Stage 1 is not supported by current evidence
- Pure onset-peak reformulation appears worse for the contest metric
- Cumulative detection appears better aligned with this benchmark

## Current Code Capabilities

Stage 2 now supports these ablations:

- temporal model:
  - `conv`
  - `transformer`
- target mode:
  - `cumulative`
  - `onset`

Inference/validation supports:

- threshold sweep over:
  - `tau_empty`
  - `tau_start`
  - `tau_video`
  - `min_consecutive_steps`
- cumulative decoding
- peak decoding

## Systems Finding From Logs

The current bottleneck is not GPU memory. It is the raw-video training pipeline:

- CPU decode
- preprocessing
- dataloader / disk contention

Observed logs showed:

- GPU memory ~ `1.4–1.6 GB`
- per-run dataloading dominates throughput
- multiple parallel raw-video runs cause severe slowdown

## New Fast Iteration Path

Added:

- `aid-extract-features`
- `aid-stage2-cached`

This enables:

1. Extract per-video clip embeddings once using the warm-started encoder
2. Train temporal heads from cached embeddings
3. Sweep architecture / target / lambda settings cheaply
4. Only fine-tune full raw-video model after selecting the best recipe

## Current Main Hypotheses

1. Best path likely keeps:
   - Stage 1 warm-start
   - cumulative target
2. The key open architecture question is:
   - `conv` vs `transformer` under cumulative mode
3. Slightly longer context (`max_steps=12`) may help more than `max_steps=8`
4. Raw-video training should not be the main search loop anymore; cached-feature training should be

## Most Important Next Experiments

- `conv + cumulative + warm + ms12`
- `transformer + cumulative + warm + ms12`
- cached-feature versions of the same
- lambda sweep on the better head family
