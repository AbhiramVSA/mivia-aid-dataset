## AID2026 Scaffold

This repository now contains the agreed architecture scaffold for the AID2026 contest submission:

- `VideoMAE-base` clip encoder
- causal dense clip sampling at `8 FPS`
- temporal convolution head with:
  - per-step onset logits
  - auxiliary video-level incident logit
- official contest metric reproduction
- submission entrypoint at `submission/test.py`

## Files

- `src/config.py`
  - experiment config dataclasses
- `src/data/annotations.py`
  - dataset manifest and CSV parsing
- `src/data/video_decode.py`
  - decoder contract and clip index construction
- `src/data/clip_dataset.py`
  - Stage 1 clip dataset contract
- `src/data/sequence_dataset.py`
  - Stage 2 sequence dataset contract
- `src/models/videomae_encoder.py`
  - pretrained VideoMAE wrapper
- `src/models/temporal_head.py`
  - temporal conv head
- `src/models/aid_model.py`
  - full model assembly
- `src/utils/metrics.py`
  - official TP/FP/FN, precision, recall, F1
- `src/utils/postprocess.py`
  - median filter and thresholded onset decoding
- `src/utils/checkpoint.py`
  - checkpoint payload helpers
- `src/train_stage1.py`
  - clip-level training entrypoint
- `src/train_stage2.py`
  - sequence fine-tuning entrypoint
- `src/eval.py`
  - validation runner
- `src/infer_video.py`
  - single-video inference contract
- `submission/test.py`
  - contest CSV writer

## Checkpoint Format

Use `src/utils/checkpoint.py` and keep this payload shape stable:

```python
{
    "model_state_dict": ...,
    "config": ...,
    "epoch": int,
    "metrics": {
        "f1_score": float,
        "precision": float,
        "recall": float,
    },
    "extra": {
        "tau_empty": float,
        "tau_start": float,
        "lambda_video": float,
    },
}
```

## Runtime Notes

- Extract `MIVIA-AID-Dataset/val.zip` into a repo-root folder named `val/`.
- Stage 1 and Stage 2 use windowed sequence samples to avoid loading very long videos into memory at once.
- The decoder backend is OpenCV. The model preprocessing path uses the Hugging Face VideoMAE image processor.

## Clone And Setup

On the A100 box:

```bash
git clone <your-repo-url>
cd mivia-aid-dataset
uv sync --frozen
```

If Python `3.11` is not already available, `uv` can manage it for you:

```bash
uv python install 3.11
uv sync --frozen
```

The project is pinned to Python `3.11` via [.python-version](C:/Users/abhir/iitm-research/mivia-aid-dataset/.python-version) and `requires-python` in [pyproject.toml](C:/Users/abhir/iitm-research/mivia-aid-dataset/pyproject.toml).

PyTorch is configured through `uv` package indexes so Linux resolves GPU wheels from the CUDA 12.8 PyTorch index, while non-Linux systems fall back to CPU wheels. This follows the official `uv` PyTorch integration guidance.

The repository is intended to track:

- source code
- `uv.lock`
- small annotation files such as [Train_GT.csv](C:/Users/abhir/iitm-research/mivia-aid-dataset/Train_GT.csv) and [Val_GT.csv](C:/Users/abhir/iitm-research/mivia-aid-dataset/MIVIA-AID-Dataset/Val_GT.csv)

It does **not** track the video datasets themselves. After cloning on the A100 machine, place:

- training videos under `train-002/train/`
- validation videos under `val/`

## Training Commands

```bash
uv run aid-stage1 --output-name stage1_best.pt
uv run aid-stage2 --init-checkpoint artifacts/checkpoints/stage1_best.pt --lambda-video 0.5 --output-name stage2_best.pt
uv run aid-eval --checkpoint artifacts/checkpoints/stage2_best.pt
```

Repeat Stage 2 with:

- `--lambda-video 0.25`
- `--lambda-video 0.5`
- `--lambda-video 1.0`

Then keep the checkpoint with the best validation F1.

## Smoke Test

Before a full run:

```bash
uv run aid-stage1 --max-train-videos 8 --max-val-videos 8
uv run aid-stage2 --init-checkpoint artifacts/checkpoints/stage1_best.pt --lambda-video 0.5 --max-train-videos 8 --max-val-videos 8
```

## Git Hygiene

The repo is configured to ignore:

- local virtual environments
- training artifacts
- submission weights
- extracted datasets and zip archives

That keeps `git status` limited to source code and docs, so the repository can be pushed and cloned cleanly.
