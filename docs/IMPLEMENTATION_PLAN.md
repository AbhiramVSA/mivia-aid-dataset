# AID2026 Implementation Sheet

## Core Interfaces

### `src/data/annotations.py`

```python
def parse_mmss(value: str | None) -> int | None
def load_annotations(csv_path: Path, videos_dir: Path, split: str) -> list[VideoAnnotation]
def validate_annotations(records: Iterable[VideoAnnotation]) -> None
def build_dataset_manifest(...) -> list[VideoAnnotation]
```

### `src/data/video_decode.py`

```python
def decode_video_frames(video_path: Path, sample_fps: int) -> tuple[torch.Tensor, float]
def build_causal_clip_indices(num_frames: int, sampling: VideoSamplingConfig) -> list[tuple[int, int]]
```

Expected tensor shape:

- decoded frames: `[T, C, H, W]`
- model clips: `[B, T_steps, C, F, H, W]`

### `src/models/aid_model.py`

```python
class AIDTemporalModel(nn.Module):
    def forward(self, clips: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

Input and output:

- input: `[B, T_steps, C, F, H, W]`
- `step_logits`: `[B, T_steps]`
- `video_logits`: `[B]`

## Loss Definition

Use:

```python
loss = step_loss_mean + lambda_video * video_loss
```

Where:

- `step_loss_mean` averages only valid masked steps per video
- `video_loss` is BCE on the video-level label

Sweep:

- `lambda_video in {0.25, 0.5, 1.0}`

## Validation Selection

Choose checkpoints by the exact contest F1, not loss.

Validation decode:

1. sigmoid on `step_logits`
2. median filter kernel `3`
3. if `max(prob) < tau_empty`: predict no incident
4. else first timestamp with `prob >= tau_start`

Tune:

- `tau_empty in {0.2, 0.3, 0.4, 0.5, 0.6}`
- `tau_start in {0.3, 0.4, 0.5, 0.6, 0.7}`

## Submission Contract

`submission/test.py` must write:

- header: `Id Video, Start(Seconds)`
- first column: filename including `.mp4`
- second column:
  - empty string for no incident
  - decimal seconds otherwise

## Current Status

Implemented:

1. CSV parsing and dataset manifest.
2. Official contest metric computation.
3. Postprocessing.
4. OpenCV-based sampled-window decoding.
5. VideoMAE encoder wrapper.
6. Stage 1 training entrypoint.
7. Stage 2 training entrypoint.
8. Checkpoint-backed inference.
9. Validation runner.

## Remaining Practical Work

1. Extract `val.zip` into `val/`.
2. Install dependencies on the training machine.
3. Run a small smoke test with `--max-train-videos 8 --max-val-videos 8`.
4. Run full Stage 1.
5. Sweep Stage 2 `lambda_video`.
6. Add threshold sweep for `tau_empty` and `tau_start` if Stage 2 converges well.
