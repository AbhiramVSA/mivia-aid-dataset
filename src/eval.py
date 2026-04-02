from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ExperimentConfig
from src.data.annotations import load_annotations
from src.infer_video import infer_single_video
from src.utils.metrics import PredictionRecord, compute_contest_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the validation set")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig().resolved()
    if not config.paths.val_videos_dir.exists():
        raise FileNotFoundError(f"Validation videos directory not found: {config.paths.val_videos_dir}")

    annotations = load_annotations(config.paths.val_csv, config.paths.val_videos_dir, split="val")
    if args.limit is not None:
        annotations = annotations[: args.limit]

    records: list[PredictionRecord] = []
    for annotation in annotations:
        prediction = infer_single_video(annotation.video_path, checkpoint_path=args.checkpoint)
        records.append(
            PredictionRecord(
                video_id=annotation.video_id,
                is_positive=annotation.is_positive,
                ground_truth_start_s=float(annotation.start_s) if annotation.start_s is not None else None,
                predicted_start_s=prediction,
            )
        )
    metrics = compute_contest_metrics(records)
    print(
        " ".join(
            [
                f"precision={metrics.precision:.4f}",
                f"recall={metrics.recall:.4f}",
                f"f1={metrics.f1_score:.4f}",
                f"tp={metrics.true_positives}",
                f"fp={metrics.false_positives}",
                f"fn={metrics.false_negatives}",
            ]
        )
    )


if __name__ == "__main__":
    main()
