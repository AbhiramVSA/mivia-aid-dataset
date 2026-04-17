from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.infer_video import infer_single_video


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AID2026 submission entrypoint")
    parser.add_argument("--videos", type=Path, required=True, help="Directory containing test videos")
    parser.add_argument("--results", type=Path, required=True, help="Directory where results.csv will be written")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.videos.exists():
        raise FileNotFoundError(f"Video directory not found: {args.videos}")
    if not args.videos.is_dir():
        raise NotADirectoryError(f"--videos must point to a directory: {args.videos}")
    args.results.mkdir(parents=True, exist_ok=True)
    csv_path = args.results / "results.csv"
    video_paths = sorted(path for path in args.videos.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Id Video", "Start(Seconds)"])
        for video_path in video_paths:
            prediction = infer_single_video(video_path)
            value = "" if prediction is None else f"{prediction:.3f}".rstrip("0").rstrip(".")
            writer.writerow([video_path.name, value])


if __name__ == "__main__":
    main()
