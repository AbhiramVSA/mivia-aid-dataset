from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class VideoAnnotation:
    video_id: str
    video_path: Path
    duration_s: int
    is_positive: bool
    start_s: int | None
    end_s: int | None
    split: str


def parse_mmss(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported time format: {value}")


def load_annotations(csv_path: Path, videos_dir: Path, split: str) -> list[VideoAnnotation]:
    rows: list[VideoAnnotation] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for raw_row in reader:
            row = {key.strip(): (value.strip() if value is not None else value) for key, value in raw_row.items()}
            video_id = row["Id Video"]
            start_s = parse_mmss(row.get("Start"))
            end_s = parse_mmss(row.get("End"))
            annotation = VideoAnnotation(
                video_id=video_id,
                video_path=videos_dir / f"{video_id}.mp4",
                duration_s=parse_mmss(row["Duration"]) or 0,
                is_positive=start_s is not None,
                start_s=start_s,
                end_s=end_s,
                split=split,
            )
            rows.append(annotation)
    return rows


def validate_annotations(records: Iterable[VideoAnnotation]) -> None:
    seen_ids: set[tuple[str, str]] = set()
    for record in records:
        dedupe_key = (record.split, record.video_id)
        if dedupe_key in seen_ids:
            raise ValueError(f"Duplicate annotation id in split {record.split}: {record.video_id}")
        seen_ids.add(dedupe_key)
        if not record.video_path.exists():
            raise FileNotFoundError(f"Missing video file for {record.video_id}: {record.video_path}")
        if record.is_positive:
            if record.start_s is None or record.end_s is None:
                raise ValueError(f"Positive sample missing boundaries: {record.video_id}")
            if record.start_s > record.end_s:
                raise ValueError(f"Invalid temporal bounds for {record.video_id}: {record.start_s}>{record.end_s}")
        else:
            if record.start_s is not None or record.end_s is not None:
                raise ValueError(f"Negative sample has temporal annotations: {record.video_id}")


def build_dataset_manifest(
    train_csv: Path,
    train_videos_dir: Path,
    val_csv: Path | None = None,
    val_videos_dir: Path | None = None,
) -> list[VideoAnnotation]:
    records = load_annotations(train_csv, train_videos_dir, split="train")
    if val_csv is not None and val_videos_dir is not None and val_videos_dir.exists():
        records.extend(load_annotations(val_csv, val_videos_dir, split="val"))
    validate_annotations(records)
    return records
