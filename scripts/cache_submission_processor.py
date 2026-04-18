from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_BACKBONE = "MCG-NJU/videomae-base"
DEFAULT_OUTPUT_DIR = Path("submission/processor/videomae-base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache the VideoMAE processor for offline Colab submission.")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoImageProcessor  # type: ignore
    except ImportError as exc:
        raise ImportError("transformers must be installed before caching the processor.") from exc

    processor = AutoImageProcessor.from_pretrained(args.backbone)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(args.output_dir)
    print(f"Cached processor to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
