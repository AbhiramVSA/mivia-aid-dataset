from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dist" / "aid2026_submission"
DEFAULT_ARCHIVE_PATH = PROJECT_ROOT / "dist" / "aid2026_submission.zip"
DEFAULT_STAGE1 = PROJECT_ROOT / "submission" / "weights" / "stage1_best.pt"
DEFAULT_STAGE2 = PROJECT_ROOT / "submission" / "weights" / "cached_conv_rgb_r90_s2026.pt"

PACKAGE_FILES = (
    "requirements.txt",
    "pyproject.toml",
)

PACKAGE_DIRS = (
    "src",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the AID2026 Colab submission archive.")
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--stage2-checkpoint", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive-path", type=Path, default=DEFAULT_ARCHIVE_PATH)
    return parser.parse_args()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _write_submission_readme(destination: Path, stage1_name: str, stage2_name: str) -> None:
    readme = f"""AID2026 submission package

This archive contains the final offline incident-onset detector described in the paper.

Selected deployment recipe
- Stage 1 encoder checkpoint: {stage1_name}
- Stage 2 temporal checkpoint: {stage2_name}
- Model family: VideoMAE-base encoder + RGB-only dilated temporal convolution head
- Selection rule: best validation F1 subject to recall >= 0.90

Required entrypoints
- test.py
- test.ipynb

Expected command
python test.py --videos foo_videos/ --results foo_results/
"""
    destination.write_text(readme, encoding="utf-8")


def _zip_directory(source_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(source_dir))


def main() -> None:
    args = parse_args()

    missing = [path for path in (args.stage1_checkpoint, args.stage2_checkpoint) if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Submission export requires the selected checkpoints to exist before packaging:\n"
            f"{formatted}"
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for relative_path in PACKAGE_FILES:
        source = PROJECT_ROOT / relative_path
        _copy_file(source, output_dir / relative_path)

    for relative_path in PACKAGE_DIRS:
        source = PROJECT_ROOT / relative_path
        _copy_tree(source, output_dir / relative_path)

    _copy_file(PROJECT_ROOT / "submission" / "test.py", output_dir / "test.py")
    _copy_file(PROJECT_ROOT / "submission" / "test.ipynb", output_dir / "test.ipynb")
    _copy_file(args.stage1_checkpoint.resolve(), output_dir / "submission" / "weights" / "stage1_best.pt")
    _copy_file(
        args.stage2_checkpoint.resolve(),
        output_dir / "submission" / "weights" / "cached_conv_rgb_r90_s2026.pt",
    )
    _write_submission_readme(
        output_dir / "README_submission.txt",
        stage1_name=args.stage1_checkpoint.name,
        stage2_name=args.stage2_checkpoint.name,
    )
    _zip_directory(output_dir, args.archive_path.resolve())
    print(f"Packaged submission directory: {output_dir}")
    print(f"Packaged submission archive: {args.archive_path.resolve()}")


if __name__ == "__main__":
    main()
