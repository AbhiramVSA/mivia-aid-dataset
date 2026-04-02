from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def checkpoint_payload(
    *,
    model_state_dict: dict[str, Any],
    config: Any,
    epoch: int,
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model_state_dict": model_state_dict,
        "config": asdict(config) if is_dataclass(config) else config,
        "epoch": epoch,
        "metrics": metrics,
        "extra": extra or {},
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)
