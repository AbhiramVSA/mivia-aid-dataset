from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _make_checkpoint_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _make_checkpoint_safe(asdict(value))
    if isinstance(value, dict):
        return {key: _make_checkpoint_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_make_checkpoint_safe(item) for item in value]
        return tuple(converted) if isinstance(value, tuple) else converted
    return value


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
        "config": _make_checkpoint_safe(config),
        "epoch": epoch,
        "metrics": _make_checkpoint_safe(metrics),
        "extra": _make_checkpoint_safe(extra or {}),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
