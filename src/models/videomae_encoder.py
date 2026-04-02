from __future__ import annotations

import torch
from torch import nn


class VideoMAEClipEncoder(nn.Module):
    """Wrapper around a pretrained VideoMAE backbone."""

    def __init__(self, backbone_name: str) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.encoder: nn.Module | None = None
        # Build eagerly so checkpoint loading can restore encoder weights before
        # the first forward pass.
        self.build()

    def build(self) -> None:
        """Instantiate the pretrained backbone lazily."""
        if self.encoder is not None:
            return
        from transformers import VideoMAEModel  # type: ignore

        self.encoder = VideoMAEModel.from_pretrained(self.backbone_name)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Encode clips into pooled embeddings.

        Args:
            clips: Tensor with shape [B, C, T, H, W]

        Returns:
            Tensor with shape [B, D]
        """
        if clips.ndim != 5:
            raise ValueError(f"Expected clips with shape [B, C, T, H, W], got {tuple(clips.shape)}")
        if self.encoder is None:
            self.build()
        assert self.encoder is not None
        pixel_values = clips.permute(0, 2, 1, 3, 4).contiguous()
        outputs = self.encoder(pixel_values=pixel_values)
        return outputs.last_hidden_state.mean(dim=1)

    @property
    def hidden_size(self) -> int:
        if self.encoder is None:
            self.build()
        assert self.encoder is not None
        return int(self.encoder.config.hidden_size)
