from __future__ import annotations

import torch
from torch import nn

from src.models.temporal_head import TemporalConvHead
from src.models.videomae_encoder import VideoMAEClipEncoder


class AIDTemporalModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        hidden_size: int = 768,
        temporal_channels: tuple[int, int, int] = (512, 512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = VideoMAEClipEncoder(backbone_name=backbone_name)
        self.temporal_head = TemporalConvHead(
            hidden_size=hidden_size,
            channels=temporal_channels,
            dropout=dropout,
        )

    def forward(self, clips: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
        clips: [B, T, C, F, H, W]
        """

        batch_size, num_steps = clips.shape[:2]
        flattened = clips.reshape(batch_size * num_steps, *clips.shape[2:])
        features = self.encoder(flattened).reshape(batch_size, num_steps, -1)
        return self.temporal_head(features)
