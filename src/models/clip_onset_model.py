from __future__ import annotations

import torch
from torch import nn

from src.models.videomae_encoder import VideoMAEClipEncoder


class ClipOnsetModel(nn.Module):
    """Stage-1 model that classifies each clip independently."""

    def __init__(self, backbone_name: str) -> None:
        super().__init__()
        self.encoder = VideoMAEClipEncoder(backbone_name=backbone_name)
        self.classifier: nn.Linear | None = None
        # Build eagerly so the classifier is present before `.to(device)` and
        # checkpoint loading.
        self.build()

    def build(self) -> None:
        if self.classifier is not None:
            return
        hidden_size = self.encoder.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        """Args:
        clip_tensor: [B, T_steps, C, F, H, W]
        Returns:
            logits: [B, T_steps]
        """

        if self.classifier is None:
            self.build()
        assert self.classifier is not None
        batch_size, num_steps = clip_tensor.shape[:2]
        flattened = clip_tensor.reshape(batch_size * num_steps, *clip_tensor.shape[2:])
        embeddings = self.encoder(flattened)
        logits = self.classifier(embeddings).reshape(batch_size, num_steps)
        return logits
