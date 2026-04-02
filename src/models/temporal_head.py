from __future__ import annotations

import torch
from torch import nn


class TemporalConvHead(nn.Module):
    def __init__(self, hidden_size: int = 768, channels: tuple[int, int, int] = (512, 512, 256), dropout: float = 0.1) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c1, c2, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c2, c3, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.step_classifier = nn.Linear(c3, 1)
        self.video_classifier = nn.Linear(c3, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
        features: [B, T, D]
        Returns:
            step_logits: [B, T]
            video_logits: [B]
        """

        x = features.transpose(1, 2)
        x = self.net(x).transpose(1, 2)
        step_logits = self.step_classifier(x).squeeze(-1)
        video_logits = self.video_classifier(x.mean(dim=1)).squeeze(-1)
        return step_logits, video_logits
