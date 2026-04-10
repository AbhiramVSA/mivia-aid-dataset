from __future__ import annotations

import torch
from torch import nn


class MotionFeatureFusion(nn.Module):
    def __init__(self, hidden_size: int, motion_feature_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        bottleneck = max(64, hidden_size // 4)
        self.motion_encoder = nn.Sequential(
            nn.Linear(motion_feature_dim, bottleneck),
            nn.GELU(),
            nn.LayerNorm(bottleneck),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_size),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, features: torch.Tensor, motion_features: torch.Tensor) -> torch.Tensor:
        motion_embed = self.motion_encoder(motion_features)
        gate = self.gate(torch.cat([features, motion_embed], dim=-1))
        fused = features + gate * motion_embed
        return self.output_norm(fused)
