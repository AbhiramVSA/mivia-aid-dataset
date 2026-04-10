from __future__ import annotations

import torch
from torch import nn

from src.models.temporal_head import TemporalConvHead, TemporalTransformerHead


class TemporalOnlyModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        temporal_model: str = "transformer",
        temporal_channels: tuple[int, int, int] = (512, 512, 256),
        dropout: float = 0.1,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_ffn_dim: int = 2048,
        use_motion_branch: bool = False,
        motion_feature_dim: int = 24,
    ) -> None:
        super().__init__()
        self.use_motion_branch = use_motion_branch
        self.motion_feature_dim = motion_feature_dim
        if use_motion_branch:
            self.motion_fusion = nn.Sequential(
                nn.Linear(hidden_size + motion_feature_dim, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
        else:
            self.motion_fusion = None
        if temporal_model == "conv":
            self.temporal_head = TemporalConvHead(
                hidden_size=hidden_size,
                channels=temporal_channels,
                dropout=dropout,
            )
        elif temporal_model == "transformer":
            self.temporal_head = TemporalTransformerHead(
                hidden_size=hidden_size,
                dropout=dropout,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                ffn_dim=transformer_ffn_dim,
            )
        else:
            raise ValueError(f"Unsupported temporal_model: {temporal_model}")

    def forward(
        self,
        features: torch.Tensor,
        step_mask: torch.Tensor | None = None,
        motion_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.motion_fusion is not None:
            if motion_features is None:
                raise ValueError("motion_features are required when use_motion_branch=True")
            features = self.motion_fusion(torch.cat([features, motion_features], dim=-1))
        return self.temporal_head(features, step_mask=step_mask)
