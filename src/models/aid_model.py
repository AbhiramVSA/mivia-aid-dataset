from __future__ import annotations

import torch
from torch import nn

from src.models.motion_fusion import MotionFeatureFusion
from src.models.temporal_head import TemporalConvHead, TemporalTransformerHead
from src.models.videomae_encoder import VideoMAEClipEncoder


class AIDTemporalModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
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
        self.encoder = VideoMAEClipEncoder(backbone_name=backbone_name)
        self.temporal_model = temporal_model
        self.use_motion_branch = use_motion_branch
        if use_motion_branch:
            self.motion_fusion = MotionFeatureFusion(
                hidden_size=hidden_size,
                motion_feature_dim=motion_feature_dim,
                dropout=dropout,
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
        clips: torch.Tensor,
        step_mask: torch.Tensor | None = None,
        motion_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
        clips: [B, T, C, F, H, W]
        """

        batch_size, num_steps = clips.shape[:2]
        flattened = clips.reshape(batch_size * num_steps, *clips.shape[2:])
        features = self.encoder(flattened).reshape(batch_size, num_steps, -1)
        if self.motion_fusion is not None:
            if motion_features is None:
                raise ValueError("motion_features are required when use_motion_branch=True")
            features = self.motion_fusion(features, motion_features)
        return self.temporal_head(features, step_mask=step_mask)
