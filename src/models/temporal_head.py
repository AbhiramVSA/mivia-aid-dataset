from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size)
        )
        pe = torch.zeros(max_len, hidden_size, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)


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

    def forward(self, features: torch.Tensor, step_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected features [B, T, D], got {tuple(features.shape)}")
        x = features.transpose(1, 2)
        x = self.net(x).transpose(1, 2)
        step_logits = self.step_classifier(x).squeeze(-1)
        if step_mask is None:
            pooled = x.mean(dim=1)
        else:
            valid = step_mask.unsqueeze(-1).to(dtype=x.dtype)
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        video_logits = self.video_classifier(pooled).squeeze(-1)
        return step_logits, video_logits


class TemporalTransformerHead(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.position_encoding = SinusoidalPositionEncoding(hidden_size=hidden_size, max_len=max_len)
        self.input_norm = nn.LayerNorm(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.step_classifier = nn.Linear(hidden_size, 1)
        self.video_classifier = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor, step_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected features [B, T, D], got {tuple(features.shape)}")
        x = self.input_norm(features)
        x = self.position_encoding(x)
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        padding_mask = None
        if step_mask is not None:
            padding_mask = ~step_mask.bool()
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.output_norm(x)
        step_logits = self.step_classifier(x).squeeze(-1)

        if step_mask is None:
            pooled = x.mean(dim=1)
        else:
            valid = step_mask.unsqueeze(-1).to(dtype=x.dtype)
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        video_logits = self.video_classifier(pooled).squeeze(-1)
        return step_logits, video_logits
