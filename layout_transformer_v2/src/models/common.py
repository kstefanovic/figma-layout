"""Shared encoder blocks for the V2 layout models."""

from __future__ import annotations

import torch
from torch import nn

from ..schema import ALIGN_H, ALIGN_V, ORIENTATIONS


class LayoutTokenEncoder(nn.Module):
    """Encode one semantic role sample with canvas, geometry, style, and orientation features."""

    def __init__(
        self,
        *,
        num_roles: int,
        num_node_types: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.role_embedding = nn.Embedding(num_roles, 64)
        self.type_embedding = nn.Embedding(num_node_types, 32)
        self.source_orientation_embedding = nn.Embedding(len(ORIENTATIONS), 16)
        self.target_orientation_embedding = nn.Embedding(len(ORIENTATIONS), 16)
        self.align_h_embedding = nn.Embedding(len(ALIGN_H), 16)
        self.align_v_embedding = nn.Embedding(len(ALIGN_V), 16)
        self.numeric_mlp = nn.Sequential(
            nn.Linear(3 + 3 + 4 + 4 + 5 + 1, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, 160),
            nn.GELU(),
        )
        input_dim = 64 + 32 + 16 + 16 + 16 + 16 + 160
        self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        *,
        role_id: torch.Tensor,
        node_type_id: torch.Tensor,
        source_canvas: torch.Tensor,
        target_canvas: torch.Tensor,
        source_orientation: torch.Tensor,
        target_orientation: torch.Tensor,
        source_bbox: torch.Tensor,
        source_relative_bbox: torch.Tensor,
        flags: torch.Tensor,
        font_size: torch.Tensor,
        align_h: torch.Tensor,
        align_v: torch.Tensor,
    ) -> torch.Tensor:
        numeric = torch.cat(
            [
                _canvas_features(source_canvas),
                _canvas_features(target_canvas),
                source_bbox,
                source_relative_bbox,
                flags,
                font_size.unsqueeze(-1),
            ],
            dim=-1,
        )
        token = torch.cat(
            [
                self.role_embedding(role_id),
                self.type_embedding(node_type_id),
                self.source_orientation_embedding(source_orientation),
                self.target_orientation_embedding(target_orientation),
                self.align_h_embedding(align_h),
                self.align_v_embedding(align_v),
                self.numeric_mlp(numeric),
            ],
            dim=-1,
        ).unsqueeze(1)
        return self.encoder(self.input_proj(token)).squeeze(1)


def make_head(d_model: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model // 2, out_dim),
    )


def _canvas_features(canvas: torch.Tensor) -> torch.Tensor:
    size = canvas[:, :2].clamp_min(1e-6)
    aspect = canvas[:, 2:3].clamp_min(1e-6)
    return torch.cat([torch.log(size), aspect], dim=-1)

