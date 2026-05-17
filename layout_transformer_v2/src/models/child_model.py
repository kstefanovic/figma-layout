"""Child relative layout transformer."""

from __future__ import annotations

import torch
from torch import nn

from ..schema import ALIGN_H, ALIGN_V, CHILD_ROLES
from .common import LayoutTokenEncoder, make_head


class ChildLayoutTransformer(nn.Module):
    """Predict child relative boxes, text size, and alignment classes."""

    def __init__(
        self,
        num_roles: int = len(CHILD_ROLES),
        num_node_types: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = LayoutTokenEncoder(
            num_roles=num_roles,
            num_node_types=num_node_types,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.relative_bbox_head = make_head(d_model, 4, dropout)
        self.font_size_head = make_head(d_model, 1, dropout)
        self.align_h_head = make_head(d_model, len(ALIGN_H), dropout)
        self.align_v_head = make_head(d_model, len(ALIGN_V), dropout)

    def forward(self, **batch: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(**batch)
        return {
            "relative_bbox": self.relative_bbox_head(encoded),
            "font_size": self.font_size_head(encoded).squeeze(-1),
            "align_h": self.align_h_head(encoded),
            "align_v": self.align_v_head(encoded),
        }

