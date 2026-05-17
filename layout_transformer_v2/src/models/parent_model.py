"""Parent structural layout transformer."""

from __future__ import annotations

import torch
from torch import nn

from ..schema import PARENT_ROLES
from .common import LayoutTokenEncoder, make_head


class ParentLayoutTransformer(nn.Module):
    """Predict target canvas-normalized bboxes and visibility for parent roles."""

    def __init__(
        self,
        num_roles: int = len(PARENT_ROLES),
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
        self.bbox_head = make_head(d_model, 4, dropout)
        self.visibility_head = make_head(d_model, 1, dropout)

    def forward(self, **batch: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(**batch)
        return {
            "bbox": self.bbox_head(encoded),
            "visibility": self.visibility_head(encoded).squeeze(-1),
        }

