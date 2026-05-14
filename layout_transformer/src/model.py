"""PyTorch model for semantic layout transformation."""

from __future__ import annotations

import torch
from torch import nn

from .roles import NUM_ROLES


class LayoutTransformer(nn.Module):
    def __init__(
        self,
        num_roles: int = NUM_ROLES,
        role_emb_dim: int = 64,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_roles = num_roles
        self.role_embedding = nn.Embedding(num_roles, role_emb_dim)
        self.numeric_mlp = nn.Sequential(
            nn.Linear(10, 128),
            nn.GELU(),
            nn.Linear(128, 160),
            nn.GELU(),
        )
        self.input_proj = nn.Linear(role_emb_dim + 160, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 4),
        )

    def forward(
        self,
        role_ids: torch.Tensor,
        source_bboxes: torch.Tensor,
        source_size: torch.Tensor,
        target_size: torch.Tensor,
        role_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_roles, _ = source_bboxes.shape
        if role_ids.dim() == 1:
            role_ids = role_ids.unsqueeze(0).expand(batch_size, -1)

        source_size = source_size.clamp_min(1.0)
        target_size = target_size.clamp_min(1.0)
        source_aspect = (source_size[:, 0] / source_size[:, 1]).unsqueeze(1).expand(-1, num_roles)
        target_aspect = (target_size[:, 0] / target_size[:, 1]).unsqueeze(1).expand(-1, num_roles)
        log_sizes = torch.log(torch.cat([source_size, target_size], dim=1)).unsqueeze(1).expand(-1, num_roles, -1)
        numeric = torch.cat(
            [
                source_bboxes,
                source_aspect.unsqueeze(-1),
                target_aspect.unsqueeze(-1),
                log_sizes,
            ],
            dim=-1,
        )

        role_emb = self.role_embedding(role_ids)
        numeric_emb = self.numeric_mlp(numeric)
        tokens = self.input_proj(torch.cat([role_emb, numeric_emb], dim=-1))

        padding_mask = None if role_mask is None else role_mask <= 0
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.bbox_head(encoded)
