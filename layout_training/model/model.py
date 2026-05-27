"""Transformer model for variable-token top-level layout prediction."""

from __future__ import annotations


def _torch_modules():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for TopLevelLayoutTransformer. Install torch to train or predict.") from exc
    return torch, nn, F


torch, nn, F = _torch_modules()


class TopLevelLayoutTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_roles: int,
        num_types: int,
        numeric_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.role_embedding = nn.Embedding(num_roles, d_model // 4, padding_idx=0)
        self.type_embedding = nn.Embedding(num_types, d_model // 4, padding_idx=0)
        self.numeric_proj = nn.Sequential(
            nn.Linear(numeric_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.ReLU(),
        )
        self.input_proj = nn.Linear(d_model, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))

    def forward(self, x_num, role_ids, type_ids, mask, source_center):
        role = self.role_embedding(role_ids)
        typ = self.type_embedding(type_ids)
        num = self.numeric_proj(x_num)
        x = self.input_proj(torch.cat([num, role, typ], dim=-1))
        padding_mask = ~mask
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        delta = self.head(encoded)
        pred = source_center + delta
        wh = F.softplus(pred[..., 2:4]) + 1e-4
        return torch.cat([pred[..., 0:2], wh], dim=-1)

