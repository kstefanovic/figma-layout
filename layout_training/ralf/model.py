"""RALF top-level layout transformer."""

from __future__ import annotations


def _torch_modules():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for RALF model.") from exc
    return torch, nn, F


torch, nn, F = _torch_modules()


class RalfTopLevelLayoutTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_roles: int,
        num_types: int,
        src_numeric_dim: int = 16,
        ret_numeric_dim: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        source_layers: int = 3,
        retrieved_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        emb_d = d_model // 4
        num_d = d_model // 2
        self.role_embedding = nn.Embedding(num_roles, emb_d, padding_idx=0)
        self.type_embedding = nn.Embedding(num_types, emb_d, padding_idx=0)
        self.src_num_proj = nn.Sequential(nn.Linear(src_numeric_dim, num_d), nn.ReLU(), nn.Linear(num_d, num_d), nn.ReLU())
        self.ret_num_proj = nn.Sequential(nn.Linear(ret_numeric_dim, num_d), nn.ReLU(), nn.Linear(num_d, num_d), nn.ReLU())
        self.src_input = nn.Linear(d_model, d_model)
        self.ret_input = nn.Linear(d_model, d_model)

        src_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation="gelu"
        )
        ret_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.source_encoder = nn.TransformerEncoder(src_layer, num_layers=source_layers)
        self.retrieved_encoder = nn.TransformerEncoder(ret_layer, num_layers=retrieved_layers)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )

    def _encode_source(self, src_num, src_role_ids, src_type_ids, src_mask):
        src = torch.cat([self.src_num_proj(src_num), self.role_embedding(src_role_ids), self.type_embedding(src_type_ids)], dim=-1)
        src = self.src_input(src)
        return self.source_encoder(src, src_key_padding_mask=~src_mask)

    def _encode_retrieval(self, ret_num, ret_role_ids, ret_type_ids, ret_mask, ret_scores):
        b, k, r, _ = ret_num.shape
        if k == 0 or r == 0:
            return ret_num.new_zeros((b, 0, ret_num.shape[-1])), ret_mask.new_zeros((b, 0))
        flat_num = ret_num.view(b * k, r, -1)
        flat_role = ret_role_ids.view(b * k, r)
        flat_type = ret_type_ids.view(b * k, r)
        flat_mask = ret_mask.view(b * k, r)
        x = torch.cat([self.ret_num_proj(flat_num), self.role_embedding(flat_role), self.type_embedding(flat_type)], dim=-1)
        x = self.ret_input(x)
        enc = self.retrieved_encoder(x, src_key_padding_mask=~flat_mask)  # [b*k, r, d]
        enc = enc.view(b, k, r, -1)
        score = ret_scores[:, :, None, None]
        enc = enc + score
        flat_enc = enc.view(b, k * r, -1)
        flat_m = ret_mask.view(b, k * r)
        return flat_enc, flat_m

    def forward(
        self,
        src_num,
        src_role_ids,
        src_type_ids,
        src_mask,
        ret_num,
        ret_role_ids,
        ret_type_ids,
        ret_mask,
        ret_scores,
        source_center_size_norm,
    ):
        src_enc = self._encode_source(src_num, src_role_ids, src_type_ids, src_mask)
        ret_enc, ret_flat_mask = self._encode_retrieval(ret_num, ret_role_ids, ret_type_ids, ret_mask, ret_scores)
        if ret_enc.shape[1] == 0:
            ctx = torch.zeros_like(src_enc)
        else:
            ctx, _ = self.cross_attn(
                query=src_enc,
                key=ret_enc,
                value=ret_enc,
                key_padding_mask=~ret_flat_mask,
                need_weights=False,
            )
        fused = torch.cat([src_enc, ctx], dim=-1)
        delta = self.pred_head(fused)
        pred = source_center_size_norm + delta
        wh = F.softplus(pred[..., 2:4]) + 1e-4
        return torch.cat([pred[..., 0:2], wh], dim=-1)
