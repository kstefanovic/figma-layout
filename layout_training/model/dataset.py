"""PyTorch dataset and collation for variable-token layout pairs."""

from __future__ import annotations

import math
from typing import Any

from layout_training.pairs import read_jsonl


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for layout_training.model. Install torch to train or predict.") from exc
    return torch


def _orientation_code(value: str) -> float:
    return {"landscape": 0.0, "square": 0.5, "portrait": 1.0}.get(str(value), 0.5)


def build_role_vocab(pairs: list[dict[str, Any]]) -> dict[str, int]:
    roles = sorted({str(tok.get("train_role") or "unknown_group") for pair in pairs for tok in pair.get("tokens") or []})
    return {"<pad>": 0, **{role: i + 1 for i, role in enumerate(roles)}}


def build_type_vocab(pairs: list[dict[str, Any]]) -> dict[str, int]:
    types = sorted({str((tok.get("source") or {}).get("type") or "unknown") for pair in pairs for tok in pair.get("tokens") or []})
    return {"<pad>": 0, **{typ: i + 1 for i, typ in enumerate(types)}}


def token_numeric_features(pair: dict[str, Any], token: dict[str, Any]) -> list[float]:
    source = token.get("source") or {}
    source_canvas = pair.get("source_canvas") or {}
    target_canvas = pair.get("target_canvas") or {}
    source_w = float(source_canvas.get("width") or 1.0)
    source_h = float(source_canvas.get("height") or 1.0)
    target_w = float(target_canvas.get("width") or 1.0)
    target_h = float(target_canvas.get("height") or 1.0)
    center_size = [float(x) for x in (source.get("center_size_norm") or [0.5, 0.5, 0.1, 0.1])[:4]]
    descendant_count = float(source.get("descendant_count") or 0.0)
    return [
        *center_size,
        float(target_canvas.get("aspect") or target_w / max(target_h, 1e-6)),
        _orientation_code(str(target_canvas.get("orientation") or "")),
        target_w / max(source_w, 1e-6),
        target_h / max(source_h, 1e-6),
        float(source_canvas.get("aspect") or source_w / max(source_h, 1e-6)),
        float(source.get("area_ratio") or 0.0),
        1.0 if source.get("has_text") else 0.0,
        1.0 if source.get("has_image") else 0.0,
        1.0 if source.get("has_gradient") else 0.0,
        1.0 if source.get("has_star") else 0.0,
        1.0 if source.get("discount_text") else 0.0,
        1.0 if source.get("is_rotated") else 0.0,
        float(source.get("rotation_deg") or 0.0) / 180.0,
        min(5.0, float(source.get("instance_count") or 1.0)) / 5.0,
        min(5.0, math.log1p(descendant_count)) / 5.0,
    ]


class LayoutPairDataset:
    """Small map-style dataset. It avoids importing torch until data is read."""

    def __init__(self, pairs_path: str, role_vocab: dict[str, int] | None = None, type_vocab: dict[str, int] | None = None):
        self.pairs = read_jsonl(pairs_path)
        self.role_vocab = role_vocab or build_role_vocab(self.pairs)
        self.type_vocab = type_vocab or build_type_vocab(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        rows = []
        for token in pair.get("tokens") or []:
            source = token.get("source") or {}
            rows.append(
                {
                    "x_num": token_numeric_features(pair, token),
                    "source_center": [float(x) for x in (source.get("center_size_norm") or [0.5, 0.5, 0.1, 0.1])[:4]],
                    "role_id": self.role_vocab.get(str(token.get("train_role") or "unknown_group"), 0),
                    "type_id": self.type_vocab.get(str(source.get("type") or "unknown"), 0),
                    "target": [float(x) for x in (token.get("target_center_size_norm") or [0.0, 0.0, 0.0, 0.0])[:4]],
                    "has_target": bool(token.get("has_target")),
                    "train_role": str(token.get("train_role") or "unknown_group"),
                }
            )
        return {"pair_id": pair.get("pair_id"), "tokens": rows}


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch = _torch()
    max_t = max((len(item["tokens"]) for item in batch), default=0)
    feat_dim = len(batch[0]["tokens"][0]["x_num"]) if batch and batch[0]["tokens"] else 0
    bsz = len(batch)
    x_num = torch.zeros((bsz, max_t, feat_dim), dtype=torch.float32)
    source_center = torch.zeros((bsz, max_t, 4), dtype=torch.float32)
    role_ids = torch.zeros((bsz, max_t), dtype=torch.long)
    type_ids = torch.zeros((bsz, max_t), dtype=torch.long)
    mask = torch.zeros((bsz, max_t), dtype=torch.bool)
    target = torch.zeros((bsz, max_t, 4), dtype=torch.float32)
    target_mask = torch.zeros((bsz, max_t), dtype=torch.bool)
    train_roles: list[list[str]] = []
    for bi, item in enumerate(batch):
        role_row = []
        for ti, tok in enumerate(item["tokens"]):
            x_num[bi, ti] = torch.tensor(tok["x_num"], dtype=torch.float32)
            source_center[bi, ti] = torch.tensor(tok["source_center"], dtype=torch.float32)
            role_ids[bi, ti] = int(tok["role_id"])
            type_ids[bi, ti] = int(tok["type_id"])
            mask[bi, ti] = True
            target[bi, ti] = torch.tensor(tok["target"], dtype=torch.float32)
            target_mask[bi, ti] = bool(tok["has_target"])
            role_row.append(tok["train_role"])
        train_roles.append(role_row)
    return {
        "x_num": x_num,
        "source_center": source_center,
        "role_ids": role_ids,
        "type_ids": type_ids,
        "mask": mask,
        "target": target,
        "target_mask": target_mask,
        "train_roles": train_roles,
    }

