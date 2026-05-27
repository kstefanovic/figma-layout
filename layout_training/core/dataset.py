"""Dataset and feature extraction for the simplified CORE layout model."""

from __future__ import annotations

from typing import Any

from layout_training.pairs import read_jsonl
from layout_training.roles import CORE_TRAIN_ROLES


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for layout_training.core.") from exc
    return torch


def build_role_vocab() -> dict[str, int]:
    return {"<pad>": 0, **{role: idx + 1 for idx, role in enumerate(CORE_TRAIN_ROLES)}}


def build_type_vocab(pairs: list[dict[str, Any]]) -> dict[str, int]:
    token_types = sorted({str((token.get("source") or {}).get("token_type") or "unknown") for pair in pairs for token in pair.get("tokens") or []})
    return {"<pad>": 0, **{token_type: idx + 1 for idx, token_type in enumerate(token_types)}}


def token_numeric_features(pair: dict[str, Any], token: dict[str, Any]) -> list[float]:
    source = token.get("source") or {}
    source_canvas = pair.get("source_canvas") or {}
    target_canvas = pair.get("target_canvas") or {}
    source_w = float(source_canvas.get("width") or 1.0)
    source_h = float(source_canvas.get("height") or 1.0)
    target_w = float(target_canvas.get("width") or 1.0)
    target_h = float(target_canvas.get("height") or 1.0)
    center_size = [float(x) for x in (source.get("center_size_norm") or [0.5, 0.5, 0.1, 0.1])[:4]]
    return [
        *center_size,
        float(source.get("area_ratio") or 0.0),
        1.0 if source.get("has_text") else 0.0,
        1.0 if source.get("has_image") else 0.0,
        1.0 if source.get("is_rotated") else 0.0,
        float(source.get("rotation_deg") or 0.0) / 180.0,
        min(1.0, float(source.get("instance_count") or 1.0) / 5.0),
        float(source_canvas.get("aspect") or 1.0),
        float(target_canvas.get("aspect") or 1.0),
        target_w / max(source_w, 1e-6),
        target_h / max(source_h, 1e-6),
        float(source.get("coverage_ratio") or 0.0),
        1.0 if source.get("bleed_left") else 0.0,
        1.0 if source.get("bleed_right") else 0.0,
        1.0 if source.get("bleed_top") else 0.0,
        1.0 if source.get("bleed_bottom") else 0.0,
    ]


class CoreLayoutPairDataset:
    def __init__(self, pairs_path: str, role_vocab: dict[str, int] | None = None, type_vocab: dict[str, int] | None = None):
        self.pairs = read_jsonl(pairs_path)
        self.role_vocab = role_vocab or build_role_vocab()
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
                    "role_id": self.role_vocab.get(str(token.get("train_role") or ""), 0),
                    "type_id": self.type_vocab.get(str(source.get("token_type") or "unknown"), 0),
                    "target": [float(x) for x in (token.get("target_center_size_norm") or [0.0, 0.0, 0.0, 0.0])[:4]],
                    "target_bottom_y": float(token.get("target_bottom_y_norm") or 0.0),
                    "has_target": bool(token.get("has_target")),
                    "train_role": str(token.get("train_role") or ""),
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
    target_bottom_y = torch.zeros((bsz, max_t), dtype=torch.float32)
    target_mask = torch.zeros((bsz, max_t), dtype=torch.bool)
    train_roles: list[list[str]] = []
    for bi, item in enumerate(batch):
        roles_row: list[str] = []
        for ti, token in enumerate(item["tokens"]):
            x_num[bi, ti] = torch.tensor(token["x_num"], dtype=torch.float32)
            source_center[bi, ti] = torch.tensor(token["source_center"], dtype=torch.float32)
            role_ids[bi, ti] = int(token["role_id"])
            type_ids[bi, ti] = int(token["type_id"])
            mask[bi, ti] = True
            target[bi, ti] = torch.tensor(token["target"], dtype=torch.float32)
            target_bottom_y[bi, ti] = float(token["target_bottom_y"])
            target_mask[bi, ti] = bool(token["has_target"])
            roles_row.append(token["train_role"])
        train_roles.append(roles_row)
    return {
        "x_num": x_num,
        "source_center": source_center,
        "role_ids": role_ids,
        "type_ids": type_ids,
        "mask": mask,
        "target": target,
        "target_bottom_y": target_bottom_y,
        "target_mask": target_mask,
        "train_roles": train_roles,
    }
