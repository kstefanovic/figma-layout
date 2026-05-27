"""Role-specific losses for the simplified CORE layout model."""

from __future__ import annotations

from typing import Any


try:
    import torch
except ImportError as exc:
    raise RuntimeError("PyTorch is required for layout_training.core.losses.") from exc


ROLE_WEIGHTS = {
    "hero_group": 1.5,
    "brand_group": 1.2,
    "text_main_group": 1.5,
    "background_cluster": 0.15,
    "legal_group": 0.5,
}


def _role_weight_tensor(train_roles: list[list[str]], shape: tuple[int, int], device: Any):
    weights = torch.ones(shape, dtype=torch.float32, device=device)
    for bi, row in enumerate(train_roles):
        for ti, role in enumerate(row):
            weights[bi, ti] = ROLE_WEIGHTS.get(role, 1.0)
    return weights


def total_core_layout_loss(pred, target, target_bottom_y, target_mask, train_roles: list[list[str]]):
    weights = _role_weight_tensor(train_roles, target_mask.shape, pred.device)
    total = torch.zeros_like(weights)
    denom = torch.zeros_like(weights)

    for bi, row in enumerate(train_roles):
        for ti, role in enumerate(row):
            if not target_mask[bi, ti]:
                continue
            weight = weights[bi, ti]
            if role == "legal_group":
                pred_bottom_y = pred[bi, ti, 1] + pred[bi, ti, 3] / 2.0
                loss = torch.abs(pred[bi, ti, 0] - target[bi, ti, 0]) + torch.abs(pred_bottom_y - target_bottom_y[bi, ti])
            else:
                loss = torch.abs(pred[bi, ti] - target[bi, ti]).mean()
                loss = loss + 0.05 * torch.relu(0.005 - pred[bi, ti, 2:4]).mean()
            total[bi, ti] = loss * weight
            denom[bi, ti] = weight
    return total.sum() / denom.sum().clamp_min(1.0)
