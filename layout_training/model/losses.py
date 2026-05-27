"""Loss functions for top-level layout adaptation."""

from __future__ import annotations

from typing import Any


try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise RuntimeError("PyTorch is required for layout_training.model.losses.") from exc


ROLE_WEIGHTS = {
    "hero_group": 1.5,
    "text_main_group": 1.5,
    "brand_group": 1.2,
    "badge_group": 1.0,
    "legal_group": 1.0,
    "decoration_group": 0.6,
    "background_gradient_1": 0.5,
    "background_gradient_2": 0.5,
    "background_gradient_3": 0.5,
    "background_shape_cluster": 0.2,
    "foreground_group": 0.4,
    "unknown_group": 0.1,
}
SOFT_OUTSIDE_ROLES = {"background_shape_cluster", "hero_group", "decoration_group"}


def masked_l1_loss(pred, target, target_mask):
    diff = torch.abs(pred - target).mean(dim=-1)
    masked = diff * target_mask.float()
    return masked.sum() / target_mask.float().sum().clamp_min(1.0)


def _role_weight_tensor(train_roles: list[list[str]], shape: tuple[int, int], device: Any):
    weights = torch.ones(shape, dtype=torch.float32, device=device)
    for bi, row in enumerate(train_roles):
        for ti, role in enumerate(row):
            weights[bi, ti] = ROLE_WEIGHTS.get(role, 1.0)
    return weights


def role_weighted_loss(pred, target, target_mask, train_roles: list[list[str]]):
    diff = torch.abs(pred - target).mean(dim=-1)
    weights = _role_weight_tensor(train_roles, diff.shape, diff.device)
    masked = diff * target_mask.float() * weights
    denom = (target_mask.float() * weights).sum().clamp_min(1.0)
    return masked.sum() / denom


def size_positive_penalty(pred, target_mask):
    penalty = torch.relu(0.005 - pred[..., 2:4]).mean(dim=-1)
    return (penalty * target_mask.float()).sum() / target_mask.float().sum().clamp_min(1.0)


def outside_canvas_soft_penalty(pred, target_mask, train_roles: list[list[str]]):
    cx, cy, w, h = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    penalty = torch.relu(-x2) + torch.relu(-y2) + torch.relu(x1 - 1.0) + torch.relu(y1 - 1.0)
    role_factor = torch.ones_like(penalty)
    for bi, row in enumerate(train_roles):
        for ti, role in enumerate(row):
            if role in SOFT_OUTSIDE_ROLES:
                role_factor[bi, ti] = 0.25
    masked = penalty * role_factor * target_mask.float()
    return masked.sum() / target_mask.float().sum().clamp_min(1.0)


def total_layout_loss(pred, target, target_mask, train_roles: list[list[str]]):
    return (
        role_weighted_loss(pred, target, target_mask, train_roles)
        + 0.1 * size_positive_penalty(pred, target_mask)
        + 0.05 * outside_canvas_soft_penalty(pred, target_mask, train_roles)
    )

