"""RALF losses for top-level layout prediction."""

from __future__ import annotations

from layout_training.model.losses import (
    outside_canvas_soft_penalty,
    role_weighted_loss,
    size_positive_penalty,
)


def total_ralf_loss(pred, target, target_mask, train_roles):
    return (
        role_weighted_loss(pred, target, target_mask, train_roles)
        + 0.1 * size_positive_penalty(pred, target_mask)
        + 0.05 * outside_canvas_soft_penalty(pred, target_mask, train_roles)
    )

