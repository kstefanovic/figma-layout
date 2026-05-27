"""Blending utilities for RALF model + retrieval predictions."""

from __future__ import annotations

from typing import Any


DEFAULT_MODEL_WEIGHT = 0.65
DEFAULT_RETRIEVAL_WEIGHT = 0.35
RETRIEVAL_WEIGHT_BY_ROLE = {
    "background_shape_cluster": 0.60,
    "background_gradient_1": 0.50,
    "background_gradient_2": 0.50,
    "background_gradient_3": 0.50,
    "decoration_group": 0.50,
    "hero_group": 0.35,
    "text_main_group": 0.35,
    "brand_group": 0.25,
    "legal_group": 0.25,
    "badge_group": 0.30,
    "unknown_group": 0.10,
}


def _blend_vec(a: list[float], b: list[float], model_w: float, retrieval_w: float) -> list[float]:
    return [(model_w * float(a[i])) + (retrieval_w * float(b[i])) for i in range(4)]


def blend_model_and_retrieval_predictions(
    model_predictions: dict,
    retrieval_priors: dict,
    tokens: list[dict[str, Any]],
) -> dict:
    out: dict[str, list[float]] = {}
    token_by_id = {str(t.get("token_id") or ""): t for t in tokens}
    all_ids = set(model_predictions) | set(retrieval_priors)
    for token_id in all_ids:
        model_vec = model_predictions.get(token_id)
        retrieval_info = retrieval_priors.get(token_id) or {}
        retrieval_vec = retrieval_info.get("center_size_norm")
        train_role = str((token_by_id.get(token_id) or {}).get("train_role") or "unknown_group")
        retrieval_w = float(RETRIEVAL_WEIGHT_BY_ROLE.get(train_role, DEFAULT_RETRIEVAL_WEIGHT))
        model_w = 1.0 - retrieval_w
        if model_vec is not None and retrieval_vec is not None:
            out[token_id] = _blend_vec(
                [float(x) for x in model_vec[:4]],
                [float(x) for x in retrieval_vec[:4]],
                model_w,
                retrieval_w,
            )
        elif model_vec is not None:
            out[token_id] = [float(x) for x in model_vec[:4]]
        elif retrieval_vec is not None:
            out[token_id] = [float(x) for x in retrieval_vec[:4]]
    return out

