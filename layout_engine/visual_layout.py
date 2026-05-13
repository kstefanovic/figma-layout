from __future__ import annotations


def _norm_to_abs(prior: dict, target_width: int, target_height: int) -> dict:
    return {
        "x": round(float(prior.get("x", 0)) * target_width, 2),
        "y": round(float(prior.get("y", 0)) * target_height, 2),
        "width": round(float(prior.get("w", prior.get("width", 0))) * target_width, 2),
        "height": round(float(prior.get("h", prior.get("height", 0))) * target_height, 2),
    }


def compute_visual_boxes(
    role_nodes: dict,
    solved_boxes: dict,
    orientation: str,
    target_width: int,
    target_height: int,
    retrieval_priors: dict | None = None,
    visual_priors: dict | None = None,
) -> dict:
    """Return boxes for visual roles, preferring retrieval priors over existing/fallback boxes."""
    out = dict(solved_boxes or {})
    retrieval = (retrieval_priors or {}).get("priors") or retrieval_priors or {}
    for role in ("hero_image", "background_shape"):
        prior = retrieval.get(role)
        if prior:
            out[role] = _norm_to_abs(prior, target_width, target_height)
            continue
        prior = (visual_priors or {}).get(role)
        if prior:
            out[role] = _norm_to_abs(prior, target_width, target_height)
    return out

