"""Prediction postprocessing and JSON application helpers."""

from __future__ import annotations

import copy
from typing import Any

from layout_training.config import MIN_SIZE_PX
from layout_training.geometry import bbox_xywh, bbox_union, denormalize_center_size


def clamp_prediction_norm(values: list[float]) -> list[float]:
    cx, cy, w, h = [float(x) for x in values[:4]]
    cx = min(2.5, max(-1.5, cx))
    cy = min(2.5, max(-1.5, cy))
    w = min(4.0, max(0.01, w))
    h = min(4.0, max(0.01, h))
    return [cx, cy, w, h]


def postprocess_token_prediction(train_role: str, pred_norm: list[float], target_w: float, target_h: float) -> list[float]:
    cx, cy, w, h = clamp_prediction_norm(pred_norm)
    if train_role == "background_shape_cluster":
        w = max(w, 1.0)
        h = max(h, 1.0)
    if train_role in {"brand_group", "badge_group"}:
        cx = min(1.2, max(-0.2, cx))
        cy = min(1.2, max(-0.2, cy))
    if train_role == "legal_group" and cy < 0.45:
        cy = 0.85
    if train_role == "hero_group":
        cx = min(1.75, max(-0.75, cx))
        cy = min(1.75, max(-0.75, cy))
    px_w = max(MIN_SIZE_PX, w * target_w)
    px_h = max(MIN_SIZE_PX, h * target_h)
    return [cx, cy, px_w / target_w, px_h / target_h]


def bbox_from_center_norm(pred_norm: list[float], target_w: float, target_h: float) -> list[float]:
    cx, cy, w, h = denormalize_center_size(*pred_norm, target_w, target_h)
    w = max(MIN_SIZE_PX, w)
    h = max(MIN_SIZE_PX, h)
    return [cx - w / 2.0, cy - h / 2.0, w, h]


def _root(tree: Any) -> dict[str, Any]:
    if isinstance(tree, list):
        if not tree or not isinstance(tree[0], dict):
            raise ValueError("JSON list must contain a root object")
        return tree[0]
    if isinstance(tree, dict):
        return tree
    raise ValueError("JSON must be a root object or list with root object")


def _set_bounds(node: dict[str, Any], bbox: list[float]) -> None:
    bounds = node.get("bounds")
    if not isinstance(bounds, dict):
        bounds = {}
        node["bounds"] = bounds
    bounds["x"], bounds["y"], bounds["width"], bounds["height"] = bbox


def _resolve_top_child_index_by_path(path_value: Any) -> int | None:
    path = str(path_value or "").strip()
    if not path:
        return None
    head = path.split("/", 1)[0].strip()
    if not head.isdigit():
        return None
    return int(head)


def apply_predictions_to_json(
    json_obj: Any,
    record: dict[str, Any],
    predictions_by_token_id: dict[str, list[float]],
    target_w: float,
    target_h: float,
    *,
    warnings: list[str] | None = None,
) -> Any:
    """Return a deep-copied JSON with only root and root.children bounds updated."""
    if warnings is None:
        warnings = []
    tree = copy.deepcopy(json_obj)
    root = _root(tree)
    root["width"] = target_w
    root["height"] = target_h
    root_bounds = root.get("bounds") if isinstance(root.get("bounds"), dict) else {}
    root_bounds["width"] = target_w
    root_bounds["height"] = target_h
    root["bounds"] = root_bounds
    root["name"] = f"{root.get('name') or 'banner'}_{int(target_w)}x{int(target_h)}"
    children = root.get("children")
    if not isinstance(children, list):
        raise ValueError("Root children missing")

    for token in record.get("tokens") or []:
        token_id = str(token.get("token_id"))
        pred = predictions_by_token_id.get(token_id)
        if pred is None:
            continue
        pred = postprocess_token_prediction(str(token.get("train_role")), pred, target_w, target_h)
        target_bbox = bbox_from_center_norm(pred, target_w, target_h)
        source_paths = [str(x) for x in token.get("source_paths") or []]
        indices = []
        for p in source_paths:
            idx = _resolve_top_child_index_by_path(p)
            if idx is None:
                warnings.append(f"unresolved_source_path:{token_id}:{p}")
                continue
            indices.append(idx)
        if not indices:
            indices = [int(i) for i in token.get("source_indices") or [] if isinstance(i, int) or str(i).isdigit()]
        if not indices:
            warnings.append(f"missing_source_indices:{token_id}")
            continue
        if len(indices) == 1:
            idx = indices[0]
            if 0 <= idx < len(children) and isinstance(children[idx], dict):
                _set_bounds(children[idx], target_bbox)
            else:
                warnings.append(f"invalid_top_child_index:{token_id}:{idx}")
            continue

        source_boxes = [bbox_xywh(children[idx]) for idx in indices if 0 <= idx < len(children) and isinstance(children[idx], dict)]
        if not source_boxes:
            warnings.append(f"no_valid_cluster_nodes:{token_id}")
            continue
        source_union = bbox_union(source_boxes)
        sx, sy, sw, sh = source_union
        tx, ty, tw, th = target_bbox
        for idx in indices:
            if idx < 0 or idx >= len(children) or not isinstance(children[idx], dict):
                warnings.append(f"invalid_top_child_index:{token_id}:{idx}")
                continue
            bx, by, bw, bh = bbox_xywh(children[idx])
            rel_x = 0.0 if sw == 0 else (bx - sx) / sw
            rel_y = 0.0 if sh == 0 else (by - sy) / sh
            rel_w = 1.0 if sw == 0 else bw / sw
            rel_h = 1.0 if sh == 0 else bh / sh
            _set_bounds(children[idx], [tx + rel_x * tw, ty + rel_y * th, max(MIN_SIZE_PX, rel_w * tw), max(MIN_SIZE_PX, rel_h * th)])
    return tree
