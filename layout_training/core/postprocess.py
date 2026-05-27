"""Prediction postprocess and JSON application for the CORE layout model."""

from __future__ import annotations

import copy
from typing import Any

from layout_training.config import MIN_SIZE_PX
from layout_training.geometry import (
    bbox_union,
    center_size_from_bbox,
    clip_bbox_to_canvas,
    compute_visual_bounds_from_layout_bounds,
    denormalize_center_size,
    get_layout_bounds,
    get_rotation_deg,
    get_visual_bounds,
    safe_float,
    coverage_ratio,
    bleed_flags,
)


def clamp_prediction_norm(values: list[float]) -> list[float]:
    cx, cy, w, h = [float(x) for x in values[:4]]
    return [
        min(2.0, max(-1.0, cx)),
        min(2.0, max(-1.0, cy)),
        min(3.0, max(0.01, w)),
        min(3.0, max(0.01, h)),
    ]


def postprocess_token_prediction(train_role: str, pred_norm: list[float], target_w: float, target_h: float, source_token: dict[str, Any]) -> list[float]:
    cx, cy, w, h = clamp_prediction_norm(pred_norm)
    if train_role == "background_cluster":
        cx, cy, w, h = 0.5, 0.5, max(1.0, w), max(1.0, h)
    if train_role == "brand_group":
        cx = min(0.96, max(0.04, cx))
        cy = min(0.96, max(0.04, cy))
    if train_role == "text_main_group":
        cx = min(0.96, max(0.04, cx))
        cy = min(0.96, max(0.04, cy))
        w = min(0.92, max(0.08, w))
        h = min(0.5, max(0.04, h))
    if train_role == "hero_group":
        cx = min(1.3, max(-0.3, cx))
        cy = min(1.3, max(-0.3, cy))
    if train_role == "legal_group":
        cx = min(0.96, max(0.04, cx))
        cy = min(0.98, max(0.75, cy))
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


def _layout_bbox_for_visual_target(node: dict[str, Any], target_visual_bbox: list[float]) -> list[float]:
    src_layout = get_layout_bounds(node)
    src_visual = get_visual_bounds(node)
    rotation = get_rotation_deg(node)
    lx = safe_float(src_layout.get("x"))
    ly = safe_float(src_layout.get("y"))
    lw = max(MIN_SIZE_PX, safe_float(src_layout.get("width"), 1.0))
    lh = max(MIN_SIZE_PX, safe_float(src_layout.get("height"), 1.0))
    svx = safe_float(src_visual.get("x"))
    svy = safe_float(src_visual.get("y"))
    svw = max(MIN_SIZE_PX, safe_float(src_visual.get("width"), 1.0))
    svh = max(MIN_SIZE_PX, safe_float(src_visual.get("height"), 1.0))
    tx, ty, tw, th = [float(x) for x in target_visual_bbox[:4]]
    scale_x = tw / max(svw, MIN_SIZE_PX)
    scale_y = th / max(svh, MIN_SIZE_PX)
    new_lw = max(MIN_SIZE_PX, lw * scale_x)
    new_lh = max(MIN_SIZE_PX, lh * scale_y)
    rotated = compute_visual_bounds_from_layout_bounds({"x": 0.0, "y": 0.0, "width": new_lw, "height": new_lh}, rotation)
    new_x = tx - safe_float(rotated.get("x"))
    new_y = ty - safe_float(rotated.get("y"))
    return [new_x, new_y, new_lw, new_lh]


def _apply_cluster_transform(children: list[dict[str, Any]], indices: list[int], target_bbox: list[float], warnings: list[str], token_id: str) -> list[list[float]]:
    valid = [idx for idx in indices if 0 <= idx < len(children) and isinstance(children[idx], dict)]
    if not valid:
        warnings.append(f"no_valid_cluster_nodes:{token_id}")
        return []
    source_union = bbox_union(get_visual_bounds(children[idx]) for idx in valid)
    sx, sy, sw, sh = source_union
    tx, ty, tw, th = target_bbox
    applied_bounds: list[list[float]] = []
    for idx in valid:
        node = children[idx]
        vb = get_visual_bounds(node)
        bx = safe_float(vb.get("x"))
        by = safe_float(vb.get("y"))
        bw = max(MIN_SIZE_PX, safe_float(vb.get("width"), 1.0))
        bh = max(MIN_SIZE_PX, safe_float(vb.get("height"), 1.0))
        rel_x = 0.0 if sw == 0 else (bx - sx) / sw
        rel_y = 0.0 if sh == 0 else (by - sy) / sh
        rel_w = 1.0 if sw == 0 else bw / sw
        rel_h = 1.0 if sh == 0 else bh / sh
        child_target = [tx + rel_x * tw, ty + rel_y * th, max(MIN_SIZE_PX, rel_w * tw), max(MIN_SIZE_PX, rel_h * th)]
        final_layout = _layout_bbox_for_visual_target(node, child_target)
        _set_bounds(node, final_layout)
        applied_bounds.append(final_layout)
    return applied_bounds


def _apply_legal_bbox(node: dict[str, Any], target_bbox: list[float], target_w: float, target_h: float) -> tuple[list[float], dict[str, Any]]:
    src_layout = get_layout_bounds(node)
    src_w = max(MIN_SIZE_PX, safe_float(src_layout.get("width"), target_w * 0.4))
    src_h = max(MIN_SIZE_PX, safe_float(src_layout.get("height"), target_h * 0.08))
    src_aspect = src_w / max(src_h, MIN_SIZE_PX)
    width = min(src_w * (target_w / max(target_w, 1.0)), target_w * 0.92)
    height = width / max(src_aspect, 0.05)
    if height > target_h * 0.2:
        scale = (target_h * 0.2) / height
        width *= scale
        height *= scale
    cx = target_bbox[0] + target_bbox[2] / 2.0
    predicted_bottom_y = target_bbox[1] + target_bbox[3]
    bottom_margin = target_h * 0.035
    x_margin = target_w * 0.04
    unclamped_x = cx - width / 2.0
    unclamped_y = predicted_bottom_y - height
    x = max(x_margin, min(target_w - x_margin - width, unclamped_x))
    y = min(target_h - bottom_margin - height, unclamped_y)
    y = max(0.0, y)
    clamp_actions: list[str] = []
    if abs(x - unclamped_x) > 1e-6:
        clamp_actions.append("x_clamped_to_safe_margin")
    if abs(y - unclamped_y) > 1e-6:
        clamp_actions.append("bottom_clamped_to_canvas")
    bbox = [x, y, width, height]
    debug = {
        "predicted_center_x": cx,
        "predicted_bottom_y": predicted_bottom_y,
        "deterministic_width": width,
        "deterministic_height": height,
        "final_x": x,
        "final_y": y,
        "final_bottom_y": y + height,
        "clamp_actions": clamp_actions,
        "anchor_decision": "bottom_safe_deterministic_fit",
    }
    return bbox, debug


def apply_core_predictions_to_json(
    json_obj: Any,
    record: dict[str, Any],
    predictions_by_token_id: dict[str, list[float]],
    target_w: float,
    target_h: float,
    *,
    warnings: list[str] | None = None,
    debug_actions: list[dict[str, Any]] | None = None,
) -> Any:
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
    children = root.get("children")
    if not isinstance(children, list):
        raise ValueError("Root children missing")

    actions = debug_actions if debug_actions is not None else []
    for token in record.get("tokens") or []:
        token_id = str(token.get("token_id") or "")
        pred = predictions_by_token_id.get(token_id)
        if pred is None:
            continue
        train_role = str(token.get("train_role") or "")
        source_paths = [str(x) for x in token.get("source_paths") or []]
        pred = postprocess_token_prediction(train_role, pred, target_w, target_h, token)
        predicted_bbox = bbox_from_center_norm(pred, target_w, target_h)
        target_bbox = list(predicted_bbox)
        action_debug: dict[str, Any] = {
            "token_id": token_id,
            "role": train_role,
            "source_paths": source_paths,
            "predicted_bbox": list(predicted_bbox),
        }
        if train_role == "background_cluster":
            action_debug["source_bleed_bbox_visual"] = list(token.get("bleed_bbox_visual") or token.get("bbox_visual") or [])
            action_debug["source_coverage_bbox_visual"] = list(token.get("coverage_bbox_visual") or token.get("bbox_visual") or [])
            action_debug["coverage_ratio_before"] = float(token.get("coverage_ratio") or 0.0)
            action_debug["bleed_flags"] = {
                "left": bool(token.get("bleed_left")),
                "right": bool(token.get("bleed_right")),
                "top": bool(token.get("bleed_top")),
                "bottom": bool(token.get("bleed_bottom")),
            }
            target_bbox = [0.0, 0.0, target_w, target_h]
            action_debug["background_cluster_coverage_decision"] = "expanded_to_cover_canvas"
            action_debug["postprocessed_bbox"] = list(target_bbox)
            action_debug["coverage_ratio_after"] = coverage_ratio(target_bbox, target_w, target_h)
        if train_role == "brand_group":
            target_bbox = list(clip_bbox_to_canvas(target_bbox, target_w, target_h))
        if train_role == "text_main_group":
            clipped = clip_bbox_to_canvas(target_bbox, target_w, target_h)
            target_bbox = [max(0.0, clipped[0]), max(0.0, clipped[1]), min(target_w * 0.92, max(MIN_SIZE_PX, clipped[2])), min(target_h * 0.6, max(MIN_SIZE_PX, clipped[3]))]
        indices = [idx for idx in (_resolve_top_child_index_by_path(path) for path in source_paths) if idx is not None]
        if not indices:
            warnings.append(f"missing_source_indices:{token_id}")
            action_debug["skipped"] = "missing_source_indices"
            actions.append(action_debug)
            continue
        if train_role == "legal_group" and len(indices) == 1 and 0 <= indices[0] < len(children):
            legal_bbox, legal_debug = _apply_legal_bbox(children[indices[0]], target_bbox, target_w, target_h)
            _set_bounds(children[indices[0]], legal_bbox)
            action_debug["action"] = "legal_fit"
            action_debug["legal_group_anchor_decision"] = legal_debug.pop("anchor_decision")
            action_debug.update(legal_debug)
            action_debug["postprocessed_bbox"] = list(target_bbox)
            action_debug["final_applied_bounds"] = [legal_bbox]
            actions.append(action_debug)
            continue
        if len(indices) == 1 and 0 <= indices[0] < len(children) and isinstance(children[indices[0]], dict):
            node = children[indices[0]]
            final_layout = _layout_bbox_for_visual_target(node, target_bbox)
            _set_bounds(node, final_layout)
            action_debug["action"] = "single_transform"
            action_debug["postprocessed_bbox"] = list(target_bbox)
            action_debug["final_applied_bounds"] = [final_layout]
            if train_role == "hero_group" and abs(float(token.get("rotation_deg") or 0.0)) > 0.5:
                action_debug["rotated_hero_detection"] = {
                    "rotation_deg": float(token.get("rotation_deg") or 0.0),
                    "is_rotated": True,
                    "source_visual_bbox": list(token.get("bbox_visual") or []),
                    "predicted_visual_bbox": list(predicted_bbox),
                    "note": "rotation preserved; model predicts visual center/size only",
                }
            actions.append(action_debug)
            continue
        applied = _apply_cluster_transform(children, indices, target_bbox, warnings, token_id)
        action_debug["action"] = "cluster_transform"
        action_debug["postprocessed_bbox"] = list(target_bbox)
        action_debug["final_applied_bounds"] = applied
        actions.append(action_debug)
    return tree
