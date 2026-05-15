"""Extract and apply normalized semantic role bounding boxes."""

from __future__ import annotations

import copy
import warnings
from typing import Any

from .roles import CHILD_ROLES, FLOATING_ROLES, ROLES

CHILD_PARENT = {
    "headline": "headline_group",
    "subheadline_delivery_time": "headline_group",
    "logo": "brand_group",
    "logo_back": "logo",
    "logo_fore": "logo",
    "brand_name_first_part_1": "brand_group",
    "brand_name_first_part_2": "brand_group",
    "brand_name_second": "brand_group",
}


def get_canvas_size(frame: dict[str, Any]) -> tuple[float, float]:
    """Return a frame's canvas width and height from its bounds."""
    bounds = frame.get("bounds")
    if not isinstance(bounds, dict):
        raise ValueError("frame is missing bounds")
    width = _safe_float(bounds.get("width"))
    height = _safe_float(bounds.get("height"))
    if width <= 0 or height <= 0:
        raise ValueError(f"frame bounds must include positive width/height, got {bounds!r}")
    return width, height


def flatten_semantic_nodes(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the largest node for each known semantic role in a frame."""
    out: dict[str, dict[str, Any]] = {}
    for node in _walk_nodes(frame):
        role = node.get("name")
        if role not in ROLES:
            continue
        if role in out:
            current_area = _area(out[role])
            new_area = _area(node)
            warnings.warn(
                f"duplicate role {role!r}; keeping largest area node",
                RuntimeWarning,
                stacklevel=2,
            )
            if new_area > current_area:
                out[role] = node
        else:
            out[role] = node
    return out


def get_bbox_norm(node: dict[str, Any], canvas_w: float, canvas_h: float) -> list[float]:
    """Return [x/w, y/h, width/w, height/h] for a node."""
    bounds = node.get("bounds")
    if not isinstance(bounds, dict):
        raise ValueError(f"node {node.get('name')!r} is missing bounds")
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError("canvas size must be positive")
    return [
        _safe_float(bounds.get("x")) / canvas_w,
        _safe_float(bounds.get("y")) / canvas_h,
        _safe_float(bounds.get("width")) / canvas_w,
        _safe_float(bounds.get("height")) / canvas_h,
    ]


def denorm_bbox(bbox_norm: list[float] | tuple[float, ...], target_w: float, target_h: float) -> dict[str, float]:
    """Convert normalized bbox values into absolute Figma-style bounds."""
    return {
        "x": float(bbox_norm[0]) * target_w,
        "y": float(bbox_norm[1]) * target_h,
        "width": float(bbox_norm[2]) * target_w,
        "height": float(bbox_norm[3]) * target_h,
    }


def copy_json_with_predicted_bounds(
    source_json: dict[str, Any],
    pred_role_bboxes: dict[str, list[float] | tuple[float, ...]],
    target_w: float,
    target_h: float,
) -> dict[str, Any]:
    """Copy source JSON and replace bounds for predicted semantic roles."""
    out = copy.deepcopy(source_json)
    root_bounds = out.setdefault("bounds", {})
    root_bounds["x"] = 0
    root_bounds["y"] = 0
    root_bounds["width"] = float(target_w)
    root_bounds["height"] = float(target_h)

    nodes = flatten_semantic_nodes(out)
    for role, bbox_norm in pred_role_bboxes.items():
        if role not in nodes:
            continue
        node = nodes[role]
        bounds = node.setdefault("bounds", {})
        bounds.update(denorm_bbox(bbox_norm, target_w, target_h))
    return out


def apply_child_relative_transform(
    source_json: dict[str, Any],
    output_json: dict[str, Any],
    role_to_predicted_bbox: dict[str, dict[str, float]],
) -> None:
    """Move child roles by preserving their source-relative bbox inside predicted parents."""
    source_nodes = flatten_semantic_nodes(source_json)
    output_nodes = flatten_semantic_nodes(output_json)
    for child_role in CHILD_ROLES:
        parent_role = CHILD_PARENT.get(child_role)
        if parent_role is None:
            continue
        source_child = source_nodes.get(child_role)
        source_parent = source_nodes.get(parent_role)
        output_child = output_nodes.get(child_role)
        predicted_parent = role_to_predicted_bbox.get(parent_role)
        if not source_child or not source_parent or not output_child or not predicted_parent:
            continue
        source_parent_bounds = _bounds(source_parent)
        source_child_bounds = _bounds(source_child)
        if source_parent_bounds["width"] == 0 or source_parent_bounds["height"] == 0:
            continue
        rel = {
            "x": (source_child_bounds["x"] - source_parent_bounds["x"]) / source_parent_bounds["width"],
            "y": (source_child_bounds["y"] - source_parent_bounds["y"]) / source_parent_bounds["height"],
            "width": source_child_bounds["width"] / source_parent_bounds["width"],
            "height": source_child_bounds["height"] / source_parent_bounds["height"],
        }
        _set_bounds(
            output_child,
            {
                "x": predicted_parent["x"] + rel["x"] * predicted_parent["width"],
                "y": predicted_parent["y"] + rel["y"] * predicted_parent["height"],
                "width": rel["width"] * predicted_parent["width"],
                "height": rel["height"] * predicted_parent["height"],
            },
        )


def place_age_badge_by_anchor(
    source_json: dict[str, Any],
    output_json: dict[str, Any],
    target_w: float,
    target_h: float,
) -> None:
    """Place age_badge by nearest source corner, preserving size and margins."""
    _place_role_by_anchor(source_json, output_json, "age_badge", target_w, target_h, corners_only=True)


def place_floating_roles_by_anchor(
    source_json: dict[str, Any],
    output_json: dict[str, Any],
    target_w: float,
    target_h: float,
) -> None:
    """Place decorative/floating roles with deterministic anchor-copy rules."""
    for role in FLOATING_ROLES:
        if role == "age_badge":
            continue
        _place_role_by_anchor(source_json, output_json, role, target_w, target_h, corners_only=False)


def _walk_nodes(node: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []

    def walk(item: Any) -> None:
        if not isinstance(item, dict):
            return
        nodes.append(item)
        for child in item.get("children") or []:
            walk(child)

    walk(node)
    return nodes


def _place_role_by_anchor(
    source_json: dict[str, Any],
    output_json: dict[str, Any],
    role: str,
    target_w: float,
    target_h: float,
    corners_only: bool,
) -> None:
    source_nodes = flatten_semantic_nodes(source_json)
    output_nodes = flatten_semantic_nodes(output_json)
    source_node = source_nodes.get(role)
    output_node = output_nodes.get(role)
    if not source_node or not output_node:
        return

    source_w, source_h = get_canvas_size(source_json)
    source_min = max(1.0, min(source_w, source_h))
    target_min = max(1.0, min(target_w, target_h))
    source_bounds = _bounds(source_node)
    anchor_x, anchor_y = _nearest_anchor(source_bounds, source_w, source_h, corners_only)

    width = source_bounds["width"] / source_min * target_min
    height = source_bounds["height"] / source_min * target_min
    if anchor_x == "left":
        x = source_bounds["x"] / source_min * target_min
    elif anchor_x == "right":
        x = target_w - ((source_w - source_bounds["x"] - source_bounds["width"]) / source_min * target_min) - width
    else:
        source_center_x = source_bounds["x"] + source_bounds["width"] / 2.0
        x = (source_center_x / source_w) * target_w - width / 2.0

    if anchor_y == "top":
        y = source_bounds["y"] / source_min * target_min
    elif anchor_y == "bottom":
        y = target_h - ((source_h - source_bounds["y"] - source_bounds["height"]) / source_min * target_min) - height
    else:
        source_center_y = source_bounds["y"] + source_bounds["height"] / 2.0
        y = (source_center_y / source_h) * target_h - height / 2.0

    _set_bounds(output_node, {"x": x, "y": y, "width": width, "height": height})


def _nearest_anchor(
    bounds: dict[str, float],
    canvas_w: float,
    canvas_h: float,
    corners_only: bool,
) -> tuple[str, str]:
    center_x = bounds["x"] + bounds["width"] / 2.0
    center_y = bounds["y"] + bounds["height"] / 2.0
    if corners_only:
        return ("left" if center_x <= canvas_w / 2.0 else "right", "top" if center_y <= canvas_h / 2.0 else "bottom")

    x_options = {
        "left": abs(center_x),
        "center": abs(center_x - canvas_w / 2.0),
        "right": abs(canvas_w - center_x),
    }
    y_options = {
        "top": abs(center_y),
        "center": abs(center_y - canvas_h / 2.0),
        "bottom": abs(canvas_h - center_y),
    }
    return min(x_options, key=x_options.get), min(y_options, key=y_options.get)


def _bounds(node: dict[str, Any]) -> dict[str, float]:
    bounds = node.get("bounds") if isinstance(node, dict) else None
    if not isinstance(bounds, dict):
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    return {
        "x": _safe_float(bounds.get("x")),
        "y": _safe_float(bounds.get("y")),
        "width": _safe_float(bounds.get("width")),
        "height": _safe_float(bounds.get("height")),
    }


def _set_bounds(node: dict[str, Any], bounds: dict[str, float]) -> None:
    node_bounds = node.setdefault("bounds", {})
    node_bounds.update({key: float(value) for key, value in bounds.items()})


def _area(node: dict[str, Any]) -> float:
    bounds = node.get("bounds")
    if not isinstance(bounds, dict):
        return 0.0
    return max(0.0, _safe_float(bounds.get("width"))) * max(0.0, _safe_float(bounds.get("height")))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if out == out and out not in (float("inf"), float("-inf")) else 0.0
