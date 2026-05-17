"""Utilities for traversing and editing rich semantic Figma JSON."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .schema import ALL_ROLES, GROUP_TYPES


def load_frames(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("frames", "banners", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    raise ValueError(f"{path} must contain a JSON object or list")


def load_one_frame(path: Path, index: int = 0) -> dict[str, Any]:
    frames = load_frames(path)
    if index < 0 or index >= len(frames):
        raise IndexError(f"--source-index {index} is out of range for {len(frames)} frames")
    return frames[index]


def clone_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a rich frame so all Figma metadata is preserved unless explicitly updated."""
    return copy.deepcopy(frame)


def walk_nodes(node: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def walk(item: Any) -> None:
        if not isinstance(item, dict):
            return
        out.append(item)
        for child in item.get("children") or []:
            walk(child)

    walk(node)
    return out


def flatten_role_nodes(frame: dict[str, Any], roles: set[str] | None = None) -> dict[str, dict[str, Any]]:
    allowed = roles or set(ALL_ROLES)
    out: dict[str, dict[str, Any]] = {}
    for node in walk_nodes(frame):
        role = node.get("name")
        if role not in allowed:
            continue
        if role not in out or area(node) > area(out[role]):
            out[role] = node
    return out


def get_canvas_size(frame: dict[str, Any]) -> tuple[float, float]:
    bounds = frame.get("bounds")
    if not isinstance(bounds, dict):
        raise ValueError("frame is missing bounds")
    width = safe_float(bounds.get("width"))
    height = safe_float(bounds.get("height"))
    if width <= 0 or height <= 0:
        raise ValueError(f"frame bounds must include positive width/height, got {bounds!r}")
    return width, height


def bounds_of(node: dict[str, Any]) -> dict[str, float]:
    bounds = node.get("bounds") if isinstance(node, dict) else None
    if not isinstance(bounds, dict):
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    return {key: safe_float(bounds.get(key)) for key in ("x", "y", "width", "height")}


def set_bounds(node: dict[str, Any], bounds: dict[str, float]) -> None:
    node.setdefault("bounds", {}).update({key: float(value) for key, value in bounds.items()})


def normalized_bbox(node: dict[str, Any], canvas_w: float, canvas_h: float) -> list[float]:
    b = bounds_of(node)
    return [b["x"] / canvas_w, b["y"] / canvas_h, b["width"] / canvas_w, b["height"] / canvas_h]


def denormalize_bbox(bbox: list[float] | tuple[float, ...], canvas_w: float, canvas_h: float) -> dict[str, float]:
    return {
        "x": float(bbox[0]) * canvas_w,
        "y": float(bbox[1]) * canvas_h,
        "width": float(bbox[2]) * canvas_w,
        "height": float(bbox[3]) * canvas_h,
    }


def relative_bbox(child: dict[str, Any], parent: dict[str, Any]) -> list[float]:
    c = bounds_of(child)
    p = bounds_of(parent)
    if p["width"] <= 0 or p["height"] <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        (c["x"] - p["x"]) / p["width"],
        (c["y"] - p["y"]) / p["height"],
        c["width"] / p["width"],
        c["height"] / p["height"],
    ]


def apply_relative_bbox(parent_bounds: dict[str, float], rel: list[float] | tuple[float, ...]) -> dict[str, float]:
    return {
        "x": parent_bounds["x"] + float(rel[0]) * parent_bounds["width"],
        "y": parent_bounds["y"] + float(rel[1]) * parent_bounds["height"],
        "width": float(rel[2]) * parent_bounds["width"],
        "height": float(rel[3]) * parent_bounds["height"],
    }


def clamp_canvas_bbox(bbox: list[float] | tuple[float, ...]) -> list[float]:
    width = min(max(float(bbox[2]), 0.001), 2.0)
    height = min(max(float(bbox[3]), 0.001), 2.0)
    x = min(max(float(bbox[0]), -1.0), 1.0)
    y = min(max(float(bbox[1]), -1.0), 1.0)
    if width <= 1.0:
        x = min(max(x, 0.0), 1.0 - width)
    if height <= 1.0:
        y = min(max(y, 0.0), 1.0 - height)
    return [x, y, width, height]


def clamp_relative_bbox(bbox: list[float] | tuple[float, ...]) -> list[float]:
    width = min(max(float(bbox[2]), 0.001), 1.0)
    height = min(max(float(bbox[3]), 0.001), 1.0)
    x = min(max(float(bbox[0]), 0.0), 1.0 - width)
    y = min(max(float(bbox[1]), 0.0), 1.0 - height)
    return [x, y, width, height]


def node_flags(node: dict[str, Any]) -> list[float]:
    fills = node.get("fills") if isinstance(node.get("fills"), list) else []
    effects = node.get("effects") if isinstance(node.get("effects"), list) else []
    node_type = str(node.get("type") or "").lower()
    has_image = any(isinstance(fill, dict) and (fill.get("type") == "IMAGE" or fill.get("imageHash")) for fill in fills)
    has_gradient = any(isinstance(fill, dict) and str(fill.get("type", "")).startswith("GRADIENT") for fill in fills)
    has_effect = any(isinstance(effect, dict) and effect.get("visible", True) for effect in effects)
    is_text = is_text_node(node)
    is_group = node_type in GROUP_TYPES
    return [float(is_text), float(is_group), float(has_image), float(has_gradient), float(has_effect)]


def is_text_node(node: dict[str, Any]) -> bool:
    return str(node.get("type") or "").lower() == "text" or "fontSize" in node


def has_image_hash(node: dict[str, Any]) -> bool:
    for item in walk_nodes(node):
        if item.get("imageHash"):
            return True
        fills = item.get("fills") if isinstance(item.get("fills"), list) else []
        if any(isinstance(fill, dict) and fill.get("imageHash") for fill in fills):
            return True
    return False


def has_gradient_transform(node: dict[str, Any]) -> bool:
    fills = node.get("fills") if isinstance(node.get("fills"), list) else []
    for fill in fills:
        if isinstance(fill, dict) and str(fill.get("type", "")).startswith("GRADIENT"):
            return bool(fill.get("gradientTransform"))
    return False


def area(node: dict[str, Any]) -> float:
    b = bounds_of(node)
    return max(0.0, b["width"]) * max(0.0, b["height"])


def contains(parent: dict[str, float], child: dict[str, float], tolerance: float = 1.0) -> bool:
    return (
        child["x"] >= parent["x"] - tolerance
        and child["y"] >= parent["y"] - tolerance
        and child["x"] + child["width"] <= parent["x"] + parent["width"] + tolerance
        and child["y"] + child["height"] <= parent["y"] + parent["height"] + tolerance
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default

