"""Geometry helpers for top-level Figma layout records."""

from __future__ import annotations

import math
from typing import Any, Iterable

from .config import EPS


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_bounds(node: Any) -> dict[str, Any]:
    if isinstance(node, dict) and isinstance(node.get("bounds"), dict):
        return node["bounds"]
    return {}


def get_layout_bounds(node: Any) -> dict[str, Any]:
    if isinstance(node, dict) and isinstance(node.get("layoutBounds"), dict):
        return node["layoutBounds"]
    return get_bounds(node)


def get_rotation_deg(node: Any) -> float:
    if not isinstance(node, dict):
        return 0.0
    if "rotation" in node:
        return safe_float(node.get("rotation"))
    transform = node.get("relativeTransform")
    if isinstance(transform, list) and len(transform) >= 2:
        try:
            a = float(transform[0][0])
            b = float(transform[1][0])
            return math.degrees(math.atan2(b, a))
        except (TypeError, ValueError, IndexError):
            return 0.0
    return 0.0


def _bounds_xywh(bounds: Any) -> tuple[float, float, float, float]:
    if not isinstance(bounds, dict):
        return 0.0, 0.0, 0.0, 0.0
    return (
        safe_float(bounds.get("x")),
        safe_float(bounds.get("y")),
        safe_float(bounds.get("width")),
        safe_float(bounds.get("height")),
    )


def bbox_xywh(node: Any) -> tuple[float, float, float, float]:
    return _bounds_xywh(get_bounds(node))


def compute_visual_bounds_from_layout_bounds(bounds: Any, rotation_deg: float) -> dict[str, float]:
    x, y, w, h = _bounds_xywh(bounds)
    if w <= 0 or h <= 0:
        return {"x": x, "y": y, "width": max(0.0, w), "height": max(0.0, h)}
    theta = math.radians(rotation_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    corners = ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h))
    xs: list[float] = []
    ys: list[float] = []
    for lx, ly in corners:
        xs.append(x + c * lx - s * ly)
        ys.append(y + s * lx + c * ly)
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    return {"x": min_x, "y": min_y, "width": max_x - min_x, "height": max_y - min_y}


def _compute_visual_bounds_from_absolute_transform(node: Any) -> dict[str, float] | None:
    if not isinstance(node, dict):
        return None
    transform = node.get("absoluteTransform")
    if not (isinstance(transform, list) and len(transform) >= 2):
        return None
    bounds = get_layout_bounds(node)
    _x, _y, w, h = _bounds_xywh(bounds)
    if w <= 0 or h <= 0:
        return {"x": 0.0, "y": 0.0, "width": max(0.0, w), "height": max(0.0, h)}
    try:
        a = float(transform[0][0])
        c = float(transform[0][1])
        tx = float(transform[0][2])
        b = float(transform[1][0])
        d = float(transform[1][1])
        ty = float(transform[1][2])
    except (TypeError, ValueError, IndexError):
        return None
    corners = ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h))
    xs: list[float] = []
    ys: list[float] = []
    for lx, ly in corners:
        xs.append(a * lx + c * ly + tx)
        ys.append(b * lx + d * ly + ty)
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    return {"x": min_x, "y": min_y, "width": max_x - min_x, "height": max_y - min_y}


def get_visual_bounds(node: Any) -> dict[str, float]:
    if isinstance(node, dict) and isinstance(node.get("visualBounds"), dict):
        return {
            "x": safe_float(node["visualBounds"].get("x")),
            "y": safe_float(node["visualBounds"].get("y")),
            "width": safe_float(node["visualBounds"].get("width")),
            "height": safe_float(node["visualBounds"].get("height")),
        }
    from_transform = _compute_visual_bounds_from_absolute_transform(node)
    if from_transform is not None:
        return from_transform
    if isinstance(node, dict):
        return compute_visual_bounds_from_layout_bounds(get_layout_bounds(node), get_rotation_deg(node))
    return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}


def get_visual_center(node: Any) -> tuple[float, float]:
    if isinstance(node, dict) and isinstance(node.get("visualCenter"), dict):
        return (
            safe_float(node["visualCenter"].get("x")),
            safe_float(node["visualCenter"].get("y")),
        )
    b = get_visual_bounds(node)
    return safe_float(b.get("x")) + safe_float(b.get("width")) / 2.0, safe_float(b.get("y")) + safe_float(b.get("height")) / 2.0


def center_size_from_visual_bounds(node: Any) -> tuple[float, float, float, float]:
    b = get_visual_bounds(node)
    x = safe_float(b.get("x"))
    y = safe_float(b.get("y"))
    w = safe_float(b.get("width"))
    h = safe_float(b.get("height"))
    return x + w / 2.0, y + h / 2.0, w, h


def center_size_from_bbox(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return x + w / 2.0, y + h / 2.0, w, h


def normalize_center_size(
    cx: float,
    cy: float,
    w: float,
    h: float,
    canvas_w: float,
    canvas_h: float,
) -> tuple[float, float, float, float]:
    return cx / max(EPS, canvas_w), cy / max(EPS, canvas_h), w / max(EPS, canvas_w), h / max(EPS, canvas_h)


def denormalize_center_size(
    cx_norm: float,
    cy_norm: float,
    w_norm: float,
    h_norm: float,
    canvas_w: float,
    canvas_h: float,
) -> tuple[float, float, float, float]:
    return cx_norm * canvas_w, cy_norm * canvas_h, w_norm * canvas_w, h_norm * canvas_h


def bbox_union(bboxes: Iterable[Any]) -> tuple[float, float, float, float]:
    boxes: list[tuple[float, float, float, float]] = []
    for bbox in bboxes:
        if isinstance(bbox, dict):
            box = _bounds_xywh(bbox)
        else:
            box = tuple(float(x) for x in bbox[:4])  # type: ignore[index]
        if box[2] > 0 and box[3] > 0:
            boxes.append(box)
    if not boxes:
        return 0.0, 0.0, 0.0, 0.0
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return x1, y1, x2 - x1, y2 - y1


def clip_bbox_to_canvas(
    bbox: Any,
    canvas_w: float,
    canvas_h: float,
) -> tuple[float, float, float, float]:
    x, y, w, h = _bounds_xywh(bbox) if isinstance(bbox, dict) else tuple(float(v) for v in bbox[:4])  # type: ignore[index]
    x1 = max(0.0, min(canvas_w, x))
    y1 = max(0.0, min(canvas_h, y))
    x2 = max(0.0, min(canvas_w, x + w))
    y2 = max(0.0, min(canvas_h, y + h))
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def clipped_bbox_to_canvas(
    bbox: tuple[float, float, float, float],
    canvas_w: float,
    canvas_h: float,
) -> tuple[float, float, float, float]:
    return clip_bbox_to_canvas(bbox, canvas_w, canvas_h)


def coverage_ratio(bbox: Any, canvas_w: float, canvas_h: float) -> float:
    clipped = clip_bbox_to_canvas(bbox, canvas_w, canvas_h)
    return max(0.0, clipped[2]) * max(0.0, clipped[3]) / max(EPS, canvas_w * canvas_h)


def area_ratio(bbox: Any, canvas_w: float, canvas_h: float) -> float:
    x, y, w, h = _bounds_xywh(bbox) if isinstance(bbox, dict) else tuple(float(v) for v in bbox[:4])  # type: ignore[index]
    return max(0.0, w) * max(0.0, h) / max(EPS, canvas_w * canvas_h)


def bleed_flags(bbox: Any, canvas_w: float, canvas_h: float) -> tuple[bool, bool, bool, bool]:
    x, y, w, h = _bounds_xywh(bbox) if isinstance(bbox, dict) else tuple(float(v) for v in bbox[:4])  # type: ignore[index]
    return x < 0.0, x + w > canvas_w, y < 0.0, y + h > canvas_h


def orientation(width: float, height: float) -> str:
    if abs(width - height) <= max(width, height, 1.0) * 0.03:
        return "square"
    return "landscape" if width > height else "portrait"
