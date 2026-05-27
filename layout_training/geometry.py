"""Geometry helpers for top-level Figma layout records."""

from __future__ import annotations

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


def bbox_xywh(node: Any) -> tuple[float, float, float, float]:
    bounds = get_bounds(node)
    return (
        safe_float(bounds.get("x")),
        safe_float(bounds.get("y")),
        safe_float(bounds.get("width")),
        safe_float(bounds.get("height")),
    )


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


def bbox_union(bboxes: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    boxes = [b for b in bboxes if b[2] > 0 and b[3] > 0]
    if not boxes:
        return 0.0, 0.0, 0.0, 0.0
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return x1, y1, x2 - x1, y2 - y1


def clipped_bbox_to_canvas(
    bbox: tuple[float, float, float, float],
    canvas_w: float,
    canvas_h: float,
) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x1 = max(0.0, min(canvas_w, x))
    y1 = max(0.0, min(canvas_h, y))
    x2 = max(0.0, min(canvas_w, x + w))
    y2 = max(0.0, min(canvas_h, y + h))
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def area_ratio(bbox: tuple[float, float, float, float], canvas_w: float, canvas_h: float) -> float:
    return max(0.0, bbox[2]) * max(0.0, bbox[3]) / max(EPS, canvas_w * canvas_h)


def orientation(width: float, height: float) -> str:
    if abs(width - height) <= max(width, height, 1.0) * 0.03:
        return "square"
    return "landscape" if width > height else "portrait"

