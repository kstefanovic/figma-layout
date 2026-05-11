"""Utilities for reading clean semantic Figma banner JSON."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .roles import NUM_ROLES, ROLE_TO_IDX, ROLES

IGNORED_NAMES = {
    "background_shape",
    "background_gradient_1",
    "background_gradient_2",
    "star_decoration_1",
    "star_decoration_2",
    "decoration_group",
    "background",
}


def normalize_name(name: str) -> str:
    """Normalize a Figma layer name into one of the predicted roles when possible."""
    raw = str(name or "").strip().lower()
    raw = re.sub(r"\s+", "_", raw)
    raw = raw.replace("-", "_")
    if not raw:
        return ""
    if raw in IGNORED_NAMES:
        return ""
    if raw == "0+" or "0+" in raw:
        return "age_badge"
    if raw == "image_zone" or raw.startswith("image_zone_"):
        return "hero_image"
    if raw == "hero_image" or raw.startswith("hero_image_"):
        return "hero_image"
    if raw.startswith("headline_group"):
        return "headline_group"
    if raw.startswith("legal_text") or raw == "legal" or raw.startswith("legal_"):
        return "legal_text"
    if raw.startswith("age_badge"):
        return "age_badge"
    if raw.startswith("brand_group"):
        return "brand_group"
    for role in ROLES:
        if raw == role or raw.startswith(role + "_"):
            return role
    return ""


def collect_nodes(node: dict) -> list[dict]:
    """Return a preorder list of all dict nodes in a Figma JSON tree."""
    out: list[dict] = []

    def walk(item: Any) -> None:
        if not isinstance(item, dict):
            return
        out.append(item)
        for child in item.get("children") or []:
            walk(child)

    walk(node)
    return out


def find_role_node(banner: dict, role: str) -> dict | None:
    """Find the largest matching node for a semantic role."""
    if role not in ROLE_TO_IDX:
        raise ValueError(f"unknown role {role!r}")
    matches = [
        node
        for node in collect_nodes(banner)
        if normalize_name(str(node.get("name", ""))) == role
    ]
    if not matches:
        return None
    return max(matches, key=_area)


def get_text_content(node: dict) -> str:
    """Return text directly stored on a Figma JSON node."""
    for key in ("characters", "text", "content"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_all_text(node: dict) -> str:
    """Return all descendant text content joined in tree order."""
    pieces: list[str] = []
    for item in collect_nodes(node):
        text = get_text_content(item)
        if text:
            pieces.append(text)
    return " ".join(pieces)


def get_banner_size(banner: dict) -> tuple[float, float]:
    bounds = banner.get("bounds") if isinstance(banner, dict) else None
    if not isinstance(bounds, dict):
        raise ValueError("banner is missing bounds")
    width = float(bounds.get("width") or 0)
    height = float(bounds.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"banner bounds must include positive width/height, got {bounds!r}")
    return width, height


def get_role_box_norm(banner: dict, role: str) -> list[float] | None:
    """Return [x_norm, y_norm, w_norm, h_norm] for a role, or None when missing."""
    node = find_role_node(banner, role)
    if node is None:
        return None
    bounds = node.get("bounds")
    if not isinstance(bounds, dict):
        return None
    width, height = get_banner_size(banner)
    return [
        _safe_float(bounds.get("x")) / width,
        _safe_float(bounds.get("y")) / height,
        _safe_float(bounds.get("width")) / width,
        _safe_float(bounds.get("height")) / height,
    ]


def extract_role_boxes(banner: dict) -> np.ndarray:
    """Return normalized boxes with shape [NUM_ROLES, 4]. Missing roles are zeros."""
    boxes = np.zeros((NUM_ROLES, 4), dtype=np.float32)
    for role, idx in ROLE_TO_IDX.items():
        box = get_role_box_norm(banner, role)
        if box is not None:
            boxes[idx] = np.asarray(box, dtype=np.float32)
    return boxes


def extract_role_mask(banner: dict) -> np.ndarray:
    """Return a float mask with shape [NUM_ROLES], where 1 means role is present."""
    mask = np.zeros((NUM_ROLES,), dtype=np.float32)
    for role, idx in ROLE_TO_IDX.items():
        if get_role_box_norm(banner, role) is not None:
            mask[idx] = 1.0
    return mask


def _area(node: dict) -> float:
    bounds = node.get("bounds")
    if not isinstance(bounds, dict):
        return 0.0
    return max(0.0, _safe_float(bounds.get("width"))) * max(0.0, _safe_float(bounds.get("height")))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return 0.0
