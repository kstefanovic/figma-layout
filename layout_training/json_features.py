"""Recursive feature extraction for Figma JSON nodes."""

from __future__ import annotations

import math
import re
from typing import Any, Iterator

from .geometry import safe_float


def walk_nodes(node: Any) -> Iterator[dict[str, Any]]:
    if not isinstance(node, dict):
        return
    yield node
    children = node.get("children")
    if isinstance(children, list):
        for child in children:
            yield from walk_nodes(child)


def _fill_types(node: Any) -> list[str]:
    fills = node.get("fills") if isinstance(node, dict) and isinstance(node.get("fills"), list) else []
    return [str(fill.get("type") or "").upper() for fill in fills if isinstance(fill, dict)]


def has_text_deep(node: Any) -> bool:
    return any(isinstance(n.get("characters"), str) and n.get("characters").strip() for n in walk_nodes(node))


def has_image_fill_deep(node: Any) -> bool:
    return any(any(ft == "IMAGE" for ft in _fill_types(n)) for n in walk_nodes(node))


def has_gradient_fill_deep(node: Any) -> bool:
    return any(any("GRADIENT" in ft for ft in _fill_types(n)) for n in walk_nodes(node))


def star_count(node: Any) -> int:
    count = 0
    for n in walk_nodes(node):
        typ = normalized_node_type(n)
        name = str(n.get("name") or "").lower()
        if typ == "star" or "star" in name or "sparkle" in name:
            count += 1
    return count


def all_text(node: Any) -> str:
    parts: list[str] = []
    for n in walk_nodes(node):
        chars = n.get("characters")
        if isinstance(chars, str) and chars.strip():
            parts.append(chars.strip())
    return " ".join(parts)


def discount_text_present(node: Any) -> bool:
    text = re.sub(r"\s+", "", all_text(node))
    return bool(re.fullmatch(r"[-–−]?\d{1,3}%", text))


def count_descendants(node: Any) -> int:
    return max(0, sum(1 for _ in walk_nodes(node)) - 1)


def normalized_node_type(node: Any) -> str:
    return str(node.get("type") if isinstance(node, dict) else "unknown").strip().lower().replace("_", " ") or "unknown"


def rotation_deg(node: Any) -> float:
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

