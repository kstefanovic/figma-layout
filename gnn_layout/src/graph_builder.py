"""Convert a source banner JSON tree into a PyTorch Geometric graph."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch_geometric.data import Data

from .orientation import get_orientation, orientation_to_onehot
from .roles import NODE_ROLE_FEATURES, ROLE_TO_IDX
from .semantic_utils import get_banner_size, get_text_content, normalize_name

FEATURE_NAMES = [
    "x_norm",
    "y_norm",
    "w_norm",
    "h_norm",
    "center_x_norm",
    "center_y_norm",
    "area_norm",
    "aspect_ratio",
    "depth",
    "child_count",
    "is_text",
    "is_rectangle",
    "is_vector",
    "is_group",
    "is_frame",
    "is_star",
    "is_ellipse",
    "is_boolean_operation",
    "text_length_norm",
    "has_digits",
    "near_top",
    "near_bottom",
    "near_left",
    "near_right",
] + [f"role_{role}" for role in NODE_ROLE_FEATURES]


@dataclass
class FlatNode:
    node: dict[str, Any]
    index: int
    parent: int | None
    depth: int


def build_graph(source_banner: dict[str, Any]) -> Data:
    """Build a PyG Data graph with source-only features."""
    source_width, source_height = get_banner_size(source_banner)
    flat = _flatten(source_banner)
    if not flat:
        raise ValueError("source banner contains no nodes")

    x = torch.tensor(
        [_node_features(item, source_width, source_height) for item in flat],
        dtype=torch.float32,
    )
    edge_index = _build_edges(flat, source_width, source_height)
    orientation = get_orientation(source_width, source_height)
    data = Data(x=x, edge_index=edge_index)
    data.source_width = torch.tensor([source_width], dtype=torch.float32)
    data.source_height = torch.tensor([source_height], dtype=torch.float32)
    data.source_orientation_onehot = torch.tensor(
        orientation_to_onehot(orientation),
        dtype=torch.float32,
    )
    return data


def _flatten(root: dict[str, Any]) -> list[FlatNode]:
    out: list[FlatNode] = []

    def walk(node: dict[str, Any], parent: int | None, depth: int) -> None:
        index = len(out)
        out.append(FlatNode(node=node, index=index, parent=parent, depth=depth))
        for child in node.get("children") or []:
            if isinstance(child, dict):
                walk(child, index, depth + 1)

    walk(root, None, 0)
    return out


def _node_features(item: FlatNode, banner_w: float, banner_h: float) -> list[float]:
    node = item.node
    b = _bounds(node)
    x = b["x"] / banner_w
    y = b["y"] / banner_h
    w = b["width"] / banner_w
    h = b["height"] / banner_h
    cx = x + w * 0.5
    cy = y + h * 0.5
    node_type = str(node.get("type") or "").lower().replace(" ", "_")
    text = get_text_content(node)
    role = normalize_name(str(node.get("name") or ""))
    role_values = [0.0] * len(NODE_ROLE_FEATURES)
    role_values[NODE_ROLE_FEATURES.index(role if role in ROLE_TO_IDX else "other")] = 1.0

    return [
        _clip_reasonable(x),
        _clip_reasonable(y),
        _clip_reasonable(w),
        _clip_reasonable(h),
        _clip_reasonable(cx),
        _clip_reasonable(cy),
        _clip_reasonable(max(0.0, w * h)),
        _clip_reasonable(b["width"] / max(1.0, b["height"])),
        min(item.depth / 12.0, 1.0),
        min(len(node.get("children") or []) / 20.0, 1.0),
        float(node_type == "text"),
        float(node_type == "rectangle"),
        float(node_type == "vector"),
        float(node_type == "group"),
        float(node_type == "frame"),
        float(node_type == "star"),
        float(node_type == "ellipse"),
        float(node_type == "boolean_operation"),
        min(len(text) / 120.0, 1.0),
        float(any(ch.isdigit() for ch in text)),
        float(y < 0.15),
        float(y + h > 0.85),
        float(x < 0.15),
        float(x + w > 0.85),
        *role_values,
    ]


def _build_edges(flat: list[FlatNode], banner_w: float, banner_h: float) -> torch.Tensor:
    edges: set[tuple[int, int]] = set()

    children_by_parent: dict[int, list[int]] = {}
    for item in flat:
        if item.parent is not None:
            edges.add((item.parent, item.index))
            edges.add((item.index, item.parent))
            children_by_parent.setdefault(item.parent, []).append(item.index)

    for siblings in children_by_parent.values():
        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                a, b = siblings[i], siblings[j]
                edges.add((a, b))
                edges.add((b, a))

    boxes = [_norm_box(item.node, banner_w, banner_h) for item in flat]
    centers = [(box[0] + box[2] * 0.5, box[1] + box[3] * 0.5) for box in boxes]
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            dist = math.dist(centers[i], centers[j])
            if dist < 0.25 or _iou(boxes[i], boxes[j]) > 0.05:
                edges.add((i, j))
                edges.add((j, i))

    if not edges:
        edges.add((0, 0))
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()


def _bounds(node: dict[str, Any]) -> dict[str, float]:
    raw = node.get("bounds")
    if not isinstance(raw, dict):
        raw = {}
    return {
        "x": _finite(raw.get("x")),
        "y": _finite(raw.get("y")),
        "width": max(0.0, _finite(raw.get("width"))),
        "height": max(0.0, _finite(raw.get("height"))),
    }


def _norm_box(node: dict[str, Any], banner_w: float, banner_h: float) -> tuple[float, float, float, float]:
    b = _bounds(node)
    return (b["x"] / banner_w, b["y"] / banner_h, b["width"] / banner_w, b["height"] / banner_h)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _finite(value: Any) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _clip_reasonable(value: float) -> float:
    return float(max(-4.0, min(4.0, value)))
