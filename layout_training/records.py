"""Build canonical top-level layout records from semantic Figma JSON.

Records use root.children only. Nested children are summarized into features but
are never emitted as prediction targets.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from .geometry import (
    area_ratio,
    bbox_union,
    bbox_xywh,
    center_size_from_bbox,
    clipped_bbox_to_canvas,
    normalize_center_size,
    orientation,
    safe_float,
)
from .json_features import (
    count_descendants,
    discount_text_present,
    has_gradient_fill_deep,
    has_image_fill_deep,
    has_text_deep,
    normalized_node_type,
    rotation_deg,
    star_count,
)
from .roles import MERGED_TRAIN_ROLES, ROLE_OCCURRENCE_BY_AREA, ROLE_OCCURRENCE_BY_POSITION, token_id_for, train_role_for


def _root_from_json(json_obj: Any) -> dict[str, Any]:
    if isinstance(json_obj, list):
        if json_obj and isinstance(json_obj[0], dict):
            return json_obj[0]
        raise ValueError("JSON list must contain a root object")
    if isinstance(json_obj, dict):
        return json_obj
    raise ValueError("Semantic JSON must be a root object or list with root object")


def _canvas_size(root: dict[str, Any]) -> tuple[float, float]:
    bounds = root.get("bounds") if isinstance(root.get("bounds"), dict) else {}
    w = safe_float(root.get("width"), safe_float(bounds.get("width")))
    h = safe_float(root.get("height"), safe_float(bounds.get("height")))
    if w <= 0 or h <= 0:
        raise ValueError("Root width/height must be positive")
    return w, h


def _stable_id(file_id: str | None, root: dict[str, Any]) -> str:
    base = file_id or str(root.get("id") or root.get("name") or "semantic_json")
    digest = hashlib.sha1(base.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{base}::{digest}" if file_id else digest


def _semantic_role(child: dict[str, Any]) -> str:
    return str(child.get("semantic_name") or child.get("name") or "unknown_group")


def _token_from_nodes(
    *,
    train_role: str,
    occurrence_index: int,
    nodes: list[tuple[int, dict[str, Any]]],
    canvas_w: float,
    canvas_h: float,
) -> dict[str, Any]:
    bboxes = [bbox_xywh(node) for _idx, node in nodes]
    bbox = bbox_union(bboxes) if len(nodes) > 1 else bboxes[0]
    cx, cy, w, h = center_size_from_bbox(*bbox)
    cxn, cyn, wn, hn = normalize_center_size(cx, cy, w, h, canvas_w, canvas_h)
    clipped = clipped_bbox_to_canvas(bbox, canvas_w, canvas_h)
    clipped_norm = (
        clipped[0] / canvas_w,
        clipped[1] / canvas_h,
        clipped[2] / canvas_w,
        clipped[3] / canvas_h,
    )
    rots = [rotation_deg(node) for _idx, node in nodes]
    rotation = rots[0] if rots else 0.0
    return {
        "token_id": token_id_for(train_role, occurrence_index),
        "train_role": train_role,
        "occurrence_index": occurrence_index,
        "semantic_roles": [_semantic_role(node) for _idx, node in nodes],
        "source_paths": [str(node.get("path", idx)) for idx, node in nodes],
        "source_indices": [idx for idx, _node in nodes],
        "instance_count": len(nodes),
        "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
        "center_size": [cx, cy, w, h],
        "center_size_norm": [cxn, cyn, wn, hn],
        "clipped_bbox_norm": [clipped_norm[0], clipped_norm[1], clipped_norm[2], clipped_norm[3]],
        "type": "cluster" if len(nodes) > 1 or train_role in MERGED_TRAIN_ROLES else normalized_node_type(nodes[0][1]),
        "has_text": any(has_text_deep(node) for _idx, node in nodes),
        "has_image": any(has_image_fill_deep(node) for _idx, node in nodes),
        "has_gradient": any(has_gradient_fill_deep(node) for _idx, node in nodes),
        "has_star": any(star_count(node) > 0 for _idx, node in nodes),
        "discount_text": any(discount_text_present(node) for _idx, node in nodes),
        "descendant_count": sum(count_descendants(node) for _idx, node in nodes),
        "rotation_deg": rotation,
        "is_rotated": abs(rotation) > 0.01,
        "area_ratio": area_ratio(bbox, canvas_w, canvas_h),
    }


def build_record_from_semantic_json(
    json_obj: Any,
    file_id: str | None = None,
    *,
    include_raw_json: bool = False,
) -> dict[str, Any]:
    root = _root_from_json(json_obj)
    canvas_w, canvas_h = _canvas_size(root)
    children = root.get("children")
    if not isinstance(children, list):
        raise ValueError("Root must contain children list")

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    singles: list[tuple[str, int, dict[str, Any]]] = []
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            continue
        train_role = train_role_for(_semantic_role(child))
        if train_role in MERGED_TRAIN_ROLES:
            grouped[train_role].append((idx, child))
        else:
            singles.append((train_role, idx, child))

    tokens: list[dict[str, Any]] = []
    for train_role, nodes in grouped.items():
        if nodes:
            tokens.append(
                _token_from_nodes(
                    train_role=train_role,
                    occurrence_index=0,
                    nodes=nodes,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                )
            )

    by_role: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for train_role, idx, child in singles:
        by_role[train_role].append((idx, child))
    for train_role, nodes in by_role.items():
        if train_role in ROLE_OCCURRENCE_BY_AREA:
            nodes.sort(key=lambda item: (-area_ratio(bbox_xywh(item[1]), canvas_w, canvas_h), bbox_xywh(item[1])[1], bbox_xywh(item[1])[0]))
        elif train_role in ROLE_OCCURRENCE_BY_POSITION:
            nodes.sort(key=lambda item: (bbox_xywh(item[1])[1], bbox_xywh(item[1])[0]))
        else:
            nodes.sort(key=lambda item: item[0])
        for occurrence_index, node_tuple in enumerate(nodes):
            tokens.append(
                _token_from_nodes(
                    train_role=train_role,
                    occurrence_index=occurrence_index,
                    nodes=[node_tuple],
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                )
            )

    tokens.sort(key=lambda t: (str(t["train_role"]), int(t["occurrence_index"])))
    record = {
        "id": _stable_id(file_id, root),
        "source_file": file_id,
        "canvas": {
            "width": canvas_w,
            "height": canvas_h,
            "aspect": canvas_w / canvas_h,
            "orientation": orientation(canvas_w, canvas_h),
        },
        "tokens": tokens,
    }
    if include_raw_json:
        record["raw_json"] = json_obj
    return record

