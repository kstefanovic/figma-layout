"""Build canonical top-level layout records from semantic Figma JSON."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any

from .geometry import (
    area_ratio,
    bbox_union,
    bbox_xywh,
    center_size_from_bbox,
    center_size_from_visual_bounds,
    clip_bbox_to_canvas,
    coverage_ratio,
    get_layout_bounds,
    get_rotation_deg,
    get_visual_bounds,
    normalize_center_size,
    orientation,
    safe_float,
    bleed_flags,
)
from .json_features import (
    all_text,
    count_descendants,
    discount_text_present,
    has_gradient_fill_deep,
    has_image_fill_deep,
    has_text_deep,
    normalized_node_type,
    star_count,
)
from .roles import (
    CORE_TRAIN_ROLES,
    MERGED_TRAIN_ROLES,
    ROLE_OCCURRENCE_BY_AREA,
    ROLE_OCCURRENCE_BY_POSITION,
    core_role_for,
    token_id_for,
    train_role_for,
)


PRICE_SIGNAL_RE = re.compile(r"(₽|руб|%|\b\d[\d\s]{1,8}\b)", re.IGNORECASE)


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
    clipped = clip_bbox_to_canvas(bbox, canvas_w, canvas_h)
    clipped_norm = (
        clipped[0] / canvas_w,
        clipped[1] / canvas_h,
        clipped[2] / canvas_w,
        clipped[3] / canvas_h,
    )
    rots = [get_rotation_deg(node) for _idx, node in nodes]
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


def _visual_bbox_tuple(node: dict[str, Any]) -> tuple[float, float, float, float]:
    b = get_visual_bounds(node)
    return safe_float(b.get("x")), safe_float(b.get("y")), safe_float(b.get("width")), safe_float(b.get("height"))


def _layout_bbox_tuple(node: dict[str, Any]) -> tuple[float, float, float, float]:
    b = get_layout_bounds(node)
    return safe_float(b.get("x")), safe_float(b.get("y")), safe_float(b.get("width")), safe_float(b.get("height"))


def _price_signal(node: dict[str, Any]) -> bool:
    text = all_text(node)
    if not text:
        return False
    if PRICE_SIGNAL_RE.search(text):
        return True
    return any(str(fill.get("type") or "").upper() == "SOLID" for fill in node.get("fills") or [] if isinstance(fill, dict)) and discount_text_present(node)


def _node_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay, aw, ah = _visual_bbox_tuple(a)
    bx, by, bw, bh = _visual_bbox_tuple(b)
    acx = ax + aw / 2.0
    acy = ay + ah / 2.0
    bcx = bx + bw / 2.0
    bcy = by + bh / 2.0
    dx = acx - bcx
    dy = acy - bcy
    return math.hypot(dx, dy)


def _boxes_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ax, ay, aw, ah = _visual_bbox_tuple(a)
    bx, by, bw, bh = _visual_bbox_tuple(b)
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _cluster_text_main_nodes(nodes: list[tuple[int, dict[str, Any]]], canvas_w: float, canvas_h: float) -> list[list[tuple[int, dict[str, Any]]]]:
    if len(nodes) <= 1:
        return [nodes] if nodes else []
    threshold = 0.12 * max(canvas_w, canvas_h)
    clusters: list[list[tuple[int, dict[str, Any]]]] = []
    used: set[int] = set()
    for i, (idx, node) in enumerate(nodes):
        if i in used:
            continue
        cluster = [(idx, node)]
        used.add(i)
        changed = True
        while changed:
            changed = False
            for j, (other_idx, other_node) in enumerate(nodes):
                if j in used:
                    continue
                close = any(
                    _boxes_overlap(existing_node, other_node)
                    or _node_distance(existing_node, other_node) < threshold
                    for _existing_idx, existing_node in cluster
                )
                if not close:
                    continue
                # Merge price + nearby headline conservatively into one offer block.
                if _price_signal(other_node) or any(_price_signal(existing_node) for _existing_idx, existing_node in cluster):
                    cluster.append((other_idx, other_node))
                    used.add(j)
                    changed = True
                    continue
                cluster.append((other_idx, other_node))
                used.add(j)
                changed = True
        clusters.append(cluster)
    return clusters


def _build_core_token(
    *,
    train_role: str,
    occurrence_index: int,
    nodes: list[tuple[int, dict[str, Any]]],
    canvas_w: float,
    canvas_h: float,
    background_cluster: bool = False,
) -> dict[str, Any]:
    visual_boxes = [_visual_bbox_tuple(node) for _idx, node in nodes]
    layout_boxes = [_layout_bbox_tuple(node) for _idx, node in nodes]
    visual_union = bbox_union(visual_boxes)
    layout_union = bbox_union(layout_boxes)
    source_paths = [str(node.get("path", idx)) for idx, node in nodes]
    semantic_roles = [_semantic_role(node) for _idx, node in nodes]
    rotation = get_rotation_deg(nodes[0][1]) if nodes else 0.0
    clipped_visual = clip_bbox_to_canvas(visual_union, canvas_w, canvas_h)
    bbox_visual = clipped_visual if background_cluster and clipped_visual[2] > 0 and clipped_visual[3] > 0 else visual_union
    cx, cy, w, h = center_size_from_bbox(*bbox_visual)
    cxn, cyn, wn, hn = normalize_center_size(cx, cy, w, h, canvas_w, canvas_h)
    bleed_left, bleed_right, bleed_top, bleed_bottom = bleed_flags(visual_union, canvas_w, canvas_h)
    token = {
        "token_id": token_id_for(train_role, occurrence_index),
        "train_role": train_role,
        "occurrence_index": occurrence_index,
        "source_paths": source_paths,
        "source_indices": [idx for idx, _node in nodes],
        "semantic_roles": semantic_roles,
        "bbox_visual": [bbox_visual[0], bbox_visual[1], bbox_visual[2], bbox_visual[3]],
        "center_size_norm": [cxn, cyn, wn, hn],
        "layout_bbox": [layout_union[0], layout_union[1], layout_union[2], layout_union[3]],
        "rotation_deg": rotation,
        "is_rotated": any(abs(get_rotation_deg(node)) > 0.01 for _idx, node in nodes),
        "has_image": any(has_image_fill_deep(node) for _idx, node in nodes),
        "has_text": any(has_text_deep(node) for _idx, node in nodes),
        "instance_count": len(nodes),
        "token_type": "cluster" if len(nodes) > 1 else normalized_node_type(nodes[0][1]),
        "area_ratio": area_ratio(bbox_visual, canvas_w, canvas_h),
        "coverage_ratio": coverage_ratio(visual_union, canvas_w, canvas_h),
        "bleed_left": bleed_left,
        "bleed_right": bleed_right,
        "bleed_top": bleed_top,
        "bleed_bottom": bleed_bottom,
        "source_center_size_norm": list(normalize_center_size(*center_size_from_bbox(*visual_union), canvas_w, canvas_h)),
    }
    if background_cluster:
        token["bleed_bbox_visual"] = [visual_union[0], visual_union[1], visual_union[2], visual_union[3]]
        token["coverage_bbox_visual"] = [clipped_visual[0], clipped_visual[1], clipped_visual[2], clipped_visual[3]]
    return token


def build_core_record_from_semantic_json(
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

    background_nodes: list[tuple[int, dict[str, Any]]] = []
    text_nodes: list[tuple[int, dict[str, Any]]] = []
    by_role: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    excluded_roles: list[str] = []

    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            continue
        semantic_role = _semantic_role(child)
        core_role = core_role_for(semantic_role)
        if core_role is None:
            excluded_roles.append(semantic_role)
            continue
        if core_role == "background_cluster":
            background_nodes.append((idx, child))
            continue
        if core_role == "text_main_group":
            text_nodes.append((idx, child))
            continue
        by_role[core_role].append((idx, child))

    tokens: list[dict[str, Any]] = []
    if background_nodes:
        tokens.append(
            _build_core_token(
                train_role="background_cluster",
                occurrence_index=0,
                nodes=background_nodes,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                background_cluster=True,
            )
        )

    if text_nodes:
        text_clusters = _cluster_text_main_nodes(text_nodes, canvas_w, canvas_h)
        for occurrence_index, cluster in enumerate(sorted(text_clusters, key=lambda items: min(_visual_bbox_tuple(node)[1] for _idx, node in items))):
            tokens.append(
                _build_core_token(
                    train_role="text_main_group",
                    occurrence_index=occurrence_index,
                    nodes=cluster,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                )
            )

    for role in ("hero_group", "brand_group", "legal_group"):
        nodes = by_role.get(role) or []
        nodes.sort(key=lambda item: (_visual_bbox_tuple(item[1])[1], _visual_bbox_tuple(item[1])[0], item[0]))
        for occurrence_index, node_tuple in enumerate(nodes):
            tokens.append(
                _build_core_token(
                    train_role=role,
                    occurrence_index=occurrence_index,
                    nodes=[node_tuple],
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                )
            )

    tokens = [token for token in tokens if str(token.get("train_role")) in set(CORE_TRAIN_ROLES)]
    tokens.sort(key=lambda t: (CORE_TRAIN_ROLES.index(str(t["train_role"])), int(t["occurrence_index"])))
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
        "excluded_roles": excluded_roles,
    }
    if include_raw_json:
        record["raw_json"] = json_obj
    return record
