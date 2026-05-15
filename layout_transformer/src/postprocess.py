"""Deterministic postprocess for structural Layout Transformer predictions."""

from __future__ import annotations

import copy
import math
from typing import Any

from .roles import FLOATING_ROLES, TRAIN_ROLES

HEADLINE_CHILD_ROLES = {"headline", "subheadline_delivery_time"}
BRAND_CHILD_ROLES = {
    "brand_name_first_part_1",
    "brand_name_first_part_2",
    "brand_name_second",
    "logo",
    "logo_back",
    "logo_fore",
}
OFFER_CHILD_ROLES = {
    "product_name",
    "price_group",
    "price_value",
    "currency_symbol",
    "old_price",
    "old_price_group",
}
STRUCTURAL_PARENT_ROLES = ["hero_image", "headline_group", "brand_group", "background_shape"]
STYLE_SCALE_FIELDS = ("fontSize", "letterSpacing", "strokeWeight", "cornerRadius")
FONT_SIZE_MIN = 4.0
FONT_SIZE_MAX = 128.0
FLOATING_RULE_ROLES = [
    "age_badge",
    "star_decoration_1",
    "star_decoration_2",
    "background_gradient_1",
    "background_gradient_2",
]


def get_bounds(node: dict[str, Any] | None) -> dict[str, float]:
    bounds = node.get("bounds") if isinstance(node, dict) else None
    if not isinstance(bounds, dict):
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    return {
        "x": _num(bounds.get("x")),
        "y": _num(bounds.get("y")),
        "width": _num(bounds.get("width")),
        "height": _num(bounds.get("height")),
    }


def set_bounds(node: dict[str, Any], x: float, y: float, w: float, h: float) -> None:
    node_bounds = node.setdefault("bounds", {})
    node_bounds["x"] = float(x)
    node_bounds["y"] = float(y)
    node_bounds["width"] = float(w)
    node_bounds["height"] = float(h)


def walk_nodes(root: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []

    def walk(item: Any) -> None:
        if not isinstance(item, dict):
            return
        nodes.append(item)
        for child in item.get("children") or []:
            walk(child)

    walk(root)
    return nodes


def find_by_role(root: Any, role: str) -> dict[str, Any] | None:
    matches = find_all_by_role(root, role)
    if not matches:
        return None
    return max(matches, key=lambda node: bbox_area(get_bounds(node)))


def find_all_by_role(root: Any, role: str) -> list[dict[str, Any]]:
    return [node for node in walk_nodes(root) if node.get("name") == role]


def bbox_area(bounds: dict[str, float]) -> float:
    return max(0.0, bounds.get("width", 0.0)) * max(0.0, bounds.get("height", 0.0))


def bbox_center(bounds: dict[str, float]) -> tuple[float, float]:
    return (
        bounds.get("x", 0.0) + bounds.get("width", 0.0) / 2.0,
        bounds.get("y", 0.0) + bounds.get("height", 0.0) / 2.0,
    )


def clone_json(obj: Any) -> Any:
    return copy.deepcopy(obj)


def transform_subtree_by_parent(
    source_root: dict[str, Any],
    output_root: dict[str, Any],
    parent_role: str,
    child_roles: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, int | list[str]]:
    source_parent = find_by_role(source_root, parent_role)
    output_parent = find_by_role(output_root, parent_role)
    report: dict[str, int | list[str]] = {"transformed": 0, "warnings": []}
    warnings = report["warnings"]
    if not isinstance(warnings, list):
        return report
    if source_parent is None or output_parent is None:
        warnings.append(f"missing parent role {parent_role}")
        return report

    source_parent_bounds = get_bounds(source_parent)
    target_parent_bounds = get_bounds(output_parent)
    if source_parent_bounds["width"] <= 0 or source_parent_bounds["height"] <= 0:
        warnings.append(f"bad source parent bounds for {parent_role}")
        return report
    if target_parent_bounds["width"] <= 0 or target_parent_bounds["height"] <= 0:
        warnings.append(f"bad target parent bounds for {parent_role}")
        return report

    sx = target_parent_bounds["width"] / source_parent_bounds["width"]
    sy = target_parent_bounds["height"] / source_parent_bounds["height"]
    style_scale = max(0.01, min(abs(sx), abs(sy)))
    role_filter = set(child_roles) if child_roles is not None else None
    source_entries = _descendant_entries(source_parent)
    output_entries = _descendant_entries(output_parent)

    output_by_id = {
        str(node.get("id")): node
        for node, _path in output_entries
        if node.get("id") is not None
    }
    output_by_path = {_path: node for node, _path in output_entries}
    output_by_name: dict[str, list[dict[str, Any]]] = {}
    for node, _path in output_entries:
        output_by_name.setdefault(str(node.get("name") or ""), []).append(node)

    transformed = 0
    active_paths = _active_role_paths(source_entries, role_filter)
    for source_node, path in source_entries:
        if not path:
            continue
        if role_filter is not None and not _path_is_active(path, active_paths):
            continue
        output_node = _match_output_node(source_node, path, output_by_id, output_by_path, output_by_name)
        if output_node is None:
            warnings.append(f"no output match for {parent_role}/{source_node.get('name') or path}")
            continue
        source_bounds = get_bounds(source_node)
        if source_bounds["width"] <= 0 or source_bounds["height"] <= 0:
            warnings.append(f"bad source child bounds for {parent_role}/{source_node.get('name') or path}")
            continue
        new_x = target_parent_bounds["x"] + (source_bounds["x"] - source_parent_bounds["x"]) * sx
        new_y = target_parent_bounds["y"] + (source_bounds["y"] - source_parent_bounds["y"]) * sy
        new_w = source_bounds["width"] * sx
        new_h = source_bounds["height"] * sy
        set_bounds(output_node, new_x, new_y, new_w, new_h)
        _scale_style_fields(output_node, style_scale)
        transformed += 1

    _fit_active_subtrees_inside_parent(output_by_path, active_paths, target_parent_bounds, warnings)
    report["transformed"] = transformed
    return report


def place_floating_by_anchor(
    source_root: dict[str, Any],
    output_root: dict[str, Any],
    role: str,
    target_w: float,
    target_h: float,
) -> dict[str, Any]:
    if role == "age_badge":
        return _place_age_badge(source_root, output_root, target_w, target_h)
    if role in {"star_decoration_1", "star_decoration_2"}:
        return _place_star(source_root, output_root, role, target_w, target_h)
    if role in {"background_gradient_1", "background_gradient_2"}:
        return _place_gradient(source_root, output_root, role, target_w, target_h)
    return {"role": role, "placed": False, "warning": f"unsupported floating role {role}"}


def postprocess_layout(
    source_json: dict[str, Any],
    output_json: dict[str, Any],
    target_w: float,
    target_h: float,
    return_report: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    report: dict[str, Any] = {
        "transformed_children_count": 0,
        "floating_roles_placed": [],
        "text_alignment": None,
        "text_alignment_applied": 0,
        "headline_children_aligned": 0,
        "font_size_fitted": 0,
        "warnings": [],
    }

    for parent_role, child_roles in (
        ("headline_group", HEADLINE_CHILD_ROLES),
        ("brand_group", BRAND_CHILD_ROLES),
    ):
        subreport = transform_subtree_by_parent(source_json, output_json, parent_role, child_roles)
        report["transformed_children_count"] += int(subreport.get("transformed") or 0)
        report["warnings"].extend(subreport.get("warnings") or [])

    font_report = apply_text_font_size_scaling(source_json, output_json)
    report["font_size_fitted"] += int(font_report.get("scaled") or 0)
    report["warnings"].extend(font_report.get("warnings") or [])

    if find_by_role(source_json, "offer_group") is not None and find_by_role(output_json, "offer_group") is not None:
        subreport = transform_subtree_by_parent(source_json, output_json, "offer_group", OFFER_CHILD_ROLES)
        report["transformed_children_count"] += int(subreport.get("transformed") or 0)
        report["warnings"].extend(subreport.get("warnings") or [])

    from .postprocess_solver import apply_orientation_text_layout

    alignment_report = apply_orientation_text_layout(source_json, output_json, target_w, target_h)
    report["text_alignment"] = alignment_report.get("text_alignment")
    report["text_alignment_applied"] = alignment_report.get("text_alignment_applied", 0)
    report["headline_children_aligned"] = alignment_report.get("headline_children_aligned", 0)
    report["font_size_fitted"] += alignment_report.get("font_size_fitted", 0)
    report["warnings"].extend(alignment_report.get("warnings") or [])

    for role in FLOATING_RULE_ROLES:
        result = place_floating_by_anchor(source_json, output_json, role, target_w, target_h)
        if result.get("placed"):
            report["floating_roles_placed"].append(role)
        if result.get("warning"):
            report["warnings"].append(result["warning"])

    validation_warnings = validate_postprocess_bounds(output_json)
    report["warnings"].extend(validation_warnings)
    if validation_warnings:
        raise ValueError(f"postprocess validation failed: {validation_warnings[:20]}")

    if return_report:
        return output_json, report
    return output_json


def validate_postprocess_bounds(root: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for node in walk_nodes(root):
        bounds = node.get("bounds")
        if not isinstance(bounds, dict):
            continue
        name = str(node.get("name") or node.get("id") or "node")
        raw_values: dict[str, float] = {}
        for key in ("x", "y", "width", "height"):
            try:
                value = float(bounds.get(key))
            except (TypeError, ValueError):
                warnings.append(f"{name}.{key} is not numeric")
                value = 0.0
            if not math.isfinite(value):
                warnings.append(f"{name}.{key} is not finite")
            raw_values[key] = value
        if raw_values["width"] <= 0 or raw_values["height"] <= 0:
            warnings.append(f"{name} has non-positive size {raw_values['width']}x{raw_values['height']}")
    return warnings


def _descendant_entries(parent: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    entries: list[tuple[dict[str, Any], str]] = []

    def walk(node: dict[str, Any], path: str) -> None:
        entries.append((node, path))
        for index, child in enumerate(node.get("children") or []):
            if isinstance(child, dict):
                child_path = f"{path}/{index}" if path else str(index)
                walk(child, child_path)

    walk(parent, "")
    return entries


def _active_role_paths(
    entries: list[tuple[dict[str, Any], str]],
    role_filter: set[str] | None,
) -> set[str]:
    if role_filter is None:
        return {path for _node, path in entries if path}
    return {
        path
        for node, path in entries
        if path and str(node.get("name") or "") in role_filter
    }


def _path_is_active(path: str, active_paths: set[str]) -> bool:
    return any(path == active or path.startswith(active + "/") for active in active_paths)


def _match_output_node(
    source_node: dict[str, Any],
    path: str,
    output_by_id: dict[str, dict[str, Any]],
    output_by_path: dict[str, dict[str, Any]],
    output_by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    source_id = source_node.get("id")
    if source_id is not None:
        by_id = output_by_id.get(str(source_id))
        if by_id is not None:
            return by_id
    by_path = output_by_path.get(path)
    if by_path is not None:
        return by_path
    matches = output_by_name.get(str(source_node.get("name") or "")) or []
    return matches[0] if len(matches) == 1 else None


def _scale_style_fields(node: dict[str, Any], scale: float) -> None:
    for field in STYLE_SCALE_FIELDS:
        value = node.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            scaled = float(value) * scale
            node[field] = _clamp_font_size(scaled) if field == "fontSize" else scaled
        style = node.get("style")
        if isinstance(style, dict):
            style_value = style.get(field)
            if isinstance(style_value, (int, float)) and math.isfinite(float(style_value)):
                scaled = float(style_value) * scale
                style[field] = _clamp_font_size(scaled) if field == "fontSize" else scaled


def apply_text_font_size_scaling(
    source_root: dict[str, Any],
    output_root: dict[str, Any],
) -> dict[str, Any]:
    """Scale text font sizes from source using the role's target/source bbox scale."""
    report: dict[str, Any] = {"scaled": 0, "warnings": []}
    warnings = report["warnings"]
    if not isinstance(warnings, list):
        return report

    parent_specs = (
        ("headline", "headline_group"),
        ("subheadline_delivery_time", "headline_group"),
    )
    for text_role, parent_role in parent_specs:
        source_text = find_by_role(source_root, text_role)
        output_text = find_by_role(output_root, text_role)
        source_parent = find_by_role(source_root, parent_role)
        output_parent = find_by_role(output_root, parent_role)
        if source_text is None or output_text is None or source_parent is None or output_parent is None:
            continue
        scale = _bbox_min_scale(get_bounds(source_parent), get_bounds(output_parent))
        if scale is None:
            warnings.append(f"bad font scale bounds for {text_role}/{parent_role}")
            continue
        if _set_scaled_font_size_from_source(source_text, output_text, scale):
            report["scaled"] = int(report["scaled"]) + 1

    source_legal = find_by_role(source_root, "legal_text")
    output_legal = find_by_role(output_root, "legal_text")
    if source_legal is not None and output_legal is not None:
        scale = _bbox_min_scale(get_bounds(source_legal), get_bounds(output_legal))
        if scale is None:
            warnings.append("bad font scale bounds for legal_text")
        elif _set_scaled_font_size_from_source(source_legal, output_legal, scale):
            report["scaled"] = int(report["scaled"]) + 1
    return report


def _bbox_min_scale(source_bounds: dict[str, float], target_bounds: dict[str, float]) -> float | None:
    if source_bounds["width"] <= 0 or source_bounds["height"] <= 0:
        return None
    return max(0.01, min(target_bounds["width"] / source_bounds["width"], target_bounds["height"] / source_bounds["height"]))


def _set_scaled_font_size_from_source(
    source_node: dict[str, Any],
    output_node: dict[str, Any],
    scale: float,
) -> bool:
    source_size = _node_font_size(source_node)
    if source_size is None:
        return False
    size = _clamp_font_size(source_size * scale)
    output_node["fontSize"] = size
    output_node["textAutoResize"] = "NONE"
    style = output_node.get("style")
    if isinstance(style, dict):
        style["fontSize"] = size
        style["textAutoResize"] = "NONE"
    return True


def _node_font_size(node: dict[str, Any]) -> float | None:
    value = node.get("fontSize")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    style = node.get("style")
    if isinstance(style, dict):
        style_value = style.get("fontSize")
        if isinstance(style_value, (int, float)) and math.isfinite(float(style_value)):
            return float(style_value)
    return None


def _clamp_font_size(value: float) -> float:
    return min(FONT_SIZE_MAX, max(FONT_SIZE_MIN, float(value)))


def _fit_active_subtrees_inside_parent(
    output_by_path: dict[str, dict[str, Any]],
    active_paths: set[str],
    parent_bounds: dict[str, float],
    warnings: list[str],
) -> None:
    root_paths = sorted(active_paths, key=lambda path: path.count("/"))
    for path in root_paths:
        if any(path.startswith(other + "/") for other in root_paths if other != path):
            continue
        node = output_by_path.get(path)
        if node is None:
            continue
        bounds = get_bounds(node)
        if bounds["width"] <= 0 or bounds["height"] <= 0:
            continue
        scale = min(
            1.0,
            parent_bounds["width"] / bounds["width"] if bounds["width"] > 0 else 1.0,
            parent_bounds["height"] / bounds["height"] if bounds["height"] > 0 else 1.0,
        )
        if scale < 1.0:
            _scale_subtree_bounds(node, bounds["x"], bounds["y"], scale)
            _scale_style_fields(node, scale)
            warnings.append(f"scaled {node.get('name') or path} by {scale:.4f} to fit parent")
            bounds = get_bounds(node)
        dx = 0.0
        dy = 0.0
        if bounds["x"] < parent_bounds["x"]:
            dx = parent_bounds["x"] - bounds["x"]
        elif bounds["x"] + bounds["width"] > parent_bounds["x"] + parent_bounds["width"]:
            dx = parent_bounds["x"] + parent_bounds["width"] - bounds["x"] - bounds["width"]
        if bounds["y"] < parent_bounds["y"]:
            dy = parent_bounds["y"] - bounds["y"]
        elif bounds["y"] + bounds["height"] > parent_bounds["y"] + parent_bounds["height"]:
            dy = parent_bounds["y"] + parent_bounds["height"] - bounds["y"] - bounds["height"]
        if dx or dy:
            _translate_subtree_bounds(node, dx, dy)


def _scale_subtree_bounds(node: dict[str, Any], origin_x: float, origin_y: float, scale: float) -> None:
    for item in walk_nodes(node):
        bounds = get_bounds(item)
        if bounds["width"] <= 0 or bounds["height"] <= 0:
            continue
        set_bounds(
            item,
            origin_x + (bounds["x"] - origin_x) * scale,
            origin_y + (bounds["y"] - origin_y) * scale,
            bounds["width"] * scale,
            bounds["height"] * scale,
        )
        _scale_style_fields(item, scale)


def _translate_subtree_bounds(node: dict[str, Any], dx: float, dy: float) -> None:
    for item in walk_nodes(node):
        bounds = get_bounds(item)
        set_bounds(item, bounds["x"] + dx, bounds["y"] + dy, bounds["width"], bounds["height"])


def _place_age_badge(
    source_root: dict[str, Any],
    output_root: dict[str, Any],
    target_w: float,
    target_h: float,
) -> dict[str, Any]:
    source_node = find_by_role(source_root, "age_badge")
    output_node = find_by_role(output_root, "age_badge")
    if source_node is None or output_node is None:
        return {"role": "age_badge", "placed": False, "warning": "missing age_badge"}
    source_canvas = get_bounds(source_root)
    source_bounds = get_bounds(source_node)
    source_min = max(1.0, min(source_canvas["width"], source_canvas["height"]))
    target_min = max(1.0, min(target_w, target_h))
    width = max(1.0, source_bounds["width"] / source_min * target_min)
    height = max(1.0, source_bounds["height"] / source_min * target_min)
    nearest = _nearest_corner(source_bounds, source_canvas["width"], source_canvas["height"])
    x, y = _corner_position(source_bounds, source_canvas, nearest, target_w, target_h, width, height, target_min)
    candidate = _clamp_candidate_to_canvas({"x": x, "y": y, "width": width, "height": height}, target_w, target_h)
    blockers = [find_by_role(output_root, role) for role in ("legal_text", "headline_group", "brand_group")]
    blockers = [node for node in blockers if node is not None]
    chosen = candidate
    anchor = nearest

    if _badge_has_bad_overlap(candidate, blockers):
        alternatives: list[tuple[str, dict[str, float]]] = []
        legal = find_by_role(output_root, "legal_text")
        if legal is not None:
            legal_bounds = get_bounds(legal)
            gap = max(4.0, target_min * 0.012)
            alternatives.append(
                (
                    "above_legal_right",
                    _clamp_candidate_to_canvas(
                        {
                            "x": candidate["x"],
                            "y": legal_bounds["y"] - height - gap,
                            "width": width,
                            "height": height,
                        },
                        target_w,
                        target_h,
                    ),
                )
            )

        background = find_by_role(output_root, "background_shape")
        if background is not None:
            bg = get_bounds(background)
            margin = max(4.0, target_min * 0.018)
            alternatives.append(
                (
                    "background_shape_top_right",
                    _clamp_candidate_to_canvas(
                        {
                            "x": bg["x"] + bg["width"] - width - margin,
                            "y": bg["y"] + margin,
                            "width": width,
                            "height": height,
                        },
                        target_w,
                        target_h,
                    ),
                )
            )

        for corner in ("top_right", "bottom_right", "top_left", "bottom_left"):
            cx, cy = _corner_position(source_bounds, source_canvas, corner, target_w, target_h, width, height, target_min)
            alternatives.append(
                (
                    f"safest_{corner}",
                    _clamp_candidate_to_canvas(
                        {"x": cx, "y": cy, "width": width, "height": height},
                        target_w,
                        target_h,
                    ),
                )
            )

        for candidate_anchor, alternative in alternatives:
            if not _badge_has_bad_overlap(alternative, blockers):
                anchor, chosen = candidate_anchor, alternative
                break
        else:
            anchor, chosen = min(alternatives, key=lambda item: _candidate_overlap_score(item[1], blockers))

    set_bounds(output_node, chosen["x"], chosen["y"], chosen["width"], chosen["height"])
    return {"role": "age_badge", "placed": True, "anchor": anchor}


def _place_star(
    source_root: dict[str, Any],
    output_root: dict[str, Any],
    role: str,
    target_w: float,
    target_h: float,
) -> dict[str, Any]:
    source_node = find_by_role(source_root, role)
    output_node = find_by_role(output_root, role)
    if source_node is None or output_node is None:
        return {"role": role, "placed": False, "warning": f"missing {role}"}
    source_canvas = get_bounds(source_root)
    source_bounds = get_bounds(source_node)
    if source_canvas["width"] <= 0 or source_canvas["height"] <= 0:
        return {"role": role, "placed": False, "warning": f"bad source canvas bounds for {role}"}
    width = max(1.0, source_bounds["width"] / source_canvas["width"] * target_w)
    height = max(1.0, source_bounds["height"] / source_canvas["width"] * target_w)
    x = (source_bounds["x"] - source_canvas["x"]) / source_canvas["width"] * target_w
    y = (source_bounds["y"] - source_canvas["y"]) / source_canvas["height"] * target_h
    set_bounds(output_node, x, y, width, height)
    return {"role": role, "placed": True, "anchor": "canvas_normalized"}


def _place_gradient(
    source_root: dict[str, Any],
    output_root: dict[str, Any],
    role: str,
    target_w: float,
    target_h: float,
) -> dict[str, Any]:
    source_node = find_by_role(source_root, role)
    output_node = find_by_role(output_root, role)
    if source_node is None or output_node is None:
        return {"role": role, "placed": False, "warning": f"missing {role}"}
    source_anchor_node = find_by_role(source_root, "background_shape")
    output_anchor_node = find_by_role(output_root, "background_shape")
    source_anchor = get_bounds(source_anchor_node) if source_anchor_node else get_bounds(source_root)
    target_anchor = get_bounds(output_anchor_node) if output_anchor_node else {"x": 0.0, "y": 0.0, "width": target_w, "height": target_h}
    source_bounds = get_bounds(source_node)
    if source_anchor["width"] <= 0 or source_anchor["height"] <= 0:
        return {"role": role, "placed": False, "warning": f"bad gradient anchor for {role}"}

    sx = target_anchor["width"] / source_anchor["width"]
    sy = target_anchor["height"] / source_anchor["height"]
    min_dim = max(1.0, min(target_w, target_h))
    width = max(min_dim * 0.08, source_bounds["width"] * sx)
    height = max(min_dim * 0.08, source_bounds["height"] * sy)
    x = target_anchor["x"] + (source_bounds["x"] - source_anchor["x"]) * sx
    y = target_anchor["y"] + (source_bounds["y"] - source_anchor["y"]) * sy
    set_bounds(output_node, x, y, width, height)
    return {"role": role, "placed": True, "anchor": "background_shape" if source_anchor_node else "canvas"}


def _nearest_source_parent(
    source_root: dict[str, Any],
    bounds: dict[str, float],
) -> tuple[str, dict[str, Any]] | None:
    cx, cy = bbox_center(bounds)
    best: tuple[float, str, dict[str, Any]] | None = None
    for role in STRUCTURAL_PARENT_ROLES:
        node = find_by_role(source_root, role)
        if node is None:
            continue
        pb = get_bounds(node)
        pcx, pcy = bbox_center(pb)
        distance = (cx - pcx) ** 2 + (cy - pcy) ** 2
        if _contains_center(pb, cx, cy):
            distance *= 0.1
        if best is None or distance < best[0]:
            best = (distance, role, node)
    return None if best is None else (best[1], best[2])


def _nearest_corner(bounds: dict[str, float], canvas_w: float, canvas_h: float) -> str:
    cx, cy = bbox_center(bounds)
    horizontal = "left" if cx <= canvas_w / 2.0 else "right"
    vertical = "top" if cy <= canvas_h / 2.0 else "bottom"
    return f"{vertical}_{horizontal}"


def _corner_position(
    source_bounds: dict[str, float],
    source_canvas: dict[str, float],
    corner: str,
    target_w: float,
    target_h: float,
    width: float,
    height: float,
    target_min: float,
) -> tuple[float, float]:
    source_min = max(1.0, min(source_canvas["width"], source_canvas["height"]))
    left_margin = (source_bounds["x"] - source_canvas["x"]) / source_min * target_min
    right_margin = (source_canvas["x"] + source_canvas["width"] - source_bounds["x"] - source_bounds["width"]) / source_min * target_min
    top_margin = (source_bounds["y"] - source_canvas["y"]) / source_min * target_min
    bottom_margin = (source_canvas["y"] + source_canvas["height"] - source_bounds["y"] - source_bounds["height"]) / source_min * target_min
    x = left_margin if corner.endswith("left") else target_w - right_margin - width
    y = top_margin if corner.startswith("top") else target_h - bottom_margin - height
    return x, y


def _clamp_candidate_to_canvas(
    candidate: dict[str, float],
    target_w: float,
    target_h: float,
) -> dict[str, float]:
    width = max(1.0, candidate["width"])
    height = max(1.0, candidate["height"])
    return {
        "x": min(max(0.0, candidate["x"]), max(0.0, target_w - width)),
        "y": min(max(0.0, candidate["y"]), max(0.0, target_h - height)),
        "width": width,
        "height": height,
    }


def _badge_has_bad_overlap(candidate: dict[str, float], blockers: list[dict[str, Any]]) -> bool:
    area = max(1.0, bbox_area(candidate))
    return any(_overlap_area(candidate, get_bounds(blocker)) > 0.2 * area for blocker in blockers)


def _candidate_overlap_score(candidate: dict[str, float], blockers: list[dict[str, Any]]) -> tuple[float, float]:
    overlap = sum(_overlap_area(candidate, get_bounds(blocker)) for blocker in blockers)
    cx, cy = bbox_center(candidate)
    # Prefer the right side for the small legal badge when overlap ties.
    right_bias = -cx + cy * 0.001
    return (overlap, right_bias)


def _contains_center(bounds: dict[str, float], cx: float, cy: float) -> bool:
    return bounds["x"] <= cx <= bounds["x"] + bounds["width"] and bounds["y"] <= cy <= bounds["y"] + bounds["height"]


def _overlap_area(a: dict[str, float], b: dict[str, float]) -> float:
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["width"], b["x"] + b["width"])
    y2 = min(a["y"] + a["height"], b["y"] + b["height"])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0
