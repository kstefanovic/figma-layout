"""Prototype-guided postprocess for Layout Transformer predictions."""

from __future__ import annotations

import math
from typing import Any

from .postprocess import find_by_role, get_bounds, set_bounds, validate_postprocess_bounds
from .prototype_index import CHILD_PARENT, CHILD_ROLES, FLOATING_PROTO_ROLES, TRAIN_ROLES

TEXT_ROLE_CLAMPS = {
    "headline": (28.0, 80.0),
    "subheadline_delivery_time": (10.0, 36.0),
    "legal_text": (5.0, 18.0),
    "age_badge": (14.0, 36.0),
}
PORTRAIT_640_TEXT_CLAMPS = {
    "headline": (30.0, 34.0),
    "subheadline_delivery_time": (13.0, 16.0),
    "legal_text": (6.0, 8.0),
    "age_badge": (22.0, 26.0),
}
NON_CLIPPING_FRAME_ROLES = {"headline_group", "brand_group", "offer_group"}


def apply_prototype_postprocess(
    *,
    source_json: dict[str, Any],
    output_json: dict[str, Any],
    target_w: float,
    target_h: float,
    prototype: dict[str, Any] | None,
    prototype_match: dict[str, Any] | None = None,
    return_report: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Keep transformer parent boxes, then snap children/floating roles to target prototype style."""
    strict_mode = _is_strict_prototype_match(prototype_match)
    report: dict[str, Any] = {
        "prototype_id": prototype.get("prototype_id") if isinstance(prototype, dict) else None,
        "postprocess_mode": "strict_prototype" if strict_mode else "transformer_plus_prototype",
        "prototype_match_score": prototype_match.get("score") if isinstance(prototype_match, dict) else None,
        "prototype_match": _public_match_report(prototype_match),
        "prototype_children_applied": 0,
        "prototype_floating_applied": [],
        "strict_roles_applied": [],
        "font_size_fitted": 0,
        "warnings": [],
    }
    if not isinstance(prototype, dict):
        report["warnings"].append("no prototype selected")
        if return_report:
            return output_json, report
        return output_json

    _apply_clipping_rules(output_json)
    if strict_mode:
        _apply_strict_prototype_bboxes(output_json, target_w, target_h, prototype, report)
    else:
        _apply_child_relative_bboxes(output_json, prototype, report)
        _apply_floating_bboxes(output_json, target_w, target_h, prototype, report)
    _apply_text_styles(source_json, output_json, prototype, target_w, target_h, strict_mode, report)

    warnings = validate_postprocess_bounds(output_json)
    report["warnings"].extend(warnings)
    if warnings:
        raise ValueError(f"prototype postprocess validation failed: {warnings[:20]}")

    if return_report:
        return output_json, report
    return output_json


def _is_strict_prototype_match(match: dict[str, Any] | None) -> bool:
    if not isinstance(match, dict):
        return False
    aspect_diff = _num(match.get("aspect_diff"), 999.0)
    width_diff = _num(match.get("width_diff_ratio"), 999.0)
    height_diff = _num(match.get("height_diff_ratio"), 999.0)
    exact_size = bool(match.get("exact_size"))
    return aspect_diff < 0.05 and (exact_size or (width_diff < 0.10 and height_diff < 0.10))


def _public_match_report(match: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(match, dict):
        return None
    return {
        "prototype_id": match.get("prototype_id"),
        "score": match.get("score"),
        "aspect_diff": match.get("aspect_diff"),
        "width_diff_ratio": match.get("width_diff_ratio"),
        "height_diff_ratio": match.get("height_diff_ratio"),
        "exact_size": match.get("exact_size"),
    }


def _apply_clipping_rules(root: dict[str, Any]) -> None:
    root["clipsContent"] = True

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        name = str(node.get("name") or "")
        if name in NON_CLIPPING_FRAME_ROLES:
            node["clipsContent"] = False
        for child in node.get("children") or []:
            walk(child)

    walk(root)


def _apply_strict_prototype_bboxes(
    output_json: dict[str, Any],
    target_w: float,
    target_h: float,
    prototype: dict[str, Any],
    report: dict[str, Any],
) -> None:
    role_bboxes = prototype.get("role_bboxes") if isinstance(prototype.get("role_bboxes"), dict) else {}
    if not role_bboxes:
        report["warnings"].append("strict prototype selected but prototype has no role_bboxes")
        return
    for role, norm in role_bboxes.items():
        if not isinstance(role, str) or not isinstance(norm, dict):
            continue
        node = find_by_role(output_json, role)
        if node is None:
            continue
        set_bounds(
            node,
            _num(norm.get("x"), 0.0) * target_w,
            _num(norm.get("y"), 0.0) * target_h,
            max(0.01, _num(norm.get("width"), 0.0) * target_w),
            max(0.01, _num(norm.get("height"), 0.0) * target_h),
        )
        report["strict_roles_applied"].append(role)


def _apply_child_relative_bboxes(
    output_json: dict[str, Any],
    prototype: dict[str, Any],
    report: dict[str, Any],
) -> None:
    rels = prototype.get("child_relative_bboxes") if isinstance(prototype.get("child_relative_bboxes"), dict) else {}
    for role in CHILD_ROLES:
        rel = rels.get(role)
        parent_role = CHILD_PARENT[role]
        node = find_by_role(output_json, role)
        parent = find_by_role(output_json, parent_role)
        if node is None or parent is None or not isinstance(rel, dict):
            continue
        pb = get_bounds(parent)
        set_bounds(
            node,
            pb["x"] + _num(rel.get("x"), 0.0) * pb["width"],
            pb["y"] + _num(rel.get("y"), 0.0) * pb["height"],
            max(0.01, _num(rel.get("width"), 0.0) * pb["width"]),
            max(0.01, _num(rel.get("height"), 0.0) * pb["height"]),
        )
        report["prototype_children_applied"] += 1


def _apply_floating_bboxes(
    output_json: dict[str, Any],
    target_w: float,
    target_h: float,
    prototype: dict[str, Any],
    report: dict[str, Any],
) -> None:
    floating = prototype.get("floating_bboxes") if isinstance(prototype.get("floating_bboxes"), dict) else {}
    for role in FLOATING_PROTO_ROLES:
        norm = floating.get(role)
        node = find_by_role(output_json, role)
        if node is None or not isinstance(norm, dict):
            continue
        set_bounds(
            node,
            _num(norm.get("x"), 0.0) * target_w,
            _num(norm.get("y"), 0.0) * target_h,
            max(0.01, _num(norm.get("width"), 0.0) * target_w),
            max(0.01, _num(norm.get("height"), 0.0) * target_h),
        )
        report["prototype_floating_applied"].append(role)


def _apply_text_styles(
    source_json: dict[str, Any],
    output_json: dict[str, Any],
    prototype: dict[str, Any],
    target_w: float,
    target_h: float,
    strict_mode: bool,
    report: dict[str, Any],
) -> None:
    proto_styles = prototype.get("text_styles") if isinstance(prototype.get("text_styles"), dict) else {}
    orientation = "portrait" if target_w <= target_h else "landscape"
    align = "CENTER" if orientation == "portrait" else "LEFT"
    headline_node = find_by_role(output_json, "headline")
    headline_size = _headline_anchor_font_size(headline_node, target_w, target_h)
    for role in ("headline", "subheadline_delivery_time", "legal_text", "age_badge"):
        node = find_by_role(output_json, role)
        if node is None:
            continue
        source_node = find_by_role(source_json, role)
        proto_style = proto_styles.get(role) if isinstance(proto_styles.get(role), dict) else {}
        _copy_text_content_and_font(source_node, node, proto_style)
        font_size = _role_based_font_size(role, node, headline_size, target_w, target_h)
        if font_size is None:
            font_size = _prototype_scaled_font_size(role, source_node, node, prototype, proto_style)
        if font_size is not None:
            node["fontSize"] = font_size
            report["font_size_fitted"] += 1
        node["textAutoResize"] = "NONE"
        if role != "age_badge":
            node["textAlignHorizontal"] = proto_style.get("textAlignHorizontal") or align
        else:
            node["textAlignHorizontal"] = proto_style.get("textAlignHorizontal") or node.get("textAlignHorizontal") or "CENTER"
            node["textAlignVertical"] = proto_style.get("textAlignVertical") or node.get("textAlignVertical") or "CENTER"
        if not strict_mode:
            _apply_alignment(role, node, output_json, align, target_w)
    if orientation == "portrait" and not strict_mode:
        _enforce_portrait_text_spacing(output_json, target_w, target_h, report)


def _copy_text_content_and_font(
    source_node: dict[str, Any] | None,
    target_node: dict[str, Any],
    proto_style: dict[str, Any],
) -> None:
    if isinstance(source_node, dict):
        if "characters" in source_node:
            target_node["characters"] = source_node.get("characters", "")
        font_name = source_node.get("fontName")
        if isinstance(font_name, dict) and font_name.get("family") and font_name.get("style"):
            target_node["fontName"] = {
                "family": str(font_name["family"]),
                "style": str(font_name["style"]),
            }
        for key in (
            "fills",
            "lineHeight",
            "letterSpacing",
            "opacity",
            "paragraphSpacing",
            "textCase",
            "textDecoration",
        ):
            if key in source_node:
                target_node[key] = source_node[key]
    for key in ("textAlignVertical",):
        value = proto_style.get(key)
        if isinstance(value, str) and value:
            target_node[key] = value
        elif isinstance(source_node, dict) and isinstance(source_node.get(key), str):
            target_node[key] = source_node[key]


def _headline_anchor_font_size(headline_node: dict[str, Any] | None, target_w: float, target_h: float) -> float | None:
    if headline_node is None:
        return None
    height = get_bounds(headline_node)["height"]
    if height <= 0:
        return None
    return _clamp_role_font("headline", height * 0.38, target_w, target_h)


def _role_based_font_size(
    role: str,
    target_node: dict[str, Any],
    headline_size: float | None,
    target_w: float,
    target_h: float,
) -> float | None:
    if role == "headline":
        return headline_size
    if role == "subheadline_delivery_time" and headline_size is not None:
        return _clamp_role_font(role, headline_size * 0.42, target_w, target_h)
    if role == "legal_text" and headline_size is not None:
        return _clamp_role_font(role, headline_size * 0.18, target_w, target_h)
    if role == "age_badge":
        height = get_bounds(target_node)["height"]
        if height > 0:
            return _clamp_role_font(role, height * 0.75, target_w, target_h)
    return None


def _clamp_role_font(role: str, size: float, target_w: float | None = None, target_h: float | None = None) -> float:
    clamps = PORTRAIT_640_TEXT_CLAMPS if _is_640_portrait_like(target_w, target_h) else TEXT_ROLE_CLAMPS
    lo, hi = clamps.get(role, (4.0, 128.0))
    return max(lo, min(hi, size))


def _is_640_portrait_like(target_w: float | None, target_h: float | None) -> bool:
    if target_w is None or target_h is None:
        return False
    return target_w <= target_h and abs(target_w - 640.0) <= 80.0 and abs(target_h - 720.0) <= 120.0


def _apply_alignment(
    role: str,
    node: dict[str, Any],
    output_json: dict[str, Any],
    align: str,
    target_w: float,
) -> None:
    if role != "age_badge":
        node["textAlignHorizontal"] = align
    if role == "age_badge":
        node["textAlignHorizontal"] = "CENTER"
        node["textAlignVertical"] = "CENTER"
        return
    if align != "CENTER":
        return
    bounds = get_bounds(node)
    if role in CHILD_PARENT:
        parent = find_by_role(output_json, CHILD_PARENT[role])
        if parent is not None:
            pb = get_bounds(parent)
            bounds["x"] = pb["x"] + (pb["width"] - bounds["width"]) / 2.0
            set_bounds(node, bounds["x"], bounds["y"], bounds["width"], bounds["height"])
            return
    bounds["x"] = (target_w - bounds["width"]) / 2.0
    set_bounds(node, bounds["x"], bounds["y"], bounds["width"], bounds["height"])


def _enforce_portrait_text_spacing(
    output_json: dict[str, Any],
    target_w: float,
    target_h: float,
    report: dict[str, Any],
) -> None:
    headline = find_by_role(output_json, "headline")
    subheadline = find_by_role(output_json, "subheadline_delivery_time")
    brand_group = find_by_role(output_json, "brand_group")
    if headline is None or subheadline is None:
        return

    if brand_group is not None:
        brand = get_bounds(brand_group)
        h = get_bounds(headline)
        min_headline_y = brand["y"] + brand["height"] + target_h * 0.012
        if h["y"] < min_headline_y:
            reduced = _reduce_font_to_role_min(headline, "headline", target_w, target_h)
            if reduced:
                report["font_size_fitted"] += 1
            h = get_bounds(headline)
            if h["y"] < min_headline_y:
                set_bounds(headline, h["x"], min_headline_y, h["width"], h["height"])
                report["warnings"].append("portrait text spacing: moved headline below brand_group")

    h = get_bounds(headline)
    s = get_bounds(subheadline)
    min_sub_y = h["y"] + h["height"] + target_h * 0.008
    if s["y"] < min_sub_y:
        set_bounds(subheadline, s["x"], min_sub_y, s["width"], s["height"])
        report["warnings"].append("portrait text spacing: moved subheadline below headline")

    headline_size = _num(headline.get("fontSize"), 0.0)
    if headline_size > 0:
        subheadline["fontSize"] = _clamp_role_font("subheadline_delivery_time", headline_size * 0.42, target_w, target_h)
        legal = find_by_role(output_json, "legal_text")
        if legal is not None:
            legal["fontSize"] = _clamp_role_font("legal_text", headline_size * 0.18, target_w, target_h)


def _reduce_font_to_role_min(
    node: dict[str, Any],
    role: str,
    target_w: float,
    target_h: float,
) -> bool:
    current = _num(node.get("fontSize"), 0.0)
    lo, _hi = (PORTRAIT_640_TEXT_CLAMPS if _is_640_portrait_like(target_w, target_h) else TEXT_ROLE_CLAMPS).get(role, (4.0, 128.0))
    if current <= lo:
        return False
    node["fontSize"] = lo
    return True


def _prototype_scaled_font_size(
    role: str,
    source_node: dict[str, Any] | None,
    target_node: dict[str, Any],
    prototype: dict[str, Any],
    proto_style: dict[str, Any],
) -> float | None:
    target_height = get_bounds(target_node)["height"]
    proto_font_size = proto_style.get("fontSize")
    proto_height = _prototype_abs_height(role, prototype)
    if isinstance(proto_font_size, (int, float)) and math.isfinite(float(proto_font_size)) and proto_height > 0:
        size = float(proto_font_size) * target_height / proto_height
    else:
        source_bounds = get_bounds(source_node)
        source_size = source_node.get("fontSize") if isinstance(source_node, dict) else None
        if not isinstance(source_size, (int, float)) or not math.isfinite(float(source_size)) or source_bounds["height"] <= 0:
            return None
        size = float(source_size) * target_height / source_bounds["height"]
    lo, hi = TEXT_ROLE_CLAMPS.get(role, (4.0, 128.0))
    return max(lo, min(hi, size))


def _prototype_abs_height(role: str, prototype: dict[str, Any]) -> float:
    canvas = prototype.get("canvas") if isinstance(prototype.get("canvas"), dict) else {}
    canvas_h = _num(canvas.get("height"), 0.0)
    if canvas_h <= 0:
        return 0.0
    if role in CHILD_PARENT:
        rels = prototype.get("child_relative_bboxes") if isinstance(prototype.get("child_relative_bboxes"), dict) else {}
        structs = prototype.get("structural_bboxes") if isinstance(prototype.get("structural_bboxes"), dict) else {}
        rel = rels.get(role)
        parent = structs.get(CHILD_PARENT[role])
        if isinstance(rel, dict) and isinstance(parent, dict):
            return _num(rel.get("height"), 0.0) * _num(parent.get("height"), 0.0) * canvas_h
    if role in FLOATING_PROTO_ROLES:
        floating = prototype.get("floating_bboxes") if isinstance(prototype.get("floating_bboxes"), dict) else {}
        bbox = floating.get(role)
        if isinstance(bbox, dict):
            return _num(bbox.get("height"), 0.0) * canvas_h
    structs = prototype.get("structural_bboxes") if isinstance(prototype.get("structural_bboxes"), dict) else {}
    bbox = structs.get(role)
    if isinstance(bbox, dict):
        return _num(bbox.get("height"), 0.0) * canvas_h
    return 0.0


def _num(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default
