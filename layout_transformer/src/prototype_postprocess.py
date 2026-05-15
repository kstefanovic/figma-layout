"""Prototype-guided postprocess: child-relative text layout + text styles only."""

from __future__ import annotations

import copy
import math
from typing import Any

from .postprocess import find_by_role, get_bounds, set_bounds, validate_postprocess_bounds
from .postprocess_solver import _is_text_node
from .prototype_index import CHILD_PARENT, FLOATING_PROTO_ROLES, inferred_text_font_size_for_role

TEXT_ROLE_CLAMPS = {
    "headline": (28.0, 80.0),
    "subheadline_delivery_time": (10.0, 36.0),
    "legal_text": (5.0, 18.0),
}
PORTRAIT_640_TEXT_CLAMPS = {
    # Match general clamps so geometry-based scaling is not capped at ~34px headline.
    "headline": (28.0, 80.0),
    "subheadline_delivery_time": (10.0, 36.0),
    "legal_text": (5.0, 18.0),
}
NON_CLIPPING_FRAME_ROLES = {"brand_group", "offer_group"}

HEADLINE_GROUP_ALLOWED_DIRECT = frozenset({"headline", "subheadline_delivery_time"})
FORBIDDEN_UNDER_HEADLINE_GROUP = frozenset(
    {
        "brand_name_first_part_1",
        "brand_name_first_part_2",
        "brand_name_second",
        "logo",
        "logo_back",
        "logo_fore",
    }
)
BRAND_GROUP_REQUIRED_DIRECT = frozenset(
    {"brand_name_first_part_1", "brand_name_first_part_2", "brand_name_second", "logo"}
)

TEXT_ROLES_LAYOUT = ("headline", "subheadline_delivery_time")
BRAND_GROUP_LAYOUT_ROLES = (
    "brand_name_first_part_1",
    "brand_name_first_part_2",
    "brand_name_second",
    "logo",
)
LOGO_LAYOUT_ROLES = ("logo_back", "logo_fore")
TEXT_ROLES_STYLE = ("headline", "subheadline_delivery_time", "legal_text", "age_badge")
PORTRAIT_640_EXACT_TEXT_STYLES: dict[str, dict[str, Any]] = {
    "headline": {
        "fontSize": 36.0,
        "textAutoResize": "NONE",
        "textAlignHorizontal": "CENTER",
        "textAlignVertical": "CENTER",
    },
    "subheadline_delivery_time": {
        "fontSize": 14.0,
        "textAutoResize": "NONE",
        "textAlignHorizontal": "CENTER",
        "textAlignVertical": "CENTER",
    },
    "legal_text": {
        "fontSize": 6.0,
        "textAutoResize": "NONE",
        "textAlignHorizontal": "CENTER",
        "textAlignVertical": "CENTER",
    },
    "age_badge": {
        "fontSize": 25.0,
        "textAutoResize": "NONE",
        "textAlignHorizontal": "CENTER",
        "textAlignVertical": "CENTER",
    },
}

PROTO_TEXT_STYLE_KEYS = (
    "fontSize",
    "fontName",
    "textAlignHorizontal",
    "textAlignVertical",
    "textAutoResize",
    "lineHeight",
    "letterSpacing",
    "fills",
    "opacity",
)


def _direct_child_role_names(node: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for ch in node.get("children") or []:
        if isinstance(ch, dict):
            nm = ch.get("name")
            if isinstance(nm, str):
                names.add(nm)
    return names


def _collect_descendant_role_names(node: dict[str, Any]) -> set[str]:
    found: set[str] = set()

    def walk(n: Any) -> None:
        if not isinstance(n, dict):
            return
        name = n.get("name")
        if isinstance(name, str):
            found.add(name)
        for ch in n.get("children") or []:
            walk(ch)

    walk(node)
    return found


def validate_text_postprocess_hierarchy(root: dict[str, Any]) -> None:
    """Transformer output tree must keep brand vs headline separation (no reparenting in postprocess)."""
    hg = find_by_role(root, "headline_group")
    if hg is not None:
        for name in _direct_child_role_names(hg):
            if name not in HEADLINE_GROUP_ALLOWED_DIRECT:
                raise ValueError(
                    "layout hierarchy invalid: headline_group has disallowed direct child "
                    f"{name!r} (allowed: headline, subheadline_delivery_time)"
                )
        under = _collect_descendant_role_names(hg)
        viol = under & FORBIDDEN_UNDER_HEADLINE_GROUP
        if viol:
            raise ValueError(
                "layout hierarchy invalid: brand/logo role(s) under headline_group subtree: " f"{sorted(viol)}"
            )

    bg = find_by_role(root, "brand_group")
    if bg is not None:
        direct = _direct_child_role_names(bg)
        missing = BRAND_GROUP_REQUIRED_DIRECT - direct
        if missing:
            raise ValueError("layout hierarchy invalid: brand_group missing direct children " f"{sorted(missing)}")


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
    """Lock structural parent boxes from the transformer; place children via prototype rels + text styles."""
    report: dict[str, Any] = {
        "prototype_id": prototype.get("prototype_id") if isinstance(prototype, dict) else None,
        "postprocess_mode": "prototype_relative_text",
        "prototype_match_score": prototype_match.get("score") if isinstance(prototype_match, dict) else None,
        "prototype_match": _public_match_report(prototype_match),
        "prototype_headline_children_applied": 0,
        "prototype_brand_children_applied": 0,
        "prototype_logo_children_applied": 0,
        "prototype_floating_applied": [],
        "legal_text_bbox_source": None,
        "font_size_fitted": 0,
        "warnings": [],
        "debug": {
            "text_postprocess_mode": "prototype_relative_to_parent",
            "text_roles_applied": list(TEXT_ROLES_STYLE),
        },
    }
    if not isinstance(prototype, dict):
        report["warnings"].append("no prototype selected")
        if return_report:
            return output_json, report
        return output_json

    validate_text_postprocess_hierarchy(output_json)

    _apply_clipping_rules(output_json)
    _apply_child_relative_roles(
        output_json, prototype, "brand_group", BRAND_GROUP_LAYOUT_ROLES, report, "prototype_brand_children_applied"
    )
    _apply_child_relative_roles(
        output_json, prototype, "logo", LOGO_LAYOUT_ROLES, report, "prototype_logo_children_applied"
    )
    _apply_child_relative_roles(
        output_json, prototype, "headline_group", TEXT_ROLES_LAYOUT, report, "prototype_headline_children_applied"
    )
    _apply_floating_bboxes(output_json, target_w, target_h, prototype, report)
    _apply_legal_text_bbox_from_prototype_if_needed(output_json, prototype, report)

    proto_styles = prototype.get("text_styles") if isinstance(prototype.get("text_styles"), dict) else {}
    _apply_prototype_text_styles(
        source_json,
        output_json,
        prototype,
        proto_styles,
        target_w,
        target_h,
        report,
    )

    validate_child_bounds_placements(output_json, target_w, target_h)

    warnings = validate_postprocess_bounds(output_json)
    report["warnings"].extend(warnings)
    if warnings:
        raise ValueError(f"prototype postprocess validation failed: {warnings[:20]}")

    if return_report:
        return output_json, report
    return output_json


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
        elif name == "headline_group":
            node["clipsContent"] = False
        for child in node.get("children") or []:
            walk(child)

    walk(root)


def _apply_child_relative_roles(
    output_json: dict[str, Any],
    prototype: dict[str, Any],
    parent_role: str,
    child_roles: tuple[str, ...],
    report: dict[str, Any],
    report_counter_key: str,
) -> None:
    """Place roles using ``prototype.child_relative_bboxes`` inside predicted ``parent_role`` (absolute canvas)."""
    rels = prototype.get("child_relative_bboxes") if isinstance(prototype.get("child_relative_bboxes"), dict) else {}
    parent = find_by_role(output_json, parent_role)
    if parent is None:
        return
    pb = get_bounds(parent)
    if pb["width"] <= 0 or pb["height"] <= 0:
        return
    for role in child_roles:
        rel = rels.get(role)
        node = find_by_role(output_json, role)
        if node is None or not isinstance(rel, dict):
            continue
        set_bounds(
            node,
            pb["x"] + _num(rel.get("x"), 0.0) * pb["width"],
            pb["y"] + _num(rel.get("y"), 0.0) * pb["height"],
            max(0.01, _num(rel.get("width"), 0.0) * pb["width"]),
            max(0.01, _num(rel.get("height"), 0.0) * pb["height"]),
        )
        report[report_counter_key] = int(report.get(report_counter_key) or 0) + 1


def _apply_floating_bboxes(
    output_json: dict[str, Any],
    target_w: float,
    target_h: float,
    prototype: dict[str, Any],
    report: dict[str, Any],
) -> None:
    """Place floating roles from prototype normalized bboxes on the target canvas."""
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


def _point_inside_rect(px: float, py: float, rect: dict[str, float], margin: float) -> bool:
    return (
        px >= rect["x"] - margin
        and px <= rect["x"] + rect["width"] + margin
        and py >= rect["y"] - margin
        and py <= rect["y"] + rect["height"] + margin
    )


def _rect_inside_rect(inner: dict[str, float], outer: dict[str, float], margin: float) -> bool:
    """True if ``inner`` is fully inside ``outer`` (with margin)."""
    return (
        inner["x"] >= outer["x"] - margin
        and inner["y"] >= outer["y"] - margin
        and inner["x"] + inner["width"] <= outer["x"] + outer["width"] + margin
        and inner["y"] + inner["height"] <= outer["y"] + outer["height"] + margin
    )


def _rects_intersect(a: dict[str, float], b: dict[str, float]) -> bool:
    ax2 = a["x"] + a["width"]
    ay2 = a["y"] + a["height"]
    bx2 = b["x"] + b["width"]
    by2 = b["y"] + b["height"]
    return not (ax2 < b["x"] or a["x"] > bx2 or ay2 < b["y"] or a["y"] > by2)


def validate_child_bounds_placements(output_json: dict[str, Any], target_w: float, target_h: float) -> None:
    """Raise if brand/headline children or floating roles sit outside their parent / canvas."""
    canvas = {"x": 0.0, "y": 0.0, "width": float(target_w), "height": float(target_h)}
    margin = max(3.0, min(target_w, target_h) * 0.004)

    bg = find_by_role(output_json, "brand_group")
    if bg is not None:
        bb = get_bounds(bg)
        if bb["width"] > 0 and bb["height"] > 0:
            for ch in bg.get("children") or []:
                if not isinstance(ch, dict):
                    continue
                name = ch.get("name")
                if not isinstance(name, str):
                    continue
                cb = get_bounds(ch)
                if cb["width"] <= 0 or cb["height"] <= 0:
                    continue
                cx = cb["x"] + cb["width"] / 2.0
                cy = cb["y"] + cb["height"] / 2.0
                if not _point_inside_rect(cx, cy, bb, margin):
                    raise ValueError(
                        f"prototype placement invalid: brand_group child {name!r} center "
                        f"({cx:.1f},{cy:.1f}) outside parent bounds {bb!r}"
                    )

    hg = find_by_role(output_json, "headline_group")
    if hg is not None:
        hb = get_bounds(hg)
        if hb["width"] > 0 and hb["height"] > 0:
            for ch in hg.get("children") or []:
                if not isinstance(ch, dict):
                    continue
                name = ch.get("name")
                if name not in HEADLINE_GROUP_ALLOWED_DIRECT:
                    continue
                cb = get_bounds(ch)
                if cb["width"] <= 0 or cb["height"] <= 0:
                    continue
                cx = cb["x"] + cb["width"] / 2.0
                cy = cb["y"] + cb["height"] / 2.0
                if not _point_inside_rect(cx, cy, hb, margin * 2):
                    raise ValueError(
                        f"prototype placement invalid: headline_group child {name!r} center "
                        f"({cx:.1f},{cy:.1f}) outside parent bounds {hb!r}"
                    )

    logo = find_by_role(output_json, "logo")
    if logo is not None:
        lb = get_bounds(logo)
        if lb["width"] > 0 and lb["height"] > 0:
            for ch in logo.get("children") or []:
                if not isinstance(ch, dict):
                    continue
                name = ch.get("name")
                if name not in ("logo_back", "logo_fore"):
                    continue
                cb = get_bounds(ch)
                if cb["width"] <= 0 or cb["height"] <= 0:
                    continue
                cx = cb["x"] + cb["width"] / 2.0
                cy = cb["y"] + cb["height"] / 2.0
                if not _point_inside_rect(cx, cy, lb, margin):
                    raise ValueError(
                        f"prototype placement invalid: logo child {name!r} center "
                        f"({cx:.1f},{cy:.1f}) outside logo bounds {lb!r}"
                    )

    for role in FLOATING_PROTO_ROLES:
        node = find_by_role(output_json, role)
        if node is None:
            continue
        b = get_bounds(node)
        if b["width"] <= 0 or b["height"] <= 0:
            continue
        if role in ("background_gradient_1", "background_gradient_2"):
            continue
        if role == "age_badge":
            if not _rect_inside_rect(b, canvas, margin):
                raise ValueError(
                    f"prototype placement invalid: floating role {role!r} bounds {b!r} "
                    f"not inside canvas {canvas!r}"
                )
        elif not _rects_intersect(b, canvas):
            raise ValueError(
                f"prototype placement invalid: floating role {role!r} bounds {b!r} "
                f"does not intersect canvas {canvas!r}"
            )


def _apply_legal_text_bbox_from_prototype_if_needed(
    output_json: dict[str, Any],
    prototype: dict[str, Any],
    report: dict[str, Any],
) -> None:
    legal = find_by_role(output_json, "legal_text")
    if legal is None:
        return
    b = get_bounds(legal)
    if b["width"] >= 1.0 and b["height"] >= 1.0:
        report["legal_text_bbox_source"] = "transformer"
        return
    info = prototype.get("legal_text_relative")
    if not isinstance(info, dict):
        report["warnings"].append("legal_text bbox tiny/missing and prototype has no legal_text_relative")
        return
    anchor = str(info.get("anchor") or "canvas")
    rel_x = _num(info.get("x"), 0.0)
    rel_y = _num(info.get("y"), 0.0)
    rel_w = _num(info.get("width"), 0.0)
    rel_h = _num(info.get("height"), 0.0)
    if anchor == "background_shape":
        parent = find_by_role(output_json, "background_shape")
    else:
        parent = output_json
    if parent is None:
        report["warnings"].append("legal_text_relative anchor background_shape not found in output")
        return
    pb = get_bounds(parent)
    if pb["width"] <= 0 or pb["height"] <= 0:
        return
    set_bounds(
        legal,
        pb["x"] + rel_x * pb["width"],
        pb["y"] + rel_y * pb["height"],
        max(0.01, rel_w * pb["width"]),
        max(0.01, rel_h * pb["height"]),
    )
    report["legal_text_bbox_source"] = "prototype_relative"


def _apply_proto_text_style_keys(proto_style: dict[str, Any], target_node: dict[str, Any]) -> None:
    for key in PROTO_TEXT_STYLE_KEYS:
        if key not in proto_style:
            continue
        value = proto_style[key]
        if key in ("fills", "lineHeight", "letterSpacing"):
            target_node[key] = copy.deepcopy(value)
        elif key == "opacity" and isinstance(value, (int, float)) and math.isfinite(float(value)):
            target_node[key] = float(value)
        elif key == "fontSize" and isinstance(value, (int, float)) and math.isfinite(float(value)):
            target_node[key] = float(value)
        elif key == "fontName" and isinstance(value, dict) and value.get("family") and value.get("style"):
            target_node[key] = {"family": str(value["family"]), "style": str(value["style"])}
        elif key == "textAlignHorizontal" and isinstance(value, str) and value.strip():
            target_node[key] = value.strip().upper()
        elif key in ("textAlignVertical", "textAutoResize") and isinstance(value, str) and value.strip():
            target_node[key] = value.strip()
        else:
            target_node[key] = value


def _apply_exact_text_style(style: dict[str, Any], target_node: dict[str, Any]) -> None:
    for key, value in style.items():
        if key in ("fontName", "lineHeight", "letterSpacing"):
            target_node[key] = copy.deepcopy(value)
        elif key in ("fontSize", "opacity") and isinstance(value, (int, float)):
            target_node[key] = float(value)
        else:
            target_node[key] = value


def _copy_characters_and_fallback_paint(source_node: dict[str, Any] | None, target_node: dict[str, Any]) -> None:
    if not isinstance(source_node, dict):
        return
    if "characters" in source_node:
        target_node["characters"] = source_node.get("characters", "")
    font_name = source_node.get("fontName")
    if isinstance(font_name, dict) and font_name.get("family") and font_name.get("style"):
        if "fontName" not in target_node:
            target_node["fontName"] = {"family": str(font_name["family"]), "style": str(font_name["style"])}
    for key in ("fills", "lineHeight", "letterSpacing", "opacity", "paragraphSpacing", "textCase", "textDecoration"):
        if key in source_node and key not in target_node:
            target_node[key] = copy.deepcopy(source_node[key]) if key in ("fills", "lineHeight", "letterSpacing") else source_node[key]


def _default_align_horizontal(target_w: float, target_h: float) -> str:
    return "CENTER" if target_w <= target_h else "LEFT"


def _exact_text_style_for_target(role: str, target_w: float, target_h: float) -> dict[str, Any] | None:
    if _is_640_portrait_like(target_w, target_h):
        return PORTRAIT_640_EXACT_TEXT_STYLES.get(role)
    return None


def _apply_prototype_text_styles(
    source_json: dict[str, Any],
    output_json: dict[str, Any],
    prototype: dict[str, Any],
    proto_styles: dict[str, Any],
    target_w: float,
    target_h: float,
    report: dict[str, Any],
) -> None:
    headline_node = find_by_role(output_json, "headline")
    headline_size = _headline_anchor_font_size(headline_node, target_w, target_h)
    default_align = _default_align_horizontal(target_w, target_h)

    for role in TEXT_ROLES_STYLE:
        node = find_by_role(output_json, role)
        if node is None or not _is_text_node(node):
            continue
        source_node = find_by_role(source_json, role)
        _copy_characters_and_fallback_paint(source_node, node)
        exact_style = _exact_text_style_for_target(role, target_w, target_h)
        if exact_style is not None:
            _apply_exact_text_style(exact_style, node)
            report["font_size_fitted"] += 1
            continue

        proto_style = proto_styles.get(role) if isinstance(proto_styles.get(role), dict) else {}
        _apply_proto_text_style_keys(proto_style, node)

        scaled = _prototype_scaled_font_size(role, source_node, node, prototype, proto_style, target_w, target_h)
        if scaled is None:
            scaled = _role_based_font_size(role, node, headline_size, target_w, target_h)
        if scaled is not None:
            node["fontSize"] = scaled
            report["font_size_fitted"] += 1
            lh = node.get("lineHeight")
            if isinstance(lh, dict) and str(lh.get("unit", "")).upper() == "PIXELS":
                node["lineHeight"] = {"unit": "PIXELS", "value": max(1.0, round(float(scaled) * 1.15))}

        ha = node.get("textAlignHorizontal")
        if not isinstance(ha, str) or not ha.strip():
            node["textAlignHorizontal"] = default_align
        elif isinstance(ha, str):
            node["textAlignHorizontal"] = ha.strip().upper()

        if not isinstance(node.get("textAlignVertical"), str) or not str(node.get("textAlignVertical")).strip():
            pv = proto_style.get("textAlignVertical")
            if isinstance(pv, str) and pv.strip():
                node["textAlignVertical"] = pv
            elif isinstance(source_node, dict) and isinstance(source_node.get("textAlignVertical"), str):
                node["textAlignVertical"] = source_node["textAlignVertical"]
            else:
                node["textAlignVertical"] = "TOP"

        tar = node.get("textAutoResize")
        if not isinstance(tar, str) or not tar.strip():
            node["textAutoResize"] = "NONE"


def _headline_anchor_font_size(headline_node: dict[str, Any] | None, target_w: float, target_h: float) -> float | None:
    if headline_node is None:
        return None
    height = get_bounds(headline_node)["height"]
    if height <= 0:
        return None
    return _clamp_role_font("headline", height * 0.48, target_w, target_h)


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
    return None


def _clamp_role_font(role: str, size: float, target_w: float | None = None, target_h: float | None = None) -> float:
    clamps = PORTRAIT_640_TEXT_CLAMPS if _is_640_portrait_like(target_w, target_h) else TEXT_ROLE_CLAMPS
    lo, hi = clamps.get(role, (4.0, 128.0))
    return max(lo, min(hi, size))


def _is_640_portrait_like(target_w: float | None, target_h: float | None) -> bool:
    if target_w is None or target_h is None:
        return False
    return target_w <= target_h and abs(target_w - 640.0) <= 80.0 and abs(target_h - 720.0) <= 120.0


def _font_size_from_source_layout(
    role: str,
    source_node: dict[str, Any] | None,
    target_node: dict[str, Any],
) -> float | None:
    """Scale effective source font size by target/source text box height (tracks reflow across aspect ratios)."""
    if not isinstance(source_node, dict):
        return None
    sb = get_bounds(source_node)
    tb = get_bounds(target_node)
    if sb["height"] <= 0 or tb["height"] <= 0:
        return None
    raw = source_node.get("fontSize")
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        src_fs = float(raw)
    else:
        src_fs = inferred_text_font_size_for_role(role, source_node)
    return src_fs * tb["height"] / sb["height"]


def _prototype_scaled_font_size(
    role: str,
    source_node: dict[str, Any] | None,
    target_node: dict[str, Any],
    prototype: dict[str, Any],
    proto_style: dict[str, Any],
    target_w: float,
    target_h: float,
) -> float | None:
    target_height = get_bounds(target_node)["height"]
    if target_height <= 0:
        return None

    from_source = _font_size_from_source_layout(role, source_node, target_node)
    if from_source is not None and math.isfinite(from_source):
        return _clamp_role_font(role, from_source, target_w, target_h)

    proto_font_size = proto_style.get("fontSize")
    proto_height = _prototype_abs_height(role, prototype)
    if isinstance(proto_font_size, (int, float)) and math.isfinite(float(proto_font_size)) and proto_height > 0:
        size = float(proto_font_size) * target_height / proto_height
        return _clamp_role_font(role, size, target_w, target_h)
    return None


def _prototype_parent_norm_bbox(parent_role: str, prototype: dict[str, Any]) -> dict[str, Any] | None:
    structs = prototype.get("structural_bboxes") if isinstance(prototype.get("structural_bboxes"), dict) else {}
    pn = structs.get(parent_role)
    if isinstance(pn, dict):
        return pn
    rb = prototype.get("role_bboxes") if isinstance(prototype.get("role_bboxes"), dict) else {}
    pn = rb.get(parent_role)
    return pn if isinstance(pn, dict) else None


def _prototype_abs_height(role: str, prototype: dict[str, Any]) -> float:
    canvas = prototype.get("canvas") if isinstance(prototype.get("canvas"), dict) else {}
    canvas_h = _num(canvas.get("height"), 0.0)
    if canvas_h <= 0:
        return 0.0
    if role in CHILD_PARENT:
        rels = prototype.get("child_relative_bboxes") if isinstance(prototype.get("child_relative_bboxes"), dict) else {}
        rel = rels.get(role)
        parent_role = CHILD_PARENT[role]
        parent = _prototype_parent_norm_bbox(parent_role, prototype)
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
