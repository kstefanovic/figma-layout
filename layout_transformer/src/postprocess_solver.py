"""Orientation-based text layout rules for layout postprocess."""

from __future__ import annotations

from typing import Any

from .postprocess import find_by_role, get_bounds, set_bounds, walk_nodes

TEXT_ALIGNMENT_ROLES = {"headline", "subheadline_delivery_time", "legal_text"}
HEADLINE_TEXT_ROLES = ("headline", "subheadline_delivery_time")


def get_target_text_alignment(target_w: float, target_h: float) -> str:
    return "LEFT" if target_w > target_h else "CENTER"


def set_text_alignment(node: dict[str, Any] | None, align: str) -> None:
    if node is None:
        return
    node["textAlignHorizontal"] = align
    node["textAutoResize"] = "NONE"
    style = node.get("style")
    if isinstance(style, dict):
        style["textAlignHorizontal"] = align
        style["textAutoResize"] = "NONE"


def apply_text_alignment_recursive(root: dict[str, Any], align: str) -> dict[str, int]:
    applied = 0
    fitted = 0

    def walk(node: Any, active_alignment_role: str | None = None) -> None:
        nonlocal applied, fitted
        if not isinstance(node, dict):
            return
        name = node.get("name")
        next_alignment_role = str(name) if name in TEXT_ALIGNMENT_ROLES else active_alignment_role

        if _is_text_node(node) and _fit_text_font_to_bounds(node, next_alignment_role):
            fitted += 1

        if name in TEXT_ALIGNMENT_ROLES or (next_alignment_role and _is_text_node(node)):
            set_text_alignment(node, align)
            applied += 1
        for child in node.get("children") or []:
            walk(child, next_alignment_role)

    walk(root)
    return {"text_alignment_applied": applied, "font_size_fitted": fitted}


def apply_orientation_text_layout(
    source_root: dict[str, Any],
    output_root: dict[str, Any],
    target_w: float,
    target_h: float,
) -> dict[str, Any]:
    """Apply target-orientation text alignment after child placement."""
    align = get_target_text_alignment(target_w, target_h)
    report: dict[str, Any] = {
        "text_alignment": align,
        "text_alignment_applied": 0,
        "headline_children_aligned": 0,
        "font_size_fitted": 0,
        "warnings": [],
    }
    text_report = apply_text_alignment_recursive(output_root, align)
    report["text_alignment_applied"] = text_report["text_alignment_applied"]
    report["font_size_fitted"] = text_report["font_size_fitted"]
    return report


def build_text_alignment_constraints(
    source_root: dict[str, Any],
    output_root: dict[str, Any],
    target_w: float,
    target_h: float,
) -> list[dict[str, Any]]:
    """Return declarative constraints represented by the deterministic alignment pass."""
    align = get_target_text_alignment(target_w, target_h)
    parent = find_by_role(output_root, "headline_group")
    if parent is None:
        return []
    parent_bounds = get_bounds(parent)
    constraints: list[dict[str, Any]] = []
    if align == "LEFT":
        left_padding = _source_left_padding(source_root)
        x = parent_bounds["x"] + left_padding * parent_bounds["width"]
        for role in HEADLINE_TEXT_ROLES:
            constraints.append({"kind": "soft_x_near", "role": role, "target_x": x, "alignment": align})
    else:
        parent_center_x = parent_bounds["x"] + parent_bounds["width"] / 2.0
        for role in HEADLINE_TEXT_ROLES:
            constraints.append(
                {"kind": "soft_center_x_near", "role": role, "target_center_x": parent_center_x, "alignment": align}
            )
    return constraints


def _align_headline_children(
    source_root: dict[str, Any],
    output_root: dict[str, Any],
    align: str,
    report: dict[str, Any],
) -> None:
    parent = find_by_role(output_root, "headline_group")
    if parent is None:
        report["warnings"].append("missing headline_group for text alignment")
        return
    parent_bounds = get_bounds(parent)
    if parent_bounds["width"] <= 0 or parent_bounds["height"] <= 0:
        report["warnings"].append("bad headline_group bounds for text alignment")
        return

    _fit_text_children_width(output_root, parent_bounds, report)
    if align == "LEFT":
        pad_ratio = _source_left_padding(source_root)
        target_x = parent_bounds["x"] + pad_ratio * parent_bounds["width"]
        for role in HEADLINE_TEXT_ROLES:
            child = find_by_role(output_root, role)
            if child is None:
                continue
            bounds = get_bounds(child)
            dx = target_x - bounds["x"]
            _translate_subtree(child, dx, 0.0)
            report["headline_children_aligned"] += 1
    else:
        for role in HEADLINE_TEXT_ROLES:
            child = find_by_role(output_root, role)
            if child is None:
                continue
            bounds = get_bounds(child)
            target_x = parent_bounds["x"] + (parent_bounds["width"] - bounds["width"]) / 2.0
            dx = target_x - bounds["x"]
            _translate_subtree(child, dx, 0.0)
            report["headline_children_aligned"] += 1

    _pack_headline_children_vertically(source_root, output_root, parent_bounds, report)


def _source_left_padding(source_root: dict[str, Any]) -> float:
    source_parent = find_by_role(source_root, "headline_group")
    if source_parent is None:
        return 0.0
    parent_bounds = get_bounds(source_parent)
    if parent_bounds["width"] <= 0:
        return 0.0
    for role in HEADLINE_TEXT_ROLES:
        child = find_by_role(source_root, role)
        if child is None:
            continue
        child_bounds = get_bounds(child)
        pad_ratio = (child_bounds["x"] - parent_bounds["x"]) / parent_bounds["width"]
        return min(0.15, max(0.0, pad_ratio))
    return 0.0


def _fit_text_children_width(output_root: dict[str, Any], parent_bounds: dict[str, float], report: dict[str, Any]) -> None:
    for role in HEADLINE_TEXT_ROLES:
        child = find_by_role(output_root, role)
        if child is None:
            continue
        bounds = get_bounds(child)
        if bounds["width"] <= parent_bounds["width"] or bounds["width"] <= 0:
            continue
        scale = parent_bounds["width"] / bounds["width"]
        _scale_subtree(child, bounds["x"], bounds["y"], scale)
        report["warnings"].append(f"scaled {role} by {scale:.4f} for text parent width")


def _pack_headline_children_vertically(
    source_root: dict[str, Any],
    output_root: dict[str, Any],
    parent_bounds: dict[str, float],
    report: dict[str, Any],
) -> None:
    children = []
    for role in HEADLINE_TEXT_ROLES:
        node = find_by_role(output_root, role)
        if node is not None:
            children.append((role, node, get_bounds(node)))
    if not children:
        return

    children.sort(key=lambda item: item[2]["y"])
    gap = max(2.0, parent_bounds["height"] * 0.035)
    total_height = sum(item[2]["height"] for item in children) + gap * max(0, len(children) - 1)
    if total_height > parent_bounds["height"]:
        scale = max(0.05, (parent_bounds["height"] - gap * max(0, len(children) - 1)) / max(1.0, sum(item[2]["height"] for item in children)))
        for role, node, bounds in children:
            _scale_subtree(node, bounds["x"], bounds["y"], scale)
            report["warnings"].append(f"scaled {role} by {scale:.4f} for text parent height")
        children = [(role, node, get_bounds(node)) for role, node, _bounds in children]
        total_height = sum(item[2]["height"] for item in children) + gap * max(0, len(children) - 1)

    top_ratio = _source_top_padding(source_root)
    desired_start = parent_bounds["y"] + top_ratio * parent_bounds["height"]
    min_start = parent_bounds["y"]
    max_start = parent_bounds["y"] + parent_bounds["height"] - total_height
    current_y = min(max(desired_start, min_start), max_start)
    for role, node, bounds in children:
        dx = 0.0
        if bounds["x"] < parent_bounds["x"]:
            dx = parent_bounds["x"] - bounds["x"]
        elif bounds["x"] + bounds["width"] > parent_bounds["x"] + parent_bounds["width"]:
            dx = parent_bounds["x"] + parent_bounds["width"] - bounds["x"] - bounds["width"]
        dy = current_y - bounds["y"]
        _translate_subtree(node, dx, dy)
        current_y += bounds["height"] + gap


def _scale_subtree(node: dict[str, Any], origin_x: float, origin_y: float, scale: float) -> None:
    for item in walk_nodes(node):
        bounds = get_bounds(item)
        set_bounds(
            item,
            origin_x + (bounds["x"] - origin_x) * scale,
            origin_y + (bounds["y"] - origin_y) * scale,
            bounds["width"] * scale,
            bounds["height"] * scale,
        )
        for field in ("fontSize", "letterSpacing", "strokeWeight", "cornerRadius"):
            value = item.get(field)
            if isinstance(value, (int, float)):
                scaled = float(value) * scale
                item[field] = min(128.0, max(4.0, scaled)) if field == "fontSize" else scaled
            style = item.get("style")
            if isinstance(style, dict):
                style_value = style.get(field)
                if isinstance(style_value, (int, float)):
                    scaled = float(style_value) * scale
                    style[field] = min(128.0, max(4.0, scaled)) if field == "fontSize" else scaled


def _translate_subtree(node: dict[str, Any], dx: float, dy: float) -> None:
    for item in walk_nodes(node):
        bounds = get_bounds(item)
        set_bounds(item, bounds["x"] + dx, bounds["y"] + dy, bounds["width"], bounds["height"])


def _source_top_padding(source_root: dict[str, Any]) -> float:
    source_parent = find_by_role(source_root, "headline_group")
    if source_parent is None:
        return 0.0
    parent_bounds = get_bounds(source_parent)
    if parent_bounds["height"] <= 0:
        return 0.0
    for role in HEADLINE_TEXT_ROLES:
        child = find_by_role(source_root, role)
        if child is None:
            continue
        child_bounds = get_bounds(child)
        pad_ratio = (child_bounds["y"] - parent_bounds["y"]) / parent_bounds["height"]
        return min(0.15, max(0.0, pad_ratio))
    return 0.0


def _is_text_node(node: dict[str, Any]) -> bool:
    node_type = str(node.get("type") or "").lower().replace("_", " ")
    return node_type == "text" or "characters" in node


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _set_font_size(node: dict[str, Any], font_size: float) -> None:
    size = min(128.0, max(4.0, float(font_size)))
    node["fontSize"] = size
    node["textAutoResize"] = "NONE"
    style = node.get("style")
    if isinstance(style, dict):
        style["fontSize"] = size
        style["textAutoResize"] = "NONE"


def _text_line_count(node: dict[str, Any], bounds: dict[str, float]) -> int:
    chars = str(node.get("characters") or "")
    explicit_lines = [line for line in chars.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    explicit_count = max(1, len(explicit_lines))
    if "\n" in chars:
        return explicit_count

    compact = "".join(chars.split())
    if not compact:
        return 1
    # Estimate wrapping for exported Figma text boxes that lost explicit line breaks.
    # Wide legal copy can be many lines; short headlines usually stay 1-2 lines.
    width = max(1.0, bounds["width"])
    height = max(1.0, bounds["height"])
    aspect = width / height
    if len(compact) > 70 and aspect > 4.0:
        return max(1, min(4, round(len(compact) / 55)))
    if len(compact) > 24 and aspect < 3.0:
        return max(1, min(4, round(len(compact) / 13)))
    return explicit_count


def _font_fit_factor(role: str | None) -> float:
    if role == "headline":
        return 0.72
    if role == "subheadline_delivery_time":
        return 0.66
    if role == "legal_text":
        return 0.46
    return 0.70


def _fit_text_font_to_bounds(node: dict[str, Any], role: str | None) -> bool:
    bounds = get_bounds(node)
    if bounds["width"] <= 0 or bounds["height"] <= 0:
        return False

    line_count = _text_line_count(node, bounds)
    height_limited = bounds["height"] / max(1, line_count) * _font_fit_factor(role)

    chars = str(node.get("characters") or "")
    longest_line = max((len(line.strip()) for line in chars.splitlines()), default=len(chars.strip()))
    if longest_line <= 0:
        width_limited = height_limited
    else:
        width_limited = bounds["width"] / max(1, longest_line) * 1.75

    fitted = max(1.0, min(height_limited, width_limited))
    current = _numeric(node.get("fontSize"))
    if current is None:
        _set_font_size(node, fitted)
        return True

    # Let source-derived typography survive when it already fits, but clamp clear overflow.
    max_allowed = max(1.0, fitted * 1.08)
    if current > max_allowed:
        _set_font_size(node, fitted)
        return True
    node["textAutoResize"] = "NONE"
    style = node.get("style")
    if isinstance(style, dict):
        style["textAutoResize"] = "NONE"
    return False
