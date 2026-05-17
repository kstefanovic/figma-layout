"""Validate rich semantic output from LayoutTransformer V2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .rich_utils import (
    bounds_of,
    contains,
    flatten_role_nodes,
    get_canvas_size,
    has_gradient_transform,
    has_image_hash,
    load_one_frame,
    normalized_bbox,
    walk_nodes,
)
from .schema import CHILD_PARENT, CHILD_ROLES, FLOATING_ROLES

TEXT_ROLES = ["headline", "subheadline_delivery_time", "legal_text", "age_badge"]
STAR_ROLES = ["star_decoration_1", "star_decoration_2"]
GRADIENT_ROLES = ["background_gradient_1", "background_gradient_2"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-width", type=float)
    parser.add_argument("--target-height", type=float)
    args = parser.parse_args()

    final_json = load_one_frame(args.json, 0)
    source_json = load_one_frame(args.source_json, args.source_index) if args.source_json else None
    errors = validate(final_json, source_json=source_json, target_width=args.target_width, target_height=args.target_height)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Rich output validation passed")


def validate(
    final_json: dict[str, Any],
    *,
    source_json: dict[str, Any] | None = None,
    target_width: float | None = None,
    target_height: float | None = None,
) -> list[str]:
    errors: list[str] = []
    nodes = flatten_role_nodes(final_json)
    canvas_w, canvas_h = get_canvas_size(final_json)
    if target_width is not None and round(canvas_w) != round(target_width):
        errors.append(f"root width {canvas_w} does not match target {target_width}")
    if target_height is not None and round(canvas_h) != round(target_height):
        errors.append(f"root height {canvas_h} does not match target {target_height}")

    hero = nodes.get("hero_image")
    if hero is not None and not has_image_hash(hero):
        errors.append("hero_image has no imageHash in node or descendants")

    for role in TEXT_ROLES:
        node = nodes.get(role)
        if node is None:
            continue
        for key in ("fontSize", "fontName", "fills", "lineHeight", "letterSpacing"):
            if key not in node or node.get(key) in (None, [], {}):
                errors.append(f"{role} missing rich text metadata: {key}")

    for role in GRADIENT_ROLES:
        node = nodes.get(role)
        if node is not None and not has_gradient_transform(node):
            errors.append(f"{role} gradient is missing gradientTransform")

    for role in STAR_ROLES:
        node = nodes.get(role)
        if node is None:
            continue
        if not node.get("fills"):
            errors.append(f"{role} missing fills")
        if "effects" not in node:
            errors.append(f"{role} missing effects")

    brand = nodes.get("brand_group")
    if brand is not None:
        brand_bounds = bounds_of(brand)
        for role in ("brand_name_first_part_1", "brand_name_first_part_2", "brand_name_second", "logo"):
            child = nodes.get(role)
            if child is not None and not contains(brand_bounds, bounds_of(child), tolerance=2.0):
                errors.append(f"{role} is outside brand_group")

    headline_group = nodes.get("headline_group")
    if headline_group is not None:
        headline_bounds = bounds_of(headline_group)
        for role in ("headline", "subheadline_delivery_time"):
            child = nodes.get(role)
            if child is not None and not contains(headline_bounds, bounds_of(child), tolerance=max(canvas_w, canvas_h) * 0.05):
                errors.append(f"{role} is not near headline_group")

    if source_json is not None:
        source_nodes = flatten_role_nodes(source_json)
        source_w, source_h = get_canvas_size(source_json)
        for role in FLOATING_ROLES:
            if role in nodes and role in source_nodes and (round(source_w) != round(canvas_w) or round(source_h) != round(canvas_h)):
                if _same_bbox(normalized_bbox(source_nodes[role], source_w, source_h), normalized_bbox(nodes[role], canvas_w, canvas_h)):
                    errors.append(f"{role} appears to be using old source coordinates")
        errors.extend(validate_no_reparent(source_json, final_json))

    return errors


def validate_no_reparent(source_json: dict[str, Any], final_json: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_parent = role_parent_map(source_json)
    final_parent = role_parent_map(final_json)
    for role in set(source_parent) & set(final_parent):
        if source_parent[role] != final_parent[role]:
            errors.append(f"{role} reparented from {source_parent[role]!r} to {final_parent[role]!r}")
    for role, expected_parent in CHILD_PARENT.items():
        if role in final_parent and final_parent[role] != expected_parent:
            errors.append(f"{role} expected parent {expected_parent!r}, got {final_parent[role]!r}")
    return errors


def role_parent_map(frame: dict[str, Any]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}

    def walk(node: Any, parent_role: str | None) -> None:
        if not isinstance(node, dict):
            return
        role = node.get("name")
        if isinstance(role, str):
            out[role] = parent_role
            next_parent = role
        else:
            next_parent = parent_role
        for child in node.get("children") or []:
            walk(child, next_parent)

    walk(frame, None)
    return out


def _same_bbox(a: list[float], b: list[float], tolerance: float = 1e-4) -> bool:
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


if __name__ == "__main__":
    main()

