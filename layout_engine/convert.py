
import copy
import json
import sys
from pathlib import Path

from .orientation import get_orientation
from .semantic_extractor import (
    collect_semantic_nodes,
    get_primary_node,
    get_largest_node,
)
from .cp_sat_layout import solve_layout
from .transform_children import move_and_scale_group, set_bounds


def find_role_nodes(nodes_by_role: dict) -> dict:
    return {
        # Only exact "background" should be solver-controlled.
        # background_shape / background_gradient_* are handled by root-scale.
        "background": get_largest_node(
            nodes_by_role,
            "background",
        ),

        "hero_image": get_largest_node(
            nodes_by_role,
            "hero_image",
            "image_zone",
        ),

        "brand_group": get_primary_node(
            nodes_by_role,
            "brand_group",
        ),

        "headline_group": get_primary_node(
            nodes_by_role,
            "headline_group",
        ),

        "legal_text": get_primary_node(
            nodes_by_role,
            "legal_text",
        ),

        "age_badge": get_primary_node(
            nodes_by_role,
            "age_badge",
        ),

        # Only real decoration_group should be solver-controlled.
        # Loose star_decoration_* nodes should only be root-scaled.
        "decoration_group": get_largest_node(
            nodes_by_role,
            "decoration_group",
        ),
    }


def get_node_id_set(nodes: list) -> set:
    return {
        node.get("id")
        for node in nodes
        if node is not None and node.get("id")
    }


def should_root_scale_node(node: dict) -> bool:
    """
    Scale loose visual helper nodes that are not controlled by CP-SAT.

    Examples:
    - background_shape
    - background_gradient_1
    - background_gradient_2
    - star_decoration_1
    - star_decoration_2

    Do NOT use this for semantic groups already optimized by solver:
    - hero_image
    - brand_group
    - headline_group
    - legal_text
    - age_badge
    - real background
    - real decoration_group
    """
    name = (node.get("name") or "").lower().strip()
    node_type = (node.get("type") or "").lower().strip()

    if name == "background_shape" or name.startswith("background_shape"):
        return True

    if name.startswith("background_gradient"):
        return True

    if name.startswith("gradient"):
        return True

    if "star_decoration" in name:
        return True

    if name.startswith("decoration_"):
        return True

    if name.startswith("shape_decoration"):
        return True

    # Loose top-level stars should scale proportionally.
    if node_type == "star":
        return True

    return False


def root_scale_node(node: dict, sx: float, sy: float):
    """
    Scale a loose node from source root coordinate space to target root coordinate space.

    Decorations/stars:
    - x uses sx
    - y uses sy
    - width/height use uniform scale to preserve shape ratio

    Gradients/background helper shapes:
    - x/width use sx
    - y/height use sy
    """
    b = node.get("bounds")
    if not b:
        return

    name = (node.get("name") or "").lower().strip()
    node_type = (node.get("type") or "").lower().strip()

    old_x = float(b.get("x", 0))
    old_y = float(b.get("y", 0))
    old_w = float(b.get("width", 0))
    old_h = float(b.get("height", 0))

    is_decoration = (
        node_type == "star"
        or "star_decoration" in name
        or name.startswith("decoration_")
        or name.startswith("shape_decoration")
    )

    if is_decoration:
        uniform = min(sx, sy)

        b["x"] = round(old_x * sx, 2)
        b["y"] = round(old_y * sy, 2)
        b["width"] = round(old_w * uniform, 2)
        b["height"] = round(old_h * uniform, 2)
        return

    # Background helpers and gradients may stretch.
    b["x"] = round(old_x * sx, 2)
    b["y"] = round(old_y * sy, 2)
    b["width"] = round(old_w * sx, 2)
    b["height"] = round(old_h * sy, 2)


def root_scale_loose_nodes(banner: dict, optimized_ids: set, sx: float, sy: float):
    """
    Scale only top-level loose helper nodes.

    Children inside optimized groups are already transformed by move_and_scale_group().
    """
    for child in banner.get("children", []) or []:
        child_id = child.get("id")

        if child_id in optimized_ids:
            continue

        if should_root_scale_node(child):
            root_scale_node(child, sx, sy)


def apply_box_to_node(node: dict, box: dict):
    if node.get("children"):
        move_and_scale_group(node, box)
    else:
        set_bounds(
            node,
            box["x"],
            box["y"],
            box["width"],
            box["height"],
        )


def convert_banner(source_json: dict, target_width: int, target_height: int) -> dict:
    banner = copy.deepcopy(source_json)

    # Save source root size BEFORE changing root bounds.
    old_bounds = copy.deepcopy(banner.get("bounds", {}) or {})
    old_width = max(float(old_bounds.get("width", target_width)), 1)
    old_height = max(float(old_bounds.get("height", target_height)), 1)

    sx = float(target_width) / old_width
    sy = float(target_height) / old_height

    # Set target root bounds.
    banner.setdefault("bounds", {})
    banner["bounds"]["x"] = 0
    banner["bounds"]["y"] = 0
    banner["bounds"]["width"] = target_width
    banner["bounds"]["height"] = target_height

    orientation = get_orientation(target_width, target_height)

    nodes_by_role = collect_semantic_nodes(banner)
    role_nodes = find_role_nodes(nodes_by_role)

    available_roles = {
        role
        for role, node in role_nodes.items()
        if node is not None
    }

    boxes, mode = solve_layout(
        orientation=orientation,
        target_w=target_width,
        target_h=target_height,
        available_roles=available_roles,
    )

    # 1. Apply solver boxes to main semantic nodes.
    for role, node in role_nodes.items():
        if node is None:
            continue

        box = boxes.get(role)
        if not box:
            continue

        apply_box_to_node(node, box)

    # 2. Root-scale loose non-optimized helper nodes.
    # Example: background_shape, background_gradient_*, star_decoration_*.
    optimized_ids = get_node_id_set(list(role_nodes.values()))

    root_scale_loose_nodes(
        banner=banner,
        optimized_ids=optimized_ids,
        sx=sx,
        sy=sy,
    )

    banner.setdefault("metadata", {})
    banner["metadata"]["layout_engine"] = {
        "engine": "ortools_cp_sat",
        "mode": mode,
        "orientation": orientation,
        "target_width": target_width,
        "target_height": target_height,
        "source_width": old_width,
        "source_height": old_height,
        "scale_x": round(sx, 4),
        "scale_y": round(sy, 4),
        "available_roles": sorted(list(available_roles)),
    }

    return banner


def load_input(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # MVP behavior:
    # If a list is provided, convert the first frame.
    # Later we can add --name or --index.
    if isinstance(data, list):
        if not data:
            raise ValueError("Input JSON list is empty.")
        return data[0]

    return data


def main():
    if len(sys.argv) != 5:
        print("Usage:")
        print("  python -m layout_engine.convert input.json width height output.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    target_w = int(sys.argv[2])
    target_h = int(sys.argv[3])
    output_path = Path(sys.argv[4])

    source = load_input(input_path)
    converted = convert_banner(source, target_w, target_h)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Saved converted banner to {output_path}")
    print(f"Layout metadata: {converted.get('metadata', {}).get('layout_engine', {})}")


if __name__ == "__main__":
    main()