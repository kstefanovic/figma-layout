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
        "background": get_largest_node(
            nodes_by_role,
            "background",
            "background_shape",
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
        "decoration_group": get_largest_node(
            nodes_by_role,
            "decoration_group",
            "decoration",
        ),
    }


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

    banner.setdefault("bounds", {})
    banner["bounds"]["x"] = 0
    banner["bounds"]["y"] = 0
    banner["bounds"]["width"] = target_width
    banner["bounds"]["height"] = target_height

    orientation = get_orientation(target_width, target_height)

    nodes_by_role = collect_semantic_nodes(banner)
    role_nodes = find_role_nodes(nodes_by_role)

    available_roles = {
        role for role, node in role_nodes.items()
        if node is not None
    }

    boxes, mode = solve_layout(
        orientation=orientation,
        target_w=target_width,
        target_h=target_height,
        available_roles=available_roles,
    )

    for role, node in role_nodes.items():
        if node is None:
            continue

        box = boxes.get(role)
        if not box:
            continue

        apply_box_to_node(node, box)

    banner.setdefault("metadata", {})
    banner["metadata"]["layout_engine"] = {
        "engine": "ortools_cp_sat",
        "mode": mode,
        "orientation": orientation,
        "target_width": target_width,
        "target_height": target_height,
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
