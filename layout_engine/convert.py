
import copy
import argparse
import json
from pathlib import Path

from .orientation import get_orientation
from .semantic_extractor import (
    collect_semantic_nodes,
    get_primary_node,
    get_largest_node,
)
from .cp_sat_layout import solve_layout
from .transform_children import move_and_scale_group, set_bounds_and_scale_text
from .visual_layout import compute_visual_boxes
from .retrieval.visual_retriever import load_visual_db, retrieve_visual_priors


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


def find_node_by_id_or_path(root: dict, figma_id: str = "", path: str = "") -> dict | None:
    found = None

    def walk(node: dict):
        nonlocal found
        if found is not None:
            return
        if figma_id and str(node.get("id") or "") == str(figma_id):
            found = node
            return
        if path and str(node.get("path") or "") == str(path):
            found = node
            return
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                walk(child)

    walk(root)
    return found


def find_largest_named_node(root: dict, name_prefix: str) -> dict | None:
    best = None
    best_area = -1.0
    prefix = name_prefix.lower()

    def walk(node: dict):
        nonlocal best, best_area
        name = str(node.get("name") or "").strip().lower()
        if name == prefix or name.startswith(prefix + "_"):
            b = node.get("bounds") or {}
            try:
                area = float(b.get("width") or 0) * float(b.get("height") or 0)
            except (TypeError, ValueError):
                area = 0.0
            if area > best_area:
                best = node
                best_area = area
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                walk(child)

    walk(root)
    return best


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
        set_bounds_and_scale_text(
            node,
            box["x"],
            box["y"],
            box["width"],
            box["height"],
        )


def normalized_prior_to_abs_box(prior: dict, target_width: int, target_height: int) -> dict:
    return {
        "x": round(float(prior.get("x", 0)) * target_width, 2),
        "y": round(float(prior.get("y", 0)) * target_height, 2),
        "width": round(float(prior.get("w", prior.get("width", 0))) * target_width, 2),
        "height": round(float(prior.get("h", prior.get("height", 0))) * target_height, 2),
    }


def _resolve_retrieval_target(banner: dict, role: str, role_nodes: dict, retrieval_result: dict | None) -> dict | None:
    if role == "hero_image" and role_nodes.get("hero_image") is not None:
        return role_nodes["hero_image"]
    if role == "background_shape":
        semantic_bg = find_largest_named_node(banner, "background_shape")
        if semantic_bg is not None:
            return semantic_bg
    selected = ((retrieval_result or {}).get("selected_raw_candidates") or {}).get(role)
    if not selected:
        return None
    return find_node_by_id_or_path(
        banner,
        figma_id=str(selected.get("figma_id") or ""),
        path=str(selected.get("path") or ""),
    )


def convert_banner(
    source_json: dict,
    target_width: int,
    target_height: int,
    visual_retrieval_db: dict | str | None = None,
    visual_retrieval_top_k: int = 15,
    visual_mode: str = "default",
    gnn_layout_checkpoint: str | Path | None = None,
) -> dict:
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
    gnn_role_nodes = {
        role: role_nodes.get(role)
        for role in ("brand_group", "headline_group", "legal_text")
    }

    retrieval_result = None
    retrieval_enabled = visual_mode == "retrieval" or visual_retrieval_db is not None
    if retrieval_enabled and visual_retrieval_db is not None:
        db = load_visual_db(str(visual_retrieval_db)) if isinstance(visual_retrieval_db, (str, Path)) else visual_retrieval_db
        retrieval_result = retrieve_visual_priors(
            db,
            source_json,
            target_width,
            target_height,
            top_k=visual_retrieval_top_k,
        )

    if retrieval_result:
        for role in ("hero_image",):
            if role_nodes.get(role) is not None:
                continue
            candidate_node = _resolve_retrieval_target(banner, role, role_nodes, retrieval_result)
            if candidate_node is not None:
                role_nodes[role] = candidate_node
                candidate_node["name"] = role

    available_roles = {
        role
        for role, node in role_nodes.items()
        if node is not None
    }

    learned_priors = (retrieval_result or {}).get("priors") or None
    locked_roles = set()
    if learned_priors and learned_priors.get("hero_image") and role_nodes.get("hero_image") is not None:
        locked_roles.add("hero_image")

    if retrieval_result:
        # Retrieval/layout-engine mode is visual-only. Text/brand roles will be handled by GNN.
        for role in ("brand_group", "headline_group", "legal_text"):
            role_nodes[role] = None
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
        learned_priors=learned_priors,
        locked_roles=locked_roles,
    )

    if retrieval_result:
        boxes = compute_visual_boxes(
            role_nodes=role_nodes,
            solved_boxes=boxes,
            orientation=orientation,
            target_width=target_width,
            target_height=target_height,
            retrieval_priors=retrieval_result,
        )

    # 1. Apply solver boxes to main semantic nodes.
    for role, node in role_nodes.items():
        if node is None:
            continue

        box = boxes.get(role)
        if not box:
            continue

        apply_box_to_node(node, box)

    retrieval_targets = {}
    if retrieval_result:
        for role in ("hero_image", "background_shape"):
            node = _resolve_retrieval_target(banner, role, role_nodes, retrieval_result)
            box = (retrieval_result.get("abs_boxes") or {}).get(role)
            if node is not None and box:
                retrieval_targets[role] = node
                apply_box_to_node(node, box)

    gnn_result = None
    gnn_targets = {}
    if gnn_layout_checkpoint:
        from gnn_layout.src.predict import predict_priors_for_banner

        gnn_result = predict_priors_for_banner(
            gnn_layout_checkpoint,
            source_json,
            target_width,
            target_height,
        )
        for role in ("brand_group", "headline_group", "legal_text"):
            node = gnn_role_nodes.get(role)
            prior = (gnn_result.get("priors") or {}).get(role)
            if node is not None and prior:
                box = normalized_prior_to_abs_box(prior, target_width, target_height)
                gnn_targets[role] = {"node": node, "box": box}
                apply_box_to_node(node, box)

    # 2. Root-scale loose non-optimized helper nodes.
    # Example: background_shape, background_gradient_*, star_decoration_*.
    optimized_ids = get_node_id_set(list(role_nodes.values()))
    optimized_ids.update(get_node_id_set(list(retrieval_targets.values())))
    optimized_ids.update(get_node_id_set([item["node"] for item in gnn_targets.values()]))

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

    if retrieval_enabled:
        vr_meta = {
            "enabled": bool(retrieval_result),
            "db": str(visual_retrieval_db) if isinstance(visual_retrieval_db, (str, Path)) else None,
            "top_k": visual_retrieval_top_k,
            "target_width": target_width,
            "target_height": target_height,
        }
        if retrieval_result:
            vr_meta.update(
                {
                    "neighbors": retrieval_result.get("neighbors", []),
                    "selected_raw_candidates": retrieval_result.get("selected_raw_candidates", {}),
                    "priors": retrieval_result.get("priors", {}),
                    "abs_boxes": retrieval_result.get("abs_boxes", {}),
                    "prior_strategy": retrieval_result.get("prior_strategy", {}),
                    "locked_roles": sorted(locked_roles),
                }
            )
            warnings = []
            for role in ("hero_image", "background_shape"):
                if role not in retrieval_targets:
                    warnings.append(f"missing_candidate:{role}")
            if retrieval_result.get("warning"):
                warnings.append(retrieval_result["warning"])
            if warnings:
                vr_meta["warning"] = "; ".join(warnings)
        else:
            vr_meta["warning"] = "Retrieval requested but no visual retrieval DB was provided."
        banner["metadata"]["visual_retrieval"] = vr_meta

    if gnn_layout_checkpoint:
        banner["metadata"]["gnn_layout"] = {
            "enabled": bool(gnn_result),
            "checkpoint": str(gnn_layout_checkpoint),
            "target_width": target_width,
            "target_height": target_height,
            "target_roles": (gnn_result or {}).get("target_roles", []),
            "priors": (gnn_result or {}).get("priors", {}),
            "abs_boxes": {role: item["box"] for role, item in gnn_targets.items()},
            "applied_roles": sorted(gnn_targets),
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
    parser = argparse.ArgumentParser(description="Convert semantic/raw Figma JSON to target layout.")
    parser.add_argument("input_json")
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    parser.add_argument("output_json")
    parser.add_argument("--visual-retrieval-db", default=None)
    parser.add_argument("--visual-retrieval-top-k", type=int, default=15)
    parser.add_argument("--visual-mode", choices=["default", "retrieval"], default="default")
    parser.add_argument("--gnn-layout-checkpoint", default=None)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    target_w = int(args.width)
    target_h = int(args.height)
    output_path = Path(args.output_json)

    source = load_input(input_path)
    converted = convert_banner(
        source,
        target_w,
        target_h,
        visual_retrieval_db=args.visual_retrieval_db,
        visual_retrieval_top_k=args.visual_retrieval_top_k,
        visual_mode=args.visual_mode,
        gnn_layout_checkpoint=args.gnn_layout_checkpoint,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Saved converted banner to {output_path}")
    print(f"Layout metadata: {converted.get('metadata', {}).get('layout_engine', {})}")


if __name__ == "__main__":
    main()