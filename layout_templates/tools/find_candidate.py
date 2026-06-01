#!/usr/bin/env python3

import argparse
import copy
import json
import re
from pathlib import Path


CLASS_ORDER = (
    "portrait",
    "landscape-narrow",
    "landscape-wide",
    "wide",
    "super_wide",
)


def classify_aspect_ratio(aspect_ratio: float) -> str:
    if aspect_ratio < 1:
        return "portrait"
    if aspect_ratio < 1.7:
        return "landscape-narrow"
    if aspect_ratio < 3:
        return "landscape-wide"
    if aspect_ratio < 6:
        return "wide"
    return "super_wide"


def similarity_score(
    input_width: float,
    input_height: float,
    candidate_width: float,
    candidate_height: float,
) -> float:
    input_ratio = input_width / input_height
    candidate_ratio = candidate_width / candidate_height

    ratio_distance = abs(input_ratio - candidate_ratio) / max(input_ratio, candidate_ratio)
    width_distance = abs(input_width - candidate_width) / max(input_width, candidate_width)
    height_distance = abs(input_height - candidate_height) / max(input_height, candidate_height)

    return (0.5 * ratio_distance) + (0.25 * width_distance) + (0.25 * height_distance)


def get_root_dimensions(banner: dict) -> tuple[float, float] | None:
    bounds = banner.get("bounds") or {}
    width = banner.get("width") or bounds.get("width")
    height = banner.get("height") or bounds.get("height")
    if not width or not height:
        return None
    return float(width), float(height)


def size_similarity(
    input_width: float,
    input_height: float,
    candidate_width: float,
    candidate_height: float,
) -> float:
    distance = similarity_score(
        input_width=input_width,
        input_height=input_height,
        candidate_width=candidate_width,
        candidate_height=candidate_height,
    )
    return max(0.0, 1.0 - distance)


def load_input_banner(input_json_path: Path) -> dict:
    with input_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"Input json is an empty list: {input_json_path}")
        return payload[0]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported input json format in: {input_json_path}")


def get_top_child_names(banner: dict) -> list[str]:
    children = banner.get("children", [])
    names = []
    for child in children:
        if isinstance(child, dict):
            names.append(child.get("name", ""))
    return names


def collect_text_char_count(node: dict) -> int:
    total = 0
    if not isinstance(node, dict):
        return 0
    if node.get("type") == "text":
        total += len(node.get("characters", "") or "")
    for child in node.get("children", []) or []:
        total += collect_text_char_count(child)
    return total


def legal_text_char_count(banner: dict) -> int:
    total = 0
    for child in banner.get("children", []) or []:
        if not isinstance(child, dict):
            continue
        child_name = (child.get("name") or "").lower()
        if "legal" in child_name:
            total += collect_text_char_count(child)
    return total


def top_children_similarity(input_names: list[str], candidate_names: list[str]) -> float:
    input_set = set(input_names)
    candidate_set = set(candidate_names)
    if not input_set and not candidate_set:
        return 1.0
    union = input_set | candidate_set
    if not union:
        return 1.0
    return len(input_set & candidate_set) / len(union)


def legal_similarity(input_legal_chars: int, candidate_legal_chars: int) -> float:
    if input_legal_chars <= 0 or candidate_legal_chars <= 0:
        return 0.0
    diff = abs(input_legal_chars - candidate_legal_chars)
    return 1.0 - (diff / max(input_legal_chars, candidate_legal_chars))


def sanitize_for_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def create_preview_svg(output_path: Path, banner: dict) -> None:
    bounds = banner.get("bounds", {}) or {}
    frame_x = float(bounds.get("x", 0))
    frame_y = float(bounds.get("y", 0))
    frame_w = float(bounds.get("width", 0))
    frame_h = float(bounds.get("height", 0))
    if frame_w <= 0 or frame_h <= 0:
        return

    canvas_w = frame_w * 3.0
    canvas_h = frame_h * 3.0
    offset_x = frame_w
    offset_y = frame_h

    root_name = (banner.get("name") or "banner_root").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        f'  <rect x="0" y="0" width="{canvas_w}" height="{canvas_h}" fill="#dbeafe"/>',
        f'  <rect x="{offset_x}" y="{offset_y}" width="{frame_w}" height="{frame_h}" fill="none" stroke="#0f172a" stroke-width="2.5"/>',
        f'  <text x="{offset_x + 4}" y="{offset_y + 16}" font-family="Arial" font-size="13" fill="#0f172a">{root_name}</text>',
    ]

    for child in banner.get("children", []) or []:
        if not isinstance(child, dict):
            continue
        cb = child.get("bounds", {}) or {}
        cx = (float(cb.get("x", 0)) - frame_x) + offset_x
        cy = (float(cb.get("y", 0)) - frame_y) + offset_y
        cw = float(cb.get("width", 0))
        ch = float(cb.get("height", 0))
        if cw <= 0 or ch <= 0:
            continue
        name = (child.get("name") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(f'  <rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" fill="none" stroke="#e74c3c" stroke-width="1.5"/>')
        lines.append(
            f'  <text x="{cx + 2}" y="{cy + 14}" font-family="Arial" font-size="12" fill="#1f2937">{name}</text>'
        )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def is_background_gradient(name: str) -> bool:
    return (name or "").lower().startswith("background_gradient_")


def build_merged_banner(input_banner: dict, candidate_banner: dict) -> dict:
    merged = copy.deepcopy(input_banner)
    input_children = merged.get("children", []) or []
    candidate_children = candidate_banner.get("children", []) or []

    candidate_by_name = {}
    for child in candidate_children:
        if not isinstance(child, dict):
            continue
        child_name = child.get("name", "")
        candidate_by_name.setdefault(child_name, []).append(child)

    input_names_lower = {(child.get("name", "") or "").lower() for child in input_children if isinstance(child, dict)}
    input_has_discount = "discount_badge_group" in input_names_lower
    input_has_background_gradient = any(is_background_gradient(name) for name in input_names_lower)
    candidate_has_background_gradient = any(
        is_background_gradient(child.get("name", ""))
        for child in candidate_children
        if isinstance(child, dict)
    )

    # For each input top child: replace bounds from candidate child of the same name.
    # If decoration_group exists in input, replace that child completely with candidate's version.
    for idx, input_child in enumerate(input_children):
        if not isinstance(input_child, dict):
            continue
        name = input_child.get("name", "")
        candidate_list = candidate_by_name.get(name, [])
        if not candidate_list:
            continue
        candidate_child = candidate_list.pop(0)

        if (name or "").lower() == "decoration_group":
            input_children[idx] = copy.deepcopy(candidate_child)
            continue

        if "bounds" in candidate_child:
            input_child["bounds"] = copy.deepcopy(candidate_child.get("bounds", {}))

    # Add candidate background_gradient_* only when input has none.
    if not input_has_background_gradient and candidate_has_background_gradient:
        for child in candidate_children:
            if not isinstance(child, dict):
                continue
            child_name = child.get("name", "")
            if is_background_gradient(child_name):
                input_children.append(copy.deepcopy(child))

    # Drop background_gradient_* when the selected candidate does not have them.
    if not candidate_has_background_gradient:
        input_children = [
            child
            for child in input_children
            if not isinstance(child, dict) or not is_background_gradient(child.get("name", ""))
        ]

    # Do not copy discount_badge_group from candidate if input does not have it.
    if not input_has_discount:
        filtered_children = []
        for child in input_children:
            if not isinstance(child, dict):
                filtered_children.append(child)
                continue
            if (child.get("name", "") or "").lower() == "discount_badge_group":
                continue
            filtered_children.append(child)
        merged["children"] = filtered_children
    else:
        merged["children"] = input_children

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the closest candidate banner by class and size."
    )
    parser.add_argument("--cat", required=True, help="Category folder name under templates/")
    parser.add_argument("--width", required=True, type=float, help="Input banner width")
    parser.add_argument("--height", required=True, type=float, help="Input banner height")
    parser.add_argument(
        "--json",
        required=True,
        help="Path to input json used for top-children similarity and preview output location",
    )
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("Both --width and --height must be positive numbers.")

    input_class = classify_aspect_ratio(args.width / args.height)

    templates_root = Path(__file__).resolve().parent.parent / "templates"
    category_dir = templates_root / args.cat
    if not category_dir.is_dir():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")

    class_file = category_dir / f"{input_class}.json"
    if not class_file.is_file():
        raise FileNotFoundError(f"Class file not found: {class_file}")

    with class_file.open("r", encoding="utf-8") as f:
        candidates = json.load(f)

    if not candidates:
        print(f"No candidates found in class '{input_class}' for category '{args.cat}'.")
        return

    input_json_path = Path(args.json).expanduser().resolve()
    if not input_json_path.is_file():
        raise FileNotFoundError(f"Input json file not found: {input_json_path}")
    input_banner = load_input_banner(input_json_path)
    input_top_children = get_top_child_names(input_banner)
    input_legal_chars = legal_text_char_count(input_banner)

    scored_by_size = []
    for candidate in candidates:
        candidate_width = candidate.get("width")
        candidate_height = candidate.get("height")
        if not candidate_width or not candidate_height:
            continue

        size_score = similarity_score(
            input_width=args.width,
            input_height=args.height,
            candidate_width=float(candidate_width),
            candidate_height=float(candidate_height),
        )
        scored_by_size.append((size_score, candidate))

    if not scored_by_size:
        print(f"No valid candidates found in class '{input_class}' for category '{args.cat}'.")
        return

    scored_by_size.sort(key=lambda x: x[0])
    top5 = [item[1] for item in scored_by_size[:5]]

    full_templates_file = templates_root / f"{args.cat}.json"
    if not full_templates_file.is_file():
        raise FileNotFoundError(f"Full category json not found: {full_templates_file}")
    with full_templates_file.open("r", encoding="utf-8") as f:
        full_templates = json.load(f)
    full_by_id = {item.get("id"): item for item in full_templates if isinstance(item, dict)}

    final_best = None
    final_best_score = None
    for candidate in top5:
        candidate_id = candidate.get("id")
        candidate_full = full_by_id.get(candidate_id)
        if not candidate_full:
            continue
        candidate_top_children = get_top_child_names(candidate_full)
        candidate_legal_chars = legal_text_char_count(candidate_full)

        top_children_score = top_children_similarity(input_top_children, candidate_top_children)
        legal_score = legal_similarity(input_legal_chars, candidate_legal_chars)

        root_dims = get_root_dimensions(candidate_full)
        if root_dims is None:
            candidate_width = candidate.get("width")
            candidate_height = candidate.get("height")
            if candidate_width and candidate_height:
                root_dims = (float(candidate_width), float(candidate_height))

        if root_dims is None:
            size_score = 0.0
        else:
            size_score = size_similarity(
                input_width=args.width,
                input_height=args.height,
                candidate_width=root_dims[0],
                candidate_height=root_dims[1],
            )

        combined = (0.6 * top_children_score) + (0.2 * legal_score) + (0.2 * size_score)

        if final_best is None or combined > final_best_score:
            final_best = candidate_full
            final_best_score = combined

    if final_best is None:
        print("Could not determine final candidate from top 5 candidates.")
        return

    candidate_id = final_best.get("id", "unknown")
    final_json = build_merged_banner(input_banner=input_banner, candidate_banner=final_best)
    candidate_dims = get_root_dimensions(final_best)
    if candidate_dims is not None:
        candidate_width, candidate_height = candidate_dims
        final_json["width"] = candidate_width
        final_json["height"] = candidate_height
        root_bounds = final_json.setdefault("bounds", {})
        if isinstance(root_bounds, dict):
            root_bounds["width"] = candidate_width
            root_bounds["height"] = candidate_height
    final_json_name = f"final_candidate_{sanitize_for_filename(str(candidate_id))}.json"
    final_json_path = input_json_path.parent / final_json_name
    final_json_path.write_text(json.dumps(final_json, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_json_name = f"selected_candidate_{sanitize_for_filename(str(candidate_id))}.json"
    selected_json_path = input_json_path.parent / selected_json_name
    selected_json_path.write_text(json.dumps(final_best, ensure_ascii=False, indent=2), encoding="utf-8")

    preview_name = f"candidate_preview_{sanitize_for_filename(str(candidate_id))}.svg"
    preview_path = input_json_path.parent / preview_name
    create_preview_svg(preview_path, final_json)

    print(f"category: {args.cat}")
    print(f"class: {input_class}")
    print(f"candidate_id: {final_best.get('id')}")
    print(f"candidate_width: {(final_best.get('bounds') or {}).get('width')}")
    print(f"candidate_height: {(final_best.get('bounds') or {}).get('height')}")
    print(f"final_json_file: {final_json_path}")
    print(f"selected_json_file: {selected_json_path}")
    print(f"preview_svg: {preview_path}")
    print("final_json:")
    print(json.dumps(final_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
