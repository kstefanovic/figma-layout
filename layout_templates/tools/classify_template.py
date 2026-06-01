#!/usr/bin/env python3

import argparse
import json
from collections import Counter
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


def to_compact_banner_record(banner: dict, width: float, height: float) -> dict:
    children = banner.get("children", [])
    top_child_names = [child.get("name", "") for child in children if isinstance(child, dict)]
    return {
        "id": banner.get("id"),
        "width": width,
        "height": height,
        "top_child_count": len(top_child_names),
        "top_child_names": top_child_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify banner templates by aspect ratio."
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent / "templates" / "sausage.json"),
        help="Path to templates JSON file (default: layout_templates/templates/sausage.json)",
    )
    args = parser.parse_args()

    templates_path = Path(args.json_file).expanduser().resolve()

    with templates_path.open("r", encoding="utf-8") as f:
        templates = json.load(f)

    counts = Counter({class_name: 0 for class_name in CLASS_ORDER})
    class_buckets = {class_name: [] for class_name in CLASS_ORDER}
    total_count = 0

    for banner in templates:
        bounds = banner.get("bounds", {})
        width = bounds.get("width")
        height = bounds.get("height")

        if not width or not height:
            continue

        aspect_ratio = width / height
        category = classify_aspect_ratio(aspect_ratio)
        counts[category] += 1
        class_buckets[category].append(to_compact_banner_record(banner, width, height))
        total_count += 1

    output_dir = templates_path.parent / templates_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_ORDER:
        class_file_path = output_dir / f"{class_name}.json"
        with class_file_path.open("w", encoding="utf-8") as f:
            json.dump(class_buckets[class_name], f, ensure_ascii=False, indent=2)

    print(f"Total count: {total_count}")
    print("Class counts:")
    for key in CLASS_ORDER:
        print(f"{key}: {counts[key]}")
    print(f"Saved class files under: {output_dir}")


if __name__ == "__main__":
    main()
