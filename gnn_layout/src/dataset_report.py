"""Print and optionally save dataset statistics for clean banners and pair JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .family import get_family_key
from .orientation import get_orientation
from .roles import ROLES
from .semantic_utils import extract_role_boxes, extract_role_mask, get_banner_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    banners = _load_banners(args.clean)
    pairs = _load_jsonl(args.pairs)
    report = build_report(banners, pairs)
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Wrote report: {args.output}")


def build_report(banners: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    orientation_counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    family_sizes: Counter[str] = Counter()
    missing_roles: Counter[str] = Counter()
    role_values: dict[str, list[list[float]]] = defaultdict(list)

    for banner in banners:
        try:
            width, height = get_banner_size(banner)
        except ValueError:
            continue
        orientation_counts[get_orientation(width, height)] += 1
        sizes[f"{int(round(width))}x{int(round(height))}"] += 1
        family_sizes[get_family_key(banner)] += 1
        boxes = extract_role_boxes(banner)
        mask = extract_role_mask(banner)
        for idx, role in enumerate(ROLES):
            if mask[idx] > 0:
                role_values[role].append(boxes[idx].tolist())
            else:
                missing_roles[role] += 1

    target_orientation_counts: Counter[str] = Counter()
    for row in pairs:
        try:
            tw = float(row.get("target_width") or 0)
            th = float(row.get("target_height") or 0)
            if tw > 0 and th > 0:
                target_orientation_counts[get_orientation(tw, th)] += 1
        except (TypeError, ValueError):
            continue

    return {
        "clean_banners": len(banners),
        "orientation_distribution": dict(sorted(orientation_counts.items())),
        "size_distribution": dict(sorted(sizes.items())),
        "family_count": len(family_sizes),
        "family_sizes": dict(sorted(family_sizes.items(), key=lambda kv: (-kv[1], kv[0]))),
        "total_pairs": len(pairs),
        "target_orientation_distribution": dict(sorted(target_orientation_counts.items())),
        "missing_role_count": {role: int(missing_roles.get(role, 0)) for role in ROLES},
        "role_bbox_stats": _role_bbox_stats(role_values),
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Clean banners: {report['clean_banners']}")
    print(f"Families: {report['family_count']}")
    print(f"Total pairs: {report['total_pairs']}")
    print(f"Orientation distribution: {report['orientation_distribution']}")
    print(f"Target orientation distribution: {report['target_orientation_distribution']}")
    print("Top sizes:")
    for size, count in list(report["size_distribution"].items())[:20]:
        print(f"  {count:5d}  {size}")
    print("Top family sizes:")
    for family, count in list(report["family_sizes"].items())[:20]:
        print(f"  {count:5d}  {family[:120]}")
    print("Missing role count:")
    for role, count in report["missing_role_count"].items():
        print(f"  {role}: {count}")
    print("Role bbox min/max/mean:")
    for role, stats in report["role_bbox_stats"].items():
        print(f"  {role}: {stats}")


def _role_bbox_stats(role_values: dict[str, list[list[float]]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for role in ROLES:
        values = role_values.get(role) or []
        if not values:
            stats[role] = {"count": 0, "min": None, "max": None, "mean": None}
            continue
        arr = np.asarray(values, dtype=np.float32)
        stats[role] = {
            "count": int(arr.shape[0]),
            "min": arr.min(axis=0).round(6).tolist(),
            "max": arr.max(axis=0).round(6).tolist(),
            "mean": arr.mean(axis=0).round(6).tolist(),
        }
    return stats


def _load_banners(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"{path} must contain a JSON object or list")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


if __name__ == "__main__":
    main()
