"""Validate collected top-level semantic JSONs.

Usage:
  python -m layout_training.tools.validate_top_level_semantic_jsons --help
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from layout_training.geometry import bbox_xywh, safe_float
from layout_training.records import build_record_from_semantic_json
from layout_training.roles import COMMON_RAW_ROLE_GROUPS, is_known_raw_role, train_role_for


def _root(json_obj: Any) -> dict[str, Any] | None:
    if isinstance(json_obj, list):
        return json_obj[0] if json_obj and isinstance(json_obj[0], dict) else None
    return json_obj if isinstance(json_obj, dict) else None


def validate_file(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    role_counts: Counter[str] = Counter()
    unknown_roles: Counter[str] = Counter()
    duplicate_roles: dict[str, int] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"file": str(path), "valid": False, "errors": [f"invalid_json:{exc}"], "warnings": []}

    root = _root(data)
    if root is None:
        errors.append("root_missing")
    else:
        try:
            build_record_from_semantic_json(data, file_id=str(path))
        except ValueError as exc:
            errors.append(str(exc))
        children = root.get("children")
        if not isinstance(children, list):
            errors.append("root_children_missing")
            children = []
        bounds = root.get("bounds") if isinstance(root.get("bounds"), dict) else {}
        w = safe_float(root.get("width"), safe_float(bounds.get("width")))
        h = safe_float(root.get("height"), safe_float(bounds.get("height")))
        if w <= 0 or h <= 0:
            errors.append("root_bounds_width_height_missing")
        raw_roles: set[str] = set()
        train_roles: set[str] = set()
        for idx, child in enumerate(children):
            if not isinstance(child, dict):
                warnings.append(f"child_{idx}_not_object")
                continue
            x, y, cw, ch = bbox_xywh(child)
            if cw <= 0 or ch <= 0:
                warnings.append(f"child_{idx}_zero_size")
            role = str(child.get("semantic_name") or child.get("name") or "").strip()
            if not role:
                warnings.append(f"child_{idx}_missing_semantic_name")
                role = "unknown_group"
            role_counts[role] += 1
            raw_roles.add(role)
            train_roles.add(train_role_for(role))
            if not is_known_raw_role(role):
                unknown_roles[role] += 1
        duplicate_roles = {role: count for role, count in role_counts.items() if count > 1}
        for common_key, raw_group in COMMON_RAW_ROLE_GROUPS.items():
            if common_key == "text_main_group":
                if "text_main_group" not in train_roles:
                    warnings.append("missing_common_role:text_main_group")
            elif raw_roles.isdisjoint(raw_group):
                warnings.append(f"missing_common_role:{common_key}")

    return {
        "file": str(path),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "role_counts": dict(role_counts),
        "duplicate_role_stats": duplicate_roles,
        "unknown_roles": dict(unknown_roles),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate top-level semantic Figma JSON files.")
    parser.add_argument("--input", required=True, help="Directory containing semantic JSON files.")
    parser.add_argument("--report", required=True, help="Output validation report JSON.")
    args = parser.parse_args(argv)

    input_dir = Path(args.input)
    files = sorted(input_dir.rglob("*.json"))
    results = [validate_file(path) for path in files]
    valid_count = sum(1 for r in results if r.get("valid"))
    report = {
        "input": str(input_dir),
        "file_count": len(files),
        "valid_count": valid_count,
        "invalid_count": len(files) - valid_count,
        "files": results,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"validated {len(files)} file(s), valid={valid_count}, report={out}")
    return 0 if valid_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

