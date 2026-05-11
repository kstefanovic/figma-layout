"""Extract strict clean semantic banner frames from mixed JSON exports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .clean_filter import is_clean_banner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Mixed JSON file or folder")
    parser.add_argument("--output", required=True, type=Path, help="Clean banner JSON list")
    parser.add_argument("--rejects", required=True, type=Path, help="Rejected banner JSONL report")
    parser.add_argument("--report", required=True, type=Path, help="Summary report JSON")
    parser.add_argument("--strict", default="true", choices=["true", "false"], help="Use strict clean checks")
    args = parser.parse_args()

    strict = args.strict.lower() == "true"
    candidates = load_candidate_banners(args.input)
    clean: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for banner, source_file in candidates:
        ok, reasons = is_clean_banner(banner, strict=strict)
        if ok:
            clean.append(banner)
            continue
        for reason in reasons:
            reason_counts[reason] += 1
        rejects.append(_reject_row(banner, reasons, source_file))

    write_json(args.output, clean)
    write_jsonl(args.rejects, rejects)
    report = {
        "total_candidates": len(candidates),
        "clean_count": len(clean),
        "reject_count": len(rejects),
        "reject_reason_counts": dict(sorted(reason_counts.items())),
        "source_files": sorted({source for _, source in candidates}),
        "strict": strict,
    }
    write_json(args.report, report)

    print(f"Candidates: {len(candidates)}")
    print(f"Clean: {len(clean)}")
    print(f"Rejected: {len(rejects)}")
    print("Top reject reasons:")
    for reason, count in reason_counts.most_common(20):
        print(f"  {count:5d}  {reason}")
    print(f"Wrote clean banners: {args.output}")
    print(f"Wrote rejects: {args.rejects}")
    print(f"Wrote report: {args.report}")


def load_candidate_banners(input_path: Path) -> list[tuple[dict[str, Any], str]]:
    paths = sorted(input_path.rglob("*.json")) if input_path.is_dir() else [input_path]
    candidates: list[tuple[dict[str, Any], str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in _walk_json(data):
            if _is_candidate_banner(item):
                candidates.append((item, str(path)))
    return candidates


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.get("children") or []:
            yield from _walk_json(child)
        for key, child in value.items():
            if key == "children":
                continue
            if isinstance(child, (dict, list)):
                yield from _walk_json(child)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)


def _is_candidate_banner(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if str(item.get("type") or "").lower() != "frame":
        return False
    bounds = item.get("bounds")
    if not isinstance(bounds, dict):
        return False
    try:
        return float(bounds.get("width") or 0) > 0 and float(bounds.get("height") or 0) > 0
    except (TypeError, ValueError):
        return False


def _reject_row(banner: dict[str, Any], reasons: list[str], source_file: str) -> dict[str, Any]:
    bounds = banner.get("bounds") if isinstance(banner.get("bounds"), dict) else {}
    return {
        "id": str(banner.get("id") or ""),
        "name": str(banner.get("name") or ""),
        "width": bounds.get("width"),
        "height": bounds.get("height"),
        "reasons": reasons,
        "source_file": source_file,
    }


if __name__ == "__main__":
    main()
