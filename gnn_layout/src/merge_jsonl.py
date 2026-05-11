"""Merge multiple pair JSONL files into one deduplicated training JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dedupe", action="store_true")
    args = parser.parse_args()

    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    input_rows = 0
    invalid_rows = 0
    duplicates_removed = 0

    for path in args.inputs:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                input_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid_rows += 1
                    continue
                if not _valid_pair_row(row):
                    invalid_rows += 1
                    continue
                key = _dedupe_key(row)
                if args.dedupe and key in seen:
                    duplicates_removed += 1
                    continue
                seen.add(key)
                output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "input_files": [str(path) for path in args.inputs],
        "input_rows": input_rows,
        "output_rows": len(output_rows),
        "duplicates_removed": duplicates_removed,
        "invalid_rows": invalid_rows,
        "dedupe": args.dedupe,
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Input files: {len(args.inputs)}")
    print(f"Input rows: {input_rows}")
    print(f"Output rows: {len(output_rows)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Invalid rows: {invalid_rows}")
    print(f"Wrote merged JSONL: {args.output}")
    print(f"Wrote summary: {summary_path}")


def _valid_pair_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if not isinstance(row.get("source"), dict) or not isinstance(row.get("target"), dict):
        return False
    try:
        return float(row.get("target_width") or 0) > 0 and float(row.get("target_height") or 0) > 0
    except (TypeError, ValueError):
        return False


def _dedupe_key(row: dict[str, Any]) -> str:
    pair_id = str(row.get("pair_id") or "").strip()
    if pair_id:
        return pair_id
    return "|".join(
        [
            str(row.get("source_id") or ""),
            str(row.get("target_id") or ""),
            str(row.get("family_key") or ""),
        ]
    )


if __name__ == "__main__":
    main()
