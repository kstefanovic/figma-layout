"""Build canonical top-level layout records JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from layout_training.records import build_record_from_semantic_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert semantic JSON files into canonical top-level layout records.")
    parser.add_argument("--input", required=True, help="Directory containing semantic JSON files.")
    parser.add_argument("--output", required=True, help="Output records JSONL path.")
    parser.add_argument("--include-raw-json", action="store_true", help="Include full raw JSON in records.")
    args = parser.parse_args(argv)

    input_dir = Path(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.parent / "summary.json"

    records = []
    skipped = []
    role_counts: Counter[str] = Counter()
    for path in sorted(input_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rec = build_record_from_semantic_json(data, file_id=str(path), include_raw_json=args.include_raw_json)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            skipped.append({"file": str(path), "error": str(exc)})
            print(f"warning: skipped {path}: {exc}", file=sys.stderr)
            continue
        records.append(rec)
        role_counts.update(token.get("train_role") for token in rec.get("tokens") or [])

    with output.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "input": str(input_dir),
        "output": str(output),
        "record_count": len(records),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "token_role_counts": dict(role_counts),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} record(s) to {output}; summary={summary_path}")
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())

