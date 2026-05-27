"""Build directed source-to-target top-level layout pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from layout_training.pairs import build_pairs, read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build directed source-target layout pairs from records.")
    parser.add_argument("--records", required=True, help="Input records JSONL.")
    parser.add_argument("--families", default=None, help="Optional families.json. Auto-group if missing.")
    parser.add_argument("--output", required=True, help="Output pairs JSONL.")
    parser.add_argument("--min-matched-tokens", type=int, default=3, help="Minimum matched tokens per pair.")
    args = parser.parse_args(argv)

    records = read_jsonl(args.records)
    families = args.families if args.families and Path(args.families).exists() else None
    if args.families and families is None:
        print(f"warning: {args.families} not found; using auto filename grouping", file=sys.stderr)
    pairs = build_pairs(records, families, min_matched_tokens=args.min_matched_tokens)
    write_jsonl(args.output, pairs)
    summary_path = Path(args.output).parent / "pairs_summary.json"
    family_counts: dict[str, int] = {}
    for pair in pairs:
        family_counts[str(pair.get("family_id"))] = family_counts.get(str(pair.get("family_id")), 0) + 1
    summary_path.write_text(
        json.dumps({"record_count": len(records), "pair_count": len(pairs), "family_pair_counts": family_counts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(pairs)} pair(s) to {args.output}; summary={summary_path}")
    return 0 if pairs else 1


if __name__ == "__main__":
    sys.exit(main())

