"""Build RALF pairs with retrieval context attached."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from layout_training.ralf.ralf_pairs import build_ralf_pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build retrieval-augmented RALF pair JSONL.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--exclude-same-family", action="store_true")
    parser.add_argument("--exclude-target-id", action="store_true")
    args = parser.parse_args(argv)
    result = build_ralf_pairs(
        records_path=args.records,
        pairs_path=args.pairs,
        output_path=args.output,
        retrieval_k=args.retrieval_k,
        exclude_same_family=args.exclude_same_family,
        exclude_target_id=args.exclude_target_id,
    )
    summary = Path(args.output).parent / "ralf_pairs_summary.json"
    summary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {result['pair_count']} ralf pair(s) to {args.output}")
    return 0 if result["pair_count"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

