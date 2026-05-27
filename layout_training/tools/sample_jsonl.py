"""Sample N rows from JSONL for smoke/medium runs."""

from __future__ import annotations

import argparse
import random
import sys

from layout_training.pairs import read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample N random rows from a JSONL file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    rows = read_jsonl(args.input)
    if not rows:
        print("error: input jsonl has no rows", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)
    take = min(args.n, len(rows))
    sampled = rng.sample(rows, take) if take < len(rows) else rows
    write_jsonl(args.output, sampled)
    print(f"wrote {len(sampled)} row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

