"""Split RALF pairs into train/validation sets."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict

from layout_training.pairs import read_jsonl, write_jsonl


def split_rows(rows: list[dict], val_ratio: float, split_by_family: bool, seed: int) -> tuple[list[dict], list[dict], str | None]:
    rng = random.Random(seed)
    warning = None
    if split_by_family:
        by_family: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_family[str(row.get("family_id") or "unknown_family")].append(row)
        fams = list(by_family)
        if len(fams) >= 3:
            rng.shuffle(fams)
            n_val = max(1, round(len(fams) * val_ratio))
            val_fams = set(fams[:n_val])
            train = [r for f, items in by_family.items() if f not in val_fams for r in items]
            val = [r for f, items in by_family.items() if f in val_fams for r in items]
            return train, val, None
        warning = "too_few_families_for_family_split;fell_back_to_pair_split"
    copy = list(rows)
    rng.shuffle(copy)
    n_val = max(1, round(len(copy) * val_ratio)) if len(copy) > 1 else 0
    return copy[n_val:], copy[:n_val], warning


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split RALF pairs into train/val JSONL.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-by-family", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    rows = read_jsonl(args.pairs)
    train, val, warning = split_rows(rows, args.val_ratio, args.split_by_family, args.seed)
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    write_jsonl(args.train, train)
    write_jsonl(args.val, val)
    print(f"train={len(train)} val={len(val)}")
    return 0 if train else 1


if __name__ == "__main__":
    sys.exit(main())

