"""Split layout pairs into train/validation JSONL files."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict

from layout_training.pairs import read_jsonl, write_jsonl


def split_pairs(pairs: list[dict], val_ratio: float, split_by_family: bool, seed: int) -> tuple[list[dict], list[dict], str | None]:
    rng = random.Random(seed)
    warning = None
    if split_by_family:
        by_family: dict[str, list[dict]] = defaultdict(list)
        for pair in pairs:
            by_family[str(pair.get("family_id"))].append(pair)
        families = list(by_family)
        if len(families) >= 3:
            rng.shuffle(families)
            val_family_count = max(1, round(len(families) * val_ratio))
            val_families = set(families[:val_family_count])
            train = [p for fam, rows in by_family.items() if fam not in val_families for p in rows]
            val = [p for fam, rows in by_family.items() if fam in val_families for p in rows]
            return train, val, None
        warning = "too_few_families_for_family_split;fell_back_to_pair_split"
    rows = list(pairs)
    rng.shuffle(rows)
    val_count = max(1, round(len(rows) * val_ratio)) if len(rows) > 1 else 0
    return rows[val_count:], rows[:val_count], warning


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split top-level layout pairs into train/val sets.")
    parser.add_argument("--pairs", required=True, help="Input pairs JSONL.")
    parser.add_argument("--train", required=True, help="Output train JSONL.")
    parser.add_argument("--val", required=True, help="Output val JSONL.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-by-family", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    pairs = read_jsonl(args.pairs)
    train, val, warning = split_pairs(pairs, args.val_ratio, args.split_by_family, args.seed)
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    write_jsonl(args.train, train)
    write_jsonl(args.val, val)
    print(f"train={len(train)} val={len(val)}")
    return 0 if train else 1


if __name__ == "__main__":
    sys.exit(main())

