"""Build directed source-target training pairs from clean semantic banner JSON."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .family import get_family_key
from .semantic_utils import get_banner_size


def load_banners(input_path: Path) -> list[dict[str, Any]]:
    """Load banners from a JSON file, a list JSON file, or a folder of .json files."""
    if input_path.is_dir():
        banners: list[dict[str, Any]] = []
        for path in sorted(input_path.rglob("*.json")):
            banners.extend(_load_json_path(path))
        return banners
    return _load_json_path(input_path)


def build_pairs(
    banners: list[dict[str, Any]],
    allow_cross_family: bool = False,
    max_pairs_per_family: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for banner in banners:
        groups[get_family_key(banner)].append(banner)

    pairs: list[dict[str, Any]] = []
    rng = random.Random(seed)
    if allow_cross_family:
        for source in banners:
            for target in banners:
                if _banner_id(source) == _banner_id(target):
                    continue
                pairs.append(_make_pair(source, target, "cross_family"))
        if shuffle:
            rng.shuffle(pairs)
        return pairs

    for family_key, family_banners in groups.items():
        if len(family_banners) < 2:
            continue
        family_pairs: list[dict[str, Any]] = []
        for source in family_banners:
            for target in family_banners:
                if _banner_id(source) == _banner_id(target):
                    continue
                family_pairs.append(_make_pair(source, target, family_key))
        if shuffle:
            rng.shuffle(family_pairs)
        if max_pairs_per_family is not None:
            family_pairs = family_pairs[: max(0, max_pairs_per_family)]
        pairs.extend(family_pairs)
    if shuffle:
        rng.shuffle(pairs)
    return pairs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Clean semantic banner JSON file or folder")
    parser.add_argument("--output", required=True, type=Path, help="Output pairs.jsonl path")
    parser.add_argument("--allow-cross-family", action="store_true", help="Create directed pairs across all families")
    parser.add_argument("--max-pairs-per-family", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    banners = load_banners(args.input)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for banner in banners:
        groups[get_family_key(banner)].append(banner)
    pairs = build_pairs(
        banners,
        allow_cross_family=args.allow_cross_family,
        max_pairs_per_family=args.max_pairs_per_family,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    write_jsonl(args.output, pairs)
    summary = {
        "input": str(args.input),
        "families": len(groups),
        "family_sizes": {key: len(items) for key, items in sorted(groups.items())},
        "total_pairs": len(pairs),
        "skipped_singleton_families": len([items for items in groups.values() if len(items) < 2]),
        "allow_cross_family": args.allow_cross_family,
        "max_pairs_per_family": args.max_pairs_per_family,
        "shuffle": args.shuffle,
        "seed": args.seed,
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    skipped = {key: len(items) for key, items in groups.items() if len(items) < 2}
    print(f"Loaded banners: {len(banners)}")
    print(f"Families: {len(groups)}")
    print("Family sizes:")
    for key, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:50]:
        print(f"  {len(items):4d}  {key[:120]}")
    if skipped:
        print(f"Skipped families with size < 2: {len(skipped)}")
    if args.allow_cross_family:
        print("Pairing mode: cross-family enabled")
    else:
        print("Pairing mode: within-family only")
    print(f"Total pairs: {len(pairs)}")
    print(f"Wrote: {args.output}")
    print(f"Wrote summary: {summary_path}")


def _load_json_path(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"{path} must contain a JSON object or list of objects")


def _make_pair(source: dict[str, Any], target: dict[str, Any], family_key: str) -> dict[str, Any]:
    source_width, source_height = get_banner_size(source)
    target_width, target_height = get_banner_size(target)
    source_id = _banner_id(source)
    target_id = _banner_id(target)
    return {
        "pair_id": f"{source_id}__to__{target_id}",
        "source_id": source_id,
        "target_id": target_id,
        "family_key": family_key,
        "source_width": source_width,
        "source_height": source_height,
        "target_width": target_width,
        "target_height": target_height,
        "source": source,
        "target": target,
    }


def _banner_id(banner: dict[str, Any]) -> str:
    return str(banner.get("id") or banner.get("name") or id(banner))


if __name__ == "__main__":
    main()
