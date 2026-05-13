"""Prepare clean GNN training pairs for brand/headline/legal layout transfer."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build_pairs import build_pairs
from .clean_filter import is_clean_banner
from .extract_clean import load_candidate_banners, write_json, write_jsonl
from .family import get_family_key
from .semantic_utils import get_role_box_norm


TARGET_ROLES = ["brand_group", "headline_group", "legal_text"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("gnn_layout/data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("gnn_layout/data/text_role_dataset"))
    parser.add_argument("--max-pairs-per-family", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = prepare_text_role_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        max_pairs_per_family=args.max_pairs_per_family,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print(f"Candidates: {result['summary']['total_candidates']}")
    print(f"Clean semantic banners: {result['summary']['clean_count']}")
    print(f"Training pairs: {result['summary']['pair_count']}")
    print(f"Wrote: {result['clean_path']}")
    print(f"Wrote: {result['pairs_path']}")
    print(f"Wrote: {result['summary_path']}")


def prepare_text_role_dataset(
    input_path: Path,
    output_dir: Path,
    max_pairs_per_family: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    candidates = load_candidate_banners(input_path)
    clean: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for banner, source_file in candidates:
        ok, reasons = is_clean_banner(banner, strict=True)
        if ok and _has_target_roles(banner):
            clean.append(banner)
        else:
            if ok:
                reasons = ["missing_text_role_target"]
            for reason in reasons:
                reason_counts[reason] += 1
            rejects.append(_reject_row(banner, reasons, source_file))

    pairs = build_pairs(
        clean,
        allow_cross_family=False,
        max_pairs_per_family=max_pairs_per_family,
        shuffle=shuffle,
        seed=seed,
    )
    text_pairs = [_text_role_pair(row) for row in pairs if _pair_has_target_labels(row)]

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for banner in clean:
        families[get_family_key(banner)].append(banner)

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / "clean_semantic_banners.json"
    pairs_path = output_dir / "pairs_brand_headline_legal.jsonl"
    rejects_path = output_dir / "rejects.jsonl"
    summary_path = output_dir / "summary.json"

    summary = {
        "source_input": str(input_path),
        "target_roles": TARGET_ROLES,
        "total_candidates": len(candidates),
        "clean_count": len(clean),
        "reject_count": len(rejects),
        "pair_count": len(text_pairs),
        "families": len(families),
        "family_sizes": {k: len(v) for k, v in sorted(families.items())},
        "reject_reason_counts": dict(sorted(reason_counts.items())),
        "max_pairs_per_family": max_pairs_per_family,
        "shuffle": shuffle,
        "seed": seed,
    }

    write_json(clean_path, clean)
    write_jsonl(pairs_path, text_pairs)
    write_jsonl(rejects_path, rejects)
    write_json(summary_path, summary)

    return {
        "clean_path": str(clean_path),
        "pairs_path": str(pairs_path),
        "rejects_path": str(rejects_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def _has_target_roles(banner: dict[str, Any]) -> bool:
    return all(get_role_box_norm(banner, role) is not None for role in TARGET_ROLES)


def _pair_has_target_labels(row: dict[str, Any]) -> bool:
    target = row.get("target")
    return isinstance(target, dict) and _has_target_roles(target)


def _text_role_pair(row: dict[str, Any]) -> dict[str, Any]:
    target = row["target"]
    return {
        **row,
        "target_roles": TARGET_ROLES,
        "target_role_boxes": {
            role: get_role_box_norm(target, role)
            for role in TARGET_ROLES
        },
    }


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

