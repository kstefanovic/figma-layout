"""Inspect compact RALF retrieval index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from layout_training.ralf.retrieval import load_compact_retrieval_index


def inspect_index(path: str) -> dict[str, object]:
    index = load_compact_retrieval_index(path)
    first = (index.get("entries") or [{}])[0]
    return {
        "path": str(Path(path)),
        "index_type": index.get("index_type"),
        "record_count": index.get("record_count"),
        "role_count": index.get("role_count"),
        "first_record_id": first.get("record_id"),
        "first_canvas": first.get("canvas"),
        "first_token_count": len(first.get("tokens") or []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect compact RALF retrieval index.")
    parser.add_argument("--index", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(inspect_index(args.index), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
