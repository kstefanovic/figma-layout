"""Build compact RALF retrieval index from top-level layout records."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from layout_training.ralf.retrieval import build_compact_retrieval_index


def _loads(line: str) -> Any:
    try:
        import orjson
    except ImportError:
        return json.loads(line)
    return orjson.loads(line)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = _loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def build_retrieval_index(records_path: str, output_path: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to build RALF retrieval index.") from exc

    records = list(_iter_jsonl(Path(records_path)))
    index = build_compact_retrieval_index(records, records_path=records_path)
    index["created_at"] = datetime.now(timezone.utc).isoformat()
    index["source_records"] = str(Path(records_path))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, str(out))
    return {
        "output": str(out),
        "record_count": index.get("record_count"),
        "role_count": index.get("role_count"),
        "index_type": "compact_pt",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build compact RALF retrieval index.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_retrieval_index(args.records, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
