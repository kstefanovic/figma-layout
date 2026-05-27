"""Inspect compact RALF tensor cache metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from layout_training.ralf.dataset import _torch_load


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.glob("*") if p.is_file())


def _fmt_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def inspect_cache(cache_dir: str) -> dict[str, object]:
    root = Path(cache_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing RALF cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_shapes = {}
    shards = manifest.get("shards") or []
    if shards:
        first_shard = _torch_load(root / str(shards[0]["file"]))
        if first_shard:
            sample = first_shard[0]
            ret_rows = sample.get("ret_num") or []
            first_shapes = {
                "src_num_length": len(sample.get("src_num") or []),
                "ret_num_k": len(ret_rows),
                "ret_num_r": [len(row) for row in ret_rows],
                "target_length": len(sample.get("target") or []),
            }
    return {
        "sample_count": manifest.get("sample_count"),
        "shard_count": manifest.get("shard_count"),
        "feature_dim": manifest.get("feature_dim"),
        "size_on_disk": _fmt_bytes(_dir_size(root)),
        "first_sample_shapes": first_shapes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect compact RALF tensor cache metadata.")
    parser.add_argument("--cache", required=True)
    args = parser.parse_args(argv)
    info = inspect_cache(args.cache)
    for key, value in info.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
