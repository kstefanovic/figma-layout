"""Build compact tensor-ready RALF layout cache shards."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from layout_training.ralf.dataset import RALF_NUMERIC_FEATURE_DIM, preprocess_ralf_pair_for_cache


def _loads(line: str) -> Any:
    try:
        import orjson
    except ImportError:
        return json.loads(line)
    return orjson.loads(line)


def _torch_save(obj: Any, path: Path, compression: str) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to build RALF tensor cache.") from exc
    if compression == "gzip":
        with gzip.open(path, "wb") as fh:
            torch.save(obj, fh)
    else:
        with path.open("wb") as fh:
            torch.save(obj, fh)


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


def _flush_shard(samples: list[dict[str, Any]], output_dir: Path, shard_id: int, compression: str) -> dict[str, Any]:
    suffix = ".pt.gz" if compression == "gzip" else ".pt"
    file_name = f"shard_{shard_id:05d}{suffix}"
    _torch_save(samples, output_dir / file_name, compression)
    return {"file": file_name, "samples": len(samples)}


def build_cache(
    *,
    input_path: str,
    output_dir: str,
    shard_size: int,
    max_samples: int | None = None,
    seed: int | None = None,
    compression: str = "none",
) -> dict[str, Any]:
    source = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    if compression not in {"none", "gzip"}:
        raise ValueError("--compression must be one of: none, gzip")

    shards: list[dict[str, Any]] = []
    shard: list[dict[str, Any]] = []
    sample_count = 0
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] | None = [] if seed is not None and max_samples is not None else None

    for seen, pair in enumerate(_iter_jsonl(source), start=1):
        sample = preprocess_ralf_pair_for_cache(pair)
        if reservoir is not None:
            assert max_samples is not None
            if len(reservoir) < max_samples:
                reservoir.append(sample)
            else:
                slot = rng.randrange(seen)
                if slot < max_samples:
                    reservoir[slot] = sample
            continue

        if max_samples is not None and sample_count >= max_samples:
            break
        shard.append(sample)
        sample_count += 1
        if len(shard) >= shard_size:
            shards.append(_flush_shard(shard, out, len(shards), compression))
            print(f"wrote {shards[-1]['file']} samples={shards[-1]['samples']}")
            shard = []

    if reservoir is not None:
        for sample in reservoir:
            shard.append(sample)
            sample_count += 1
            if len(shard) >= shard_size:
                shards.append(_flush_shard(shard, out, len(shards), compression))
                print(f"wrote {shards[-1]['file']} samples={shards[-1]['samples']}")
                shard = []

    if shard:
        shards.append(_flush_shard(shard, out, len(shards), compression))
        print(f"wrote {shards[-1]['file']} samples={shards[-1]['samples']}")

    manifest = {
        "sample_count": sample_count,
        "shard_count": len(shards),
        "feature_dim": RALF_NUMERIC_FEATURE_DIM,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(source),
        "compression": compression,
        "shards": shards,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build compact tensor-ready RALF cache shards.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--compression", choices=["none", "gzip"], default="none")
    args = parser.parse_args(argv)
    manifest = build_cache(
        input_path=args.input,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        max_samples=args.max_samples,
        seed=args.seed,
        compression=args.compression,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
