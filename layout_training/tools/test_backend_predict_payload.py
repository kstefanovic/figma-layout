"""Build request payload JSON for /figma/top-level-layout-predict."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_target_size(value: str) -> tuple[int, int]:
    clean = value.lower().replace("х", "x").strip()
    m = re.fullmatch(r"(\d+)\s*x\s*(\d+)", clean)
    if not m:
        raise ValueError("--target-size must look like 600x1024")
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        raise ValueError("target dimensions must be positive")
    return w, h


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build backend prediction payload from semantic JSON.")
    parser.add_argument("--input", required=True, help="Input semantic JSON path.")
    parser.add_argument("--target-size", required=True, help="Target size like 600x1024.")
    parser.add_argument("--output", required=True, help="Output payload path.")
    args = parser.parse_args(argv)

    target_w, target_h = parse_target_size(args.target_size)
    src = Path(args.input)
    out = Path(args.output)
    data = json.loads(src.read_text(encoding="utf-8"))
    payload = {
        "semantic_json": data,
        "target_width": target_w,
        "target_height": target_h,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

