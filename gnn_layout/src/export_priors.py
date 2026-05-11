"""Export GNN predictions in the layout-prior JSON format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .predict import predict_priors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target-width", required=True, type=float)
    parser.add_argument("--target-height", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = predict_priors(args.checkpoint, args.source, args.target_width, args.target_height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Exported OR-Tools prior JSON: {args.output}")


if __name__ == "__main__":
    main()
