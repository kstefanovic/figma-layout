"""CLI wrapper for top-level layout prediction."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict top-level Figma layout for a target resolution.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-size", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrieval", dest="retrieval_enabled", action="store_true", default=True)
    parser.add_argument("--no-retrieval", dest="retrieval_enabled", action="store_false")
    parser.add_argument("--retrieval-records", default=None)
    parser.add_argument("--retrieval-k", type=int, default=5)
    args = parser.parse_args(argv)
    from layout_training.model.predict import predict_file

    result = predict_file(
        args.checkpoint,
        args.input,
        args.target_size,
        args.output,
        device=args.device,
        retrieval_enabled=args.retrieval_enabled,
        retrieval_records_path=args.retrieval_records,
        retrieval_k=args.retrieval_k,
        retrieval_blend=args.retrieval_enabled,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
