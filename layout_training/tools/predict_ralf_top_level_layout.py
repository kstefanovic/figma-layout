"""Predict top-level layout using trained RALF model."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict top-level layout with RALF model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-size", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    from layout_training.ralf.predict import predict_file

    result = predict_file(
        checkpoint=args.checkpoint,
        input_path=args.input,
        target_size=args.target_size,
        records=args.records,
        retrieval_k=args.retrieval_k,
        output_path=args.output,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

