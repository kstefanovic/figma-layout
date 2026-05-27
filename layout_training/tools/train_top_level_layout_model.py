"""CLI wrapper for training the top-level layout transformer."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train top-level Figma layout transformer V1.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reports-dir", default="layout_training/reports")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience in epochs.")
    args = parser.parse_args(argv)
    from layout_training.model.train import train_model

    result = train_model(
        train_path=args.train,
        val_path=args.val,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        reports_dir=args.reports_dir,
        patience=args.patience,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
