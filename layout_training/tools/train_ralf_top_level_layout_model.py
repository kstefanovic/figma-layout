"""Train full RALF top-level layout model."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train RALF top-level layout transformer.")
    parser.add_argument("--train", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--train-cache", default=None)
    parser.add_argument("--val-cache", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--reports-dir", default="layout_training/reports")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-max-open-shards", type=int, default=2)
    parser.add_argument("--cache-load-all", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    parser.add_argument("--compile", dest="compile_model", action="store_true", default=False)
    parser.add_argument("--save-optimizer-state", action="store_true", default=False)
    args = parser.parse_args(argv)
    if not args.train and not args.train_cache:
        parser.error("one of --train or --train-cache is required")
    if not args.val and not args.val_cache:
        parser.error("one of --val or --val-cache is required")
    from layout_training.ralf.train import train_ralf_model

    result = train_ralf_model(
        train_path=args.train,
        val_path=args.val,
        train_cache=args.train_cache,
        val_cache=args.val_cache,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        d_model=args.d_model,
        nhead=args.nhead,
        retrieval_k=args.retrieval_k,
        reports_dir=args.reports_dir,
        patience=args.patience,
        resume=args.resume,
        num_workers=args.num_workers,
        cache_max_open_shards=args.cache_max_open_shards,
        cache_load_all=args.cache_load_all,
        amp=args.amp,
        tf32=args.tf32,
        compile_model=args.compile_model,
        save_optimizer_state=args.save_optimizer_state,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
