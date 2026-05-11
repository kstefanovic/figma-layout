#!/usr/bin/env bash
set -euo pipefail

cd /root/figma-layout

PAIR_FILE="gnn_layout/data/pairs/all_pairs.jsonl"
OUT_MODEL="gnn_layout/data/checkpoints/gnn_layout.pt"

mkdir -p gnn_layout/data/checkpoints

echo "Starting full GPU training: 100 epochs"

CUDA_VISIBLE_DEVICES=0 python -m gnn_layout.src.train \
  --pairs "$PAIR_FILE" \
  --epochs 100 \
  --batch-size 64 \
  --hidden 256 \
  --lr 0.001 \
  --out "$OUT_MODEL"

echo ""
echo "Full training finished."
ls -lh "$OUT_MODEL"
