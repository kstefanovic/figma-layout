#!/usr/bin/env bash
set -euo pipefail

cd /root/figma-layout

PAIR_FILE="gnn_layout/data/pairs/all_pairs.jsonl"
MODEL="gnn_layout/data/checkpoints/gnn_layout.pt"

echo "Evaluating model..."

CUDA_VISIBLE_DEVICES=0 python -m gnn_layout.src.evaluate \
  --pairs "$PAIR_FILE" \
  --checkpoint "$MODEL"