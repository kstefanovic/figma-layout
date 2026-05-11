#!/usr/bin/env bash
set -euo pipefail

cd /root/figma-layout

MODEL="gnn_layout/data/checkpoints/gnn_layout.pt"
SOURCE="${1:-gnn_layout/data/clean_banners/test_source.json}"
TARGET_W="${2:-1536}"
TARGET_H="${3:-640}"
OUT_FILE="${4:-gnn_layout/data/pairs/predicted_priors_${TARGET_W}x${TARGET_H}.json}"

echo "Predicting priors..."
echo "Source: $SOURCE"
echo "Target: ${TARGET_W}x${TARGET_H}"
echo "Output: $OUT_FILE"

CUDA_VISIBLE_DEVICES=0 python -m gnn_layout.src.predict \
  --checkpoint "$MODEL" \
  --source "$SOURCE" \
  --target-width "$TARGET_W" \
  --target-height "$TARGET_H" \
  --output "$OUT_FILE"

echo ""
cat "$OUT_FILE"