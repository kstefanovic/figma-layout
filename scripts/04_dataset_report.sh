#!/usr/bin/env bash
set -euo pipefail

cd /root/figma-layout

PAIR_FILE="gnn_layout/data/pairs/all_pairs.jsonl"
OUT_REPORT="gnn_layout/data/pairs/all_pairs_report.json"

echo "Creating dataset report..."

python -m gnn_layout.src.dataset_report \
  --pairs "$PAIR_FILE" \
  --output "$OUT_REPORT"

echo ""
echo "Report saved:"
ls -lh "$OUT_REPORT"

echo ""
cat "$OUT_REPORT"