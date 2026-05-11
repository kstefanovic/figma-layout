
#!/usr/bin/env bash
set -euo pipefail

cd /root/figma-layout

CLEAN_DIR="gnn_layout/data/clean_banners"
PAIR_DIR="gnn_layout/data/pairs"

mkdir -p "$PAIR_DIR"

echo "Looking for clean JSON files in: $CLEAN_DIR"
ls -lh "$CLEAN_DIR"/*_clean.json

for f in "$CLEAN_DIR"/*_clean.json; do
  base=$(basename "$f" _clean.json)

  echo ""
  echo "======================================"
  echo "Building pairs from: $base"
  echo "======================================"

  python -m gnn_layout.src.build_pairs \
    --input "$f" \
    --output "$PAIR_DIR/${base}_pairs.jsonl"
done

echo ""
echo "Done building pair files."
ls -lh "$PAIR_DIR"/*_pairs.jsonl