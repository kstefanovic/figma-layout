
#!/usr/bin/env bash
set -euo pipefail

cd /root/figma-layout

RAW_DIR="gnn_layout/data/raw"
OUT_DIR="gnn_layout/data/clean_banners"

mkdir -p "$RAW_DIR" "$OUT_DIR"

echo "Looking for mixed JSON files in: $RAW_DIR"
ls -lh "$RAW_DIR"/*.json

for f in "$RAW_DIR"/*.json; do
  base=$(basename "$f" .json)

  echo ""
  echo "======================================"
  echo "Extracting clean banners from: $base"
  echo "======================================"

  python -m gnn_layout.src.extract_clean \
    --input "$f" \
    --output "$OUT_DIR/${base}_clean.json" \
    --rejects "$OUT_DIR/${base}_rejects.jsonl" \
    --report "$OUT_DIR/${base}_report.json"
done

echo ""
echo "Done extracting clean banners."
ls -lh "$OUT_DIR"