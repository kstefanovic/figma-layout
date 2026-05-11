
#!/usr/bin/env bash
set -euo pipefail

cd /root/figma-layout

PAIR_DIR="gnn_layout/data/pairs"
OUT_FILE="$PAIR_DIR/all_pairs.jsonl"

mkdir -p "$PAIR_DIR"

echo "Merging pair JSONL files..."
ls -lh "$PAIR_DIR"/*_pairs.jsonl

python -m gnn_layout.src.merge_jsonl \
  --inputs "$PAIR_DIR"/*_pairs.jsonl \
  --output "$OUT_FILE" \
  --dedupe

echo ""
echo "Merged pairs:"
wc -l "$OUT_FILE"

echo ""
echo "Summary:"
cat "$PAIR_DIR/all_pairs.summary.json"