#!/usr/bin/env bash
set -euo pipefail

cd /root/figma-layout

CLEAN_FILE="${1:-gnn_layout/data/clean_banners/mixed_001_clean.json}"
OUT_FILE="gnn_layout/data/clean_banners/test_source.json"

python - <<PY
import json

clean_file = "$CLEAN_FILE"
out_file = "$OUT_FILE"

with open(clean_file, "r", encoding="utf-8") as f:
    banners = json.load(f)

if not banners:
    raise SystemExit(f"No banners found in {clean_file}")

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(banners[0], f, ensure_ascii=False, indent=2)

print("Saved test source:", out_file)
print("Source banner:", banners[0].get("name"))
PY