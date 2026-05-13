# Visual Retrieval Priors

Build DB from repaired clean file:

```bash
python -m layout_engine.retrieval.build_visual_db \
  --input gnn_layout/data/clean_banners/repaired_clean_all.json \
  --output layout_engine/retrieval_db/visual_layout_db.json
```

Build DB from big mixed JSON:

```bash
python -m layout_engine.retrieval.build_visual_db \
  --input gnn_layout/data/raw/big_mixed.json \
  --output layout_engine/retrieval_db/visual_layout_db.json
```

Convert with retrieval:

```bash
python -m layout_engine.convert \
  layout_engine/input_json/test.json \
  1536 \
  640 \
  output_1536x640_retrieval.json \
  --visual-mode retrieval \
  --visual-retrieval-db layout_engine/retrieval_db/visual_layout_db.json \
  --visual-retrieval-top-k 15
```

Inspect metadata:

```bash
grep -A80 "visual_retrieval" output_1536x640_retrieval.json
```

Notes:
- Retrieval only supplies visual priors for `hero_image` and `background_shape`.
- Raw candidate selection uses geometry, node type, and text/child summaries. It does not use numeric layer-name mappings.
- Qwen, GNN training, text role solving, gradients, stars, glow, and decorations are intentionally out of scope.

