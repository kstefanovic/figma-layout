# GNN Layout Priors

This package trains a small GraphSAGE model to predict normalized bbox priors for the main semantic Figma banner roles:

- `hero_image`
- `brand_group`
- `headline_group`
- `legal_text`
- `age_badge`

It does not generate full JSON and does not render Figma. The output is intended to become soft priors for the existing OR-Tools layout engine later.

## Data

Input training data should be clean semantic banner JSONs with stable role names. Helper visuals such as `background_shape`, `background_gradient_1`, `background_gradient_2`, stars, `decoration_group`, and `background` are ignored for GNN target prediction because the deterministic layout engine already handles those.

## Step 1: Extract Clean Banners

```bash
python -m gnn_layout.src.extract_clean \
  --input /root/figma-layout/layout_test_1778461892292.json \
  --output gnn_layout/data/clean_banners/clean_001.json \
  --rejects gnn_layout/data/clean_banners/rejects_001.jsonl \
  --report gnn_layout/data/clean_banners/report_001.json
```

The extractor recursively scans mixed JSON files/folders and keeps only strict clean semantic frame blocks. Rejected candidates are written to JSONL with reason codes so bad generated/raw blocks can be inspected.

## Step 2: Build Pairs

```bash
python -m gnn_layout.src.build_pairs \
  --input gnn_layout/data/clean_banners/clean_001.json \
  --output gnn_layout/data/pairs/pairs_001.jsonl
```

By default, pairs are only created inside the same creative family. Use `--allow-cross-family` only for experiments where template leakage is acceptable. A `*.summary.json` sidecar is written next to the pair JSONL.

Useful options:

```bash
--max-pairs-per-family 500 --shuffle --seed 42
```

## Step 3: Merge Pair Files

```bash
python -m gnn_layout.src.merge_jsonl \
  --inputs gnn_layout/data/pairs/pairs_001.jsonl gnn_layout/data/pairs/pairs_002.jsonl \
  --output gnn_layout/data/pairs/all_pairs.jsonl \
  --dedupe
```

The merger validates pair rows, skips blanks, optionally dedupes by `pair_id`, and writes `all_pairs.summary.json`.

## Step 4: Dataset Report

```bash
python -m gnn_layout.src.dataset_report \
  --clean gnn_layout/data/clean_banners/clean_001.json \
  --pairs gnn_layout/data/pairs/all_pairs.jsonl \
  --output gnn_layout/data/pairs/all_pairs_report.json
```

The report includes banner counts, orientation/size/family distributions, pair target orientation distribution, missing roles, and normalized bbox min/max/mean per role.

## Step 5: Train

```bash
python -m gnn_layout.src.train \
  --pairs gnn_layout/data/pairs/all_pairs.jsonl \
  --epochs 100 \
  --batch-size 16 \
  --hidden 128 \
  --lr 0.001 \
  --out gnn_layout/data/checkpoints/gnn_layout.pt
```

The trainer splits by `family_key` so all pairs from the same creative/template family stay in the same split. It saves the best checkpoint by validation loss plus a sidecar config at `gnn_layout.pt.config.json`.

## Evaluate

```bash
python -m gnn_layout.src.evaluate \
  --pairs gnn_layout/data/pairs/pairs.jsonl \
  --checkpoint gnn_layout/data/checkpoints/gnn_layout.pt
```

Metrics include mean L1 bbox error, role-wise L1, role-wise IoU, and missing-role counts.

## Predict

```bash
python -m gnn_layout.src.predict \
  --checkpoint gnn_layout/data/checkpoints/gnn_layout.pt \
  --source /root/figma-layout/layout_engine/input_json/test.json \
  --target-width 1536 \
  --target-height 640 \
  --output predicted_priors.json
```

Output format:

```json
{
  "orientation": "landscape",
  "target_width": 1536,
  "target_height": 640,
  "priors": {
    "hero_image": {"x": 0.48, "y": 0.0, "w": 0.52, "h": 1.0, "confidence": 1.0}
  }
}
```

## Export Priors

```bash
python -m gnn_layout.src.export_priors \
  --checkpoint gnn_layout/data/checkpoints/gnn_layout.pt \
  --source /root/figma-layout/layout_engine/input_json/test.json \
  --target-width 1536 \
  --target-height 640 \
  --output layout_engine/gnn_priors.json
```

This uses the same output format as `predict.py`, placed where later OR-Tools integration can read it.
