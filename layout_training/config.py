"""Shared defaults for the top-level layout training pipeline."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
REPORTS_DIR = PACKAGE_ROOT / "reports"
CHECKPOINTS_DIR = PACKAGE_ROOT / "checkpoints"

DEFAULT_RECORDS = DATA_DIR / "layout_records" / "top_level_records.jsonl"
DEFAULT_PAIRS = DATA_DIR / "layout_pairs" / "top_level_pairs.jsonl"
DEFAULT_TRAIN = DATA_DIR / "layout_pairs" / "train.jsonl"
DEFAULT_VAL = DATA_DIR / "layout_pairs" / "val.jsonl"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "top_level_layout_v1.pt"

MIN_SIZE_PX = 1.0
EPS = 1e-6

