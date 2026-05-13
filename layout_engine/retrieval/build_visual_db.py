from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .clean_semantic_loader import load_clean_semantic_banners
from .feature_extractor import make_clean_exemplar


def _size_key(width: float, height: float) -> str:
    return f"{int(round(width))}x{int(round(height))}"


def build_visual_db(input_path: str, output_path: str) -> dict:
    banners, report = load_clean_semantic_banners(input_path)
    exemplars = [ex for b in banners if (ex := make_clean_exemplar(b)) is not None]
    by_orientation = Counter(ex["orientation"] for ex in exemplars)
    by_aspect_bucket = Counter(ex["aspect_bucket"] for ex in exemplars)
    by_archetype = Counter(ex["visual_archetype"] for ex in exemplars)
    by_size = Counter(_size_key(ex["width"], ex["height"]) for ex in exemplars)
    feature_dim = len(exemplars[0]["feature_vector"]) if exemplars else 0
    db = {
        "version": "visual_retrieval_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_input": str(input_path),
        "stats": {
            **report,
            "exemplar_count": len(exemplars),
            "by_orientation": dict(by_orientation),
            "by_aspect_bucket": dict(by_aspect_bucket),
            "by_archetype": dict(by_archetype),
            "by_size": dict(by_size),
        },
        "feature_dim": feature_dim,
        "exemplars": exemplars,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    summary = {
        "version": db["version"],
        "source_input": db["source_input"],
        "stats": db["stats"],
        "feature_dim": feature_dim,
    }
    with out.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return db


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visual retrieval DB from clean semantic banners.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    db = build_visual_db(args.input, args.output)
    stats = db["stats"]
    print(f"candidates: {stats['total_candidates']}")
    print(f"clean banners used: {stats['clean_count']}")
    print(f"exemplars: {stats['exemplar_count']}")
    print(f"orientation distribution: {stats['by_orientation']}")
    print(f"aspect bucket distribution: {stats['by_aspect_bucket']}")
    print(f"archetype distribution: {stats['by_archetype']}")


if __name__ == "__main__":
    main()

