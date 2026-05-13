from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .feature_extractor import (
    denormalize_box,
    get_aspect_bucket,
    get_orientation,
    make_runtime_query_features,
)
from .raw_candidate_selector import select_raw_visual_candidates


def load_visual_db(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(float(a[i]) * float(b[i]) for i in range(n))
    na = math.sqrt(sum(float(a[i]) ** 2 for i in range(n)))
    nb = math.sqrt(sum(float(b[i]) ** 2 for i in range(n)))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _weighted_average_box(boxes: list[tuple[list[float], float]]) -> list[float] | None:
    clean = []
    for box, weight in boxes:
        if not box or len(box) < 4:
            continue
        x, y, w, h = [float(v) for v in box[:4]]
        if w <= 0 or h <= 0 or x < -0.8 or y < -0.8 or w > 2.5 or h > 2.5:
            continue
        clean.append(([x, y, w, h], max(float(weight), 0.0)))
    total = sum(w for _box, w in clean)
    if not clean or total <= 0:
        return None
    return [sum(box[i] * w for box, w in clean) / total for i in range(4)]


def _first_valid_box(scored_exemplars: list[tuple[float, dict]], key: str) -> list[float] | None:
    for _score, ex in scored_exemplars:
        box = ex.get(key)
        if not box or len(box) < 4:
            continue
        x, y, w, h = [float(v) for v in box[:4]]
        if w <= 0 or h <= 0 or x < -0.8 or y < -0.8 or w > 2.5 or h > 2.5:
            continue
        return [x, y, w, h]
    return None


def retrieve_visual_priors(
    db: dict,
    raw_banner: dict,
    target_width: int,
    target_height: int,
    top_k: int = 15,
    min_count: int = 3,
) -> dict:
    raw_candidates = select_raw_visual_candidates(raw_banner)
    query = make_runtime_query_features(raw_banner, target_width, target_height, raw_candidates)
    target_aspect = max(float(target_width) / max(float(target_height), 1e-6), 1e-6)
    target_orientation = get_orientation(target_width, target_height)
    target_bucket = get_aspect_bucket(target_width, target_height)
    estimated_archetype = raw_candidates.get("estimated_archetype") or "mixed"

    scored: list[tuple[float, dict]] = []
    for ex in db.get("exemplars") or []:
        ex_aspect = max(float(ex.get("aspect") or 1.0), 1e-6)
        feature_cosine = _cosine(query, ex.get("feature_vector") or [])
        aspect_similarity = math.exp(-abs(math.log(target_aspect / ex_aspect)))
        orientation_match = 1.0 if ex.get("orientation") == target_orientation else 0.0
        archetype_match = 1.0 if ex.get("visual_archetype") == estimated_archetype else 0.0
        score = (
            0.45 * feature_cosine
            + 0.30 * aspect_similarity
            + 0.15 * orientation_match
            + 0.10 * archetype_match
        )
        if ex.get("aspect_bucket") == target_bucket:
            score += 0.05
        scored.append((score, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[: max(1, int(top_k))]

    # Hero must stay visually exact: use the top retrieved exemplar's hero box, not a softened average.
    hero_box = _first_valid_box(chosen, "hero_image_box")
    bg_box = _weighted_average_box([(ex.get("background_shape_box"), max(score, 0.001)) for score, ex in chosen])
    priors: dict[str, dict[str, float]] = {}
    abs_boxes: dict[str, dict[str, float]] = {}
    if hero_box:
        priors["hero_image"] = {"x": hero_box[0], "y": hero_box[1], "w": hero_box[2], "h": hero_box[3]}
        abs_boxes["hero_image"] = denormalize_box(hero_box, target_width, target_height)
    if bg_box:
        priors["background_shape"] = {"x": bg_box[0], "y": bg_box[1], "w": bg_box[2], "h": bg_box[3]}
        abs_boxes["background_shape"] = denormalize_box(bg_box, target_width, target_height)

    result: dict[str, Any] = {
        "source": "retrieval_visual_priors",
        "top_k": top_k,
        "target_width": target_width,
        "target_height": target_height,
        "target_orientation": target_orientation,
        "estimated_raw_archetype": estimated_archetype,
        "priors": priors,
        "abs_boxes": abs_boxes,
        "prior_strategy": {
            "hero_image": "top_neighbor_exact",
            "background_shape": "top_k_weighted_average",
        },
        "selected_raw_candidates": raw_candidates.get("selected") or {},
        "neighbors": [
            {
                "id": ex.get("id"),
                "name": ex.get("name"),
                "width": ex.get("width"),
                "height": ex.get("height"),
                "score": round(score, 6),
                "orientation": ex.get("orientation"),
                "archetype": ex.get("visual_archetype"),
                "hero_image_box": ex.get("hero_image_box"),
                "background_shape_box": ex.get("background_shape_box"),
            }
            for score, ex in chosen
        ],
    }
    if len(chosen) < min_count:
        result["warning"] = f"Only {len(chosen)} retrieval neighbors available; requested min_count={min_count}."
    return result

