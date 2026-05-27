"""Retrieval utilities for RALF-style top-level layout inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_layout_records(records_path: str) -> list[dict]:
    path = Path(records_path)
    if not path.exists():
        raise FileNotFoundError(f"Retrieval records not found: {path}")
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                records.append(row)
    return records


def _record_roles(record: dict[str, Any]) -> set[str]:
    return {str(token.get("train_role") or "unknown_group") for token in record.get("tokens") or []}


def build_retrieval_index(records_path: str) -> object:
    records = load_layout_records(records_path)
    entries = []
    for record in records:
        canvas = record.get("canvas") or {}
        aspect = float(canvas.get("aspect") or 1.0)
        orient = str(canvas.get("orientation") or "")
        entries.append(
            {
                "record": record,
                "roles": _record_roles(record),
                "aspect": aspect,
                "orientation": orient,
            }
        )
    return {"records_path": str(Path(records_path).resolve()), "entries": entries}


def _ratio_similarity(a: float, b: float) -> float:
    a = float(a or 0.0)
    b = float(b or 0.0)
    if a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)


def retrieve_similar_layouts(
    query_record: dict,
    target_width: int,
    target_height: int,
    index: object,
    k: int = 5,
) -> list[dict]:
    query_roles = _record_roles(query_record)
    query_canvas = query_record.get("canvas") or {}
    query_aspect = float(query_canvas.get("aspect") or 1.0)
    target_aspect = float(target_width) / max(float(target_height), 1e-6)
    target_orientation = "square"
    if target_width > target_height:
        target_orientation = "landscape"
    elif target_width < target_height:
        target_orientation = "portrait"

    rows = []
    for item in (index or {}).get("entries", []):
        cand = item["record"]
        cand_roles = item["roles"]
        inter = len(query_roles & cand_roles)
        union = len(query_roles | cand_roles)
        role_jaccard = (inter / union) if union else 0.0
        cand_aspect = float(item["aspect"] or 1.0)
        target_aspect_similarity = _ratio_similarity(target_aspect, cand_aspect)
        source_aspect_similarity = _ratio_similarity(query_aspect, cand_aspect)
        orientation_match = 1.0 if target_orientation == str(item["orientation"] or "") else 0.0
        score = (
            0.40 * role_jaccard
            + 0.30 * target_aspect_similarity
            + 0.20 * source_aspect_similarity
            + 0.10 * orientation_match
        )
        rows.append(
            {
                "record": cand,
                "score": float(score),
                "role_jaccard": float(role_jaccard),
                "target_aspect_similarity": float(target_aspect_similarity),
                "source_aspect_similarity": float(source_aspect_similarity),
                "orientation_match": float(orientation_match),
            }
        )
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[: max(0, int(k))]


def build_retrieval_role_priors(
    query_record: dict,
    retrieved: list[dict],
) -> dict:
    priors: dict[str, dict[str, Any]] = {}
    query_tokens = query_record.get("tokens") or []
    for q_token in query_tokens:
        token_id = str(q_token.get("token_id") or "")
        train_role = str(q_token.get("train_role") or "")
        weighted = [0.0, 0.0, 0.0, 0.0]
        weight_sum = 0.0
        matched_count = 0
        score_sum = 0.0
        for item in retrieved:
            score = float(item.get("score") or 0.0)
            if score <= 0:
                continue
            cand = item.get("record") or {}
            cand_tokens = cand.get("tokens") or []
            by_id = {str(t.get("token_id") or ""): t for t in cand_tokens}
            match = by_id.get(token_id)
            if match is None:
                same_role = [t for t in cand_tokens if str(t.get("train_role") or "") == train_role]
                if len(same_role) == 1:
                    match = same_role[0]
            if match is None:
                continue
            vec = match.get("center_size_norm")
            if not isinstance(vec, list) or len(vec) < 4:
                continue
            vals = [float(x) for x in vec[:4]]
            for i in range(4):
                weighted[i] += score * vals[i]
            weight_sum += score
            matched_count += 1
            score_sum += score
        if weight_sum > 0:
            priors[token_id] = {
                "center_size_norm": [v / weight_sum for v in weighted],
                "matched_count": matched_count,
                "avg_score": score_sum / max(1, matched_count),
            }
    return priors

