from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .feature_extractor import flatten_nodes, get_bounds

try:
    from gnn_layout.src.clean_filter import is_clean_banner as _gnn_is_clean_banner
except Exception:  # pragma: no cover - optional local dependency
    _gnn_is_clean_banner = None


REQUIRED_ROLES = {
    "hero_image",
    "background_shape",
    "brand_group",
    "headline_group",
    "legal_text",
    "age_badge",
}


def load_json_inputs(input_path: str) -> list[dict]:
    path = Path(input_path)
    if path.is_dir():
        out: list[dict] = []
        for p in sorted(path.rglob("*.json")):
            out.extend(load_json_inputs(str(p)))
        return out
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def collect_candidate_frames(obj: Any) -> list[dict]:
    frames: list[dict] = []

    def walk(item: Any) -> None:
        if isinstance(item, list):
            for x in item:
                walk(x)
            return
        if not isinstance(item, dict):
            return
        b = get_bounds(item)
        try:
            w = float(b.get("width") or 0)
            h = float(b.get("height") or 0)
        except (TypeError, ValueError):
            w = h = 0.0
        if str(item.get("type") or "").lower() == "frame" and w > 0 and h > 0:
            frames.append(item)
        for child in item.get("children") or []:
            walk(child)

    walk(obj)
    return frames


def _fallback_role_name(name: str) -> str:
    raw = str(name or "").strip().lower().replace("-", "_")
    if not raw:
        return ""
    if raw == "image_zone" or raw.startswith("image_zone_"):
        return "hero_image"
    for role in REQUIRED_ROLES | {"unassigned"}:
        if raw == role or raw.startswith(role + "_"):
            return role
    return ""


def _fallback_clean_banner(banner: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    counts: Counter[str] = Counter()
    for item in flatten_nodes(banner):
        role = _fallback_role_name(str(item["node"].get("name") or ""))
        if role:
            counts[role] += 1
    for role in REQUIRED_ROLES:
        if counts.get(role, 0) == 0:
            reasons.append(f"missing_role:{role}")
        elif counts.get(role, 0) > 1:
            reasons.append(f"duplicate_role:{role}:{counts[role]}")
    if counts.get("unassigned", 0) > 0:
        reasons.append("contains_unassigned")
    return len(reasons) == 0, reasons


def is_strict_clean_semantic_banner(banner: dict) -> tuple[bool, list[str]]:
    if _gnn_is_clean_banner is not None:
        ok, reasons = _gnn_is_clean_banner(banner, strict=True)
        if ok:
            # clean_filter intentionally ignores background_shape; require it here for visual retrieval.
            bg_count = sum(
                1
                for item in flatten_nodes(banner)
                if _fallback_role_name(str(item["node"].get("name") or "")) == "background_shape"
            )
            if bg_count == 1:
                return True, []
            return False, ["missing_role:background_shape" if bg_count == 0 else f"duplicate_role:background_shape:{bg_count}"]
        return ok, list(reasons)
    return _fallback_clean_banner(banner)


def load_clean_semantic_banners(input_path: str) -> tuple[list[dict], dict]:
    inputs = load_json_inputs(input_path)
    candidates: list[dict] = []
    for obj in inputs:
        candidates.extend(collect_candidate_frames(obj))
    clean: list[dict] = []
    reject_counts: Counter[str] = Counter()
    for banner in candidates:
        ok, reasons = is_strict_clean_semantic_banner(banner)
        if ok:
            clean.append(banner)
        else:
            for r in reasons or ["unknown_reject"]:
                reject_counts[r] += 1
    report = {
        "total_candidates": len(candidates),
        "clean_count": len(clean),
        "reject_count": len(candidates) - len(clean),
        "reject_reason_counts": dict(reject_counts),
    }
    return clean, report

