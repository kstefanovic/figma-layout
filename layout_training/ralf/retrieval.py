"""Retrieval helpers for RALF pair building and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

from layout_training.pairs import read_jsonl

_COMPACT_INDEX_CACHE: dict[str, dict[str, Any]] = {}
_JSONL_INDEX_CACHE: dict[str, dict[str, Any]] = {}


def _torch_load(path: str | Path):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load compact RALF retrieval index.") from exc
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _family_from_record(record: dict[str, Any]) -> str:
    source = str(record.get("source_file") or "")
    name = Path(source).name
    if "__" in name:
        return name.split("__", 1)[0]
    return "unknown_family"


def _record_roles(record: dict[str, Any]) -> set[str]:
    return {str(tok.get("train_role") or "unknown_group") for tok in (record.get("tokens") or [])}


def _aspect(record: dict[str, Any]) -> float:
    canvas = record.get("canvas") or {}
    return float(canvas.get("aspect") or 1.0)


def _orientation(record: dict[str, Any]) -> str:
    canvas = record.get("canvas") or {}
    return str(canvas.get("orientation") or "square")


def _ratio_similarity(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)


def _target_orientation(target_w: int, target_h: int) -> str:
    if target_w == target_h:
        return "square"
    return "landscape" if target_w > target_h else "portrait"


def _compact_canvas(canvas: dict[str, Any] | None) -> dict[str, Any]:
    canvas = canvas or {}
    width = float(canvas.get("width") or 0.0)
    height = float(canvas.get("height") or 0.0)
    aspect = float(canvas.get("aspect") or 0.0)
    if aspect <= 0.0:
        aspect = width / max(height, 1e-6) if width > 0 and height > 0 else 1.0
    orientation = str(canvas.get("orientation") or _target_orientation(int(width), int(height)))
    return {"width": width, "height": height, "aspect": aspect, "orientation": orientation}


def compact_token(token: dict[str, Any]) -> dict[str, Any]:
    return {
        "token_id": str(token.get("token_id") or ""),
        "train_role": str(token.get("train_role") or "unknown_group"),
        "center_size_norm": [float(x) for x in (token.get("center_size_norm") or [0.5, 0.5, 0.1, 0.1])[:4]],
        "type": str(token.get("type") or "unknown"),
        "area_ratio": float(token.get("area_ratio") or 0.0),
        "has_text": bool(token.get("has_text")),
        "has_image": bool(token.get("has_image")),
        "has_gradient": bool(token.get("has_gradient")),
        "has_star": bool(token.get("has_star")),
        "discount_text": bool(token.get("discount_text")),
        "rotation_deg": float(token.get("rotation_deg") or 0.0),
        "instance_count": float(token.get("instance_count") or 1.0),
        "descendant_count": float(token.get("descendant_count") or 0.0),
    }


def compact_record(record: dict[str, Any], role_to_bit: dict[str, int] | None = None) -> dict[str, Any]:
    rid = str(record.get("id") or record.get("record_id") or "")
    roles = sorted(_record_roles(record))
    role_mask = 0
    if role_to_bit is not None:
        for role in roles:
            bit = role_to_bit.get(role)
            if bit is not None:
                role_mask |= 1 << bit
    canvas = _compact_canvas(record.get("canvas"))
    return {
        "record_id": rid,
        "family_id": str(record.get("family_id") or _family_from_record(record)),
        "source_file": str(record.get("source_file") or ""),
        "canvas": canvas,
        "role_set": roles,
        "role_mask": role_mask,
        "tokens": [compact_token(tok) for tok in (record.get("tokens") or [])],
    }


def build_compact_retrieval_index(records: list[dict[str, Any]], records_path: str | None = None) -> dict[str, Any]:
    all_roles = sorted({str(tok.get("train_role") or "unknown_group") for rec in records for tok in (rec.get("tokens") or [])})
    role_to_bit = {role: idx for idx, role in enumerate(all_roles)}
    entries = []
    by_id: dict[str, dict[str, Any]] = {}
    by_orientation: dict[str, list[int]] = {}
    for rec in records:
        rid = str(rec.get("id") or "")
        if not rid:
            continue
        entry = compact_record(rec, role_to_bit)
        entries.append(entry)
        by_id[rid] = entry
        by_orientation.setdefault(str(entry["canvas"]["orientation"]), []).append(len(entries) - 1)
    return {
        "index_type": "compact_pt",
        "version": 1,
        "entries": entries,
        "by_id": by_id,
        "by_orientation": by_orientation,
        "role_to_bit": role_to_bit,
        "role_count": len(role_to_bit),
        "record_count": len(entries),
        "records_path": str(Path(records_path).resolve()) if records_path else None,
    }


def _normalize_loaded_index(index: dict[str, Any], path: str | Path, index_type: str) -> dict[str, Any]:
    entries = list(index.get("entries") or [])
    role_to_bit = dict(index.get("role_to_bit") or {})
    for entry in entries:
        if not isinstance(entry.get("role_set"), set):
            entry["role_set"] = set(entry.get("role_set") or [])
        if "aspect" not in entry:
            entry["aspect"] = float((entry.get("canvas") or {}).get("aspect") or 1.0)
        if "orientation" not in entry:
            entry["orientation"] = str((entry.get("canvas") or {}).get("orientation") or "square")
    by_id = {str(e.get("record_id") or e.get("id") or ""): e for e in entries}
    index["entries"] = entries
    index["by_id"] = by_id
    index["role_to_bit"] = role_to_bit
    index["record_count"] = len(entries)
    index["index_type"] = index_type
    index["path"] = str(Path(path).expanduser().resolve())
    return index


def build_record_index(records_path: str) -> dict[str, Any]:
    records = read_jsonl(records_path)
    index = build_compact_retrieval_index(records, records_path=records_path)
    return _normalize_loaded_index(index, records_path, "jsonl_cached")


def load_jsonl_record_index(records_path: str) -> dict[str, Any]:
    path = str(Path(records_path).expanduser().resolve())
    cached = _JSONL_INDEX_CACHE.get(path)
    if cached is not None:
        return cached
    warnings.warn(f"RALF compact retrieval index missing; loading and caching JSONL records once: {path}")
    index = build_record_index(path)
    _JSONL_INDEX_CACHE[path] = index
    return index


def load_compact_retrieval_index(path: str) -> dict[str, Any]:
    resolved = str(Path(path).expanduser().resolve())
    cached = _COMPACT_INDEX_CACHE.get(resolved)
    if cached is not None:
        return cached
    index = _torch_load(resolved)
    if not isinstance(index, dict):
        raise ValueError(f"RALF retrieval index must be a dict: {resolved}")
    index = _normalize_loaded_index(index, resolved, "compact_pt")
    _COMPACT_INDEX_CACHE[resolved] = index
    return index


def retrieval_index_status(path: str) -> dict[str, Any]:
    resolved = str(Path(path).expanduser().resolve())
    index = _COMPACT_INDEX_CACHE.get(resolved) or _JSONL_INDEX_CACHE.get(resolved)
    return {
        "path": resolved,
        "exists": Path(resolved).exists(),
        "loaded_in_memory": index is not None,
        "record_count": (index or {}).get("record_count"),
        "index_type": (index or {}).get("index_type"),
    }


def load_retrieval_index_prefer_compact(index_path: str, records_path: str) -> dict[str, Any]:
    if index_path:
        compact = Path(index_path).expanduser().resolve()
        if compact.exists():
            return load_compact_retrieval_index(str(compact))
    return load_jsonl_record_index(records_path)


def retrieve_for_query(
    *,
    query_record: dict[str, Any],
    target_width: int,
    target_height: int,
    index: dict[str, Any],
    k: int,
    exclude_target_id: str | None = None,
    exclude_family_id: str | None = None,
) -> list[dict[str, Any]]:
    return retrieve_similar_layouts(
        query_record=query_record,
        target_width=target_width,
        target_height=target_height,
        index=index,
        k=k,
        exclude_target_id=exclude_target_id,
        exclude_family_id=exclude_family_id,
    )


def retrieve_similar_layouts(
    query_record: dict[str, Any],
    target_width: int,
    target_height: int,
    index: dict[str, Any],
    k: int = 5,
    exclude_target_id: str | None = None,
    exclude_family_id: str | None = None,
) -> list[dict[str, Any]]:
    query_roles = _record_roles(query_record)
    query_aspect = _aspect(query_record)
    target_aspect = float(target_width) / max(float(target_height), 1e-6)
    target_orient = _target_orientation(target_width, target_height)
    role_to_bit = index.get("role_to_bit") or {}
    query_mask = 0
    for role in query_roles:
        bit = role_to_bit.get(role)
        if bit is not None:
            query_mask |= 1 << bit
    rows: list[dict[str, Any]] = []
    for item in index.get("entries", []):
        rid = str(item.get("record_id") or item.get("id") or "")
        fam = str(item["family_id"])
        if exclude_target_id and rid == exclude_target_id:
            continue
        if exclude_family_id and fam == exclude_family_id:
            continue
        item_mask = int(item.get("role_mask") or 0)
        if query_mask and item_mask:
            union = (query_mask | item_mask).bit_count()
            inter = (query_mask & item_mask).bit_count()
            role_jaccard = (inter / union) if union else 0.0
        else:
            roles = item.get("role_set")
            if not isinstance(roles, set):
                roles = set(roles or [])
            union = len(query_roles | roles)
            inter = len(query_roles & roles)
            role_jaccard = (inter / union) if union else 0.0
        item_aspect = float(item.get("aspect") or (item.get("canvas") or {}).get("aspect") or 1.0)
        target_aspect_similarity = _ratio_similarity(target_aspect, item_aspect)
        source_aspect_similarity = _ratio_similarity(query_aspect, item_aspect)
        item_orientation = str(item.get("orientation") or (item.get("canvas") or {}).get("orientation") or "square")
        orientation_match = 1.0 if target_orient == item_orientation else 0.0
        score = (
            0.40 * role_jaccard
            + 0.35 * target_aspect_similarity
            + 0.15 * source_aspect_similarity
            + 0.10 * orientation_match
        )
        rows.append(
            {
                "record": item,
                "record_id": rid,
                "family_id": fam,
                "score": float(score),
                "role_jaccard": float(role_jaccard),
                "target_aspect_similarity": float(target_aspect_similarity),
                "source_aspect_similarity": float(source_aspect_similarity),
                "orientation_match": float(orientation_match),
            }
        )
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[: max(0, int(k))]
