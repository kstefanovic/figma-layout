"""Family grouping and directed source-to-target pair building."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


RESOLUTION_RE = re.compile(r"(?P<w>\d{3,5})[xх](?P<h>\d{3,5})", re.IGNORECASE)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def family_key_from_filename(value: str) -> str:
    name = Path(value).name
    stem = Path(name).stem
    match = RESOLUTION_RE.search(stem)
    if match:
        prefix = stem[: match.start()].rstrip("_- .")
        return prefix or stem
    return stem


def load_families(families_path: str | Path | None, records: list[dict[str, Any]]) -> dict[str, list[str]]:
    path = Path(families_path) if families_path else None
    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("families.json must be an object")
        return {str(k): [str(x) for x in v] for k, v in data.items() if isinstance(v, list)}
    groups: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        source = str(rec.get("source_file") or rec.get("id"))
        groups[family_key_from_filename(source)].append(str(rec.get("id")))
    return dict(groups)


def group_records_by_family(
    records: list[dict[str, Any]],
    families_path: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    families = load_families(families_path, records)
    by_id = {str(r.get("id")): r for r in records}
    by_source_name = {Path(str(r.get("source_file") or "")).name: r for r in records}
    out: dict[str, list[dict[str, Any]]] = {}
    for family_id, members in families.items():
        recs: list[dict[str, Any]] = []
        for member in members:
            rec = by_id.get(member) or by_source_name.get(Path(member).name)
            if rec is not None:
                recs.append(rec)
        if recs:
            out[family_id] = recs
    return out


def _canvas_size_key(record: dict[str, Any]) -> tuple[float, float]:
    canvas = record.get("canvas") or {}
    return float(canvas.get("width") or 0), float(canvas.get("height") or 0)


def _target_lookup(tokens: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_token_id = {str(t.get("token_id")): t for t in tokens}
    by_role_lists: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for token in tokens:
        by_role_lists[str(token.get("train_role"))].append(token)
    unique_by_role = {role: items[0] for role, items in by_role_lists.items() if len(items) == 1}
    return by_token_id, unique_by_role


def build_pairs(records: list[dict[str, Any]], families_path: str | Path | None = None, min_matched_tokens: int = 3) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    families = group_records_by_family(records, families_path)
    for family_id, recs in families.items():
        for source in recs:
            for target in recs:
                if source is target or _canvas_size_key(source) == _canvas_size_key(target):
                    continue
                target_by_id, target_by_role = _target_lookup(target.get("tokens") or [])
                pair_tokens = []
                matched = 0
                for src_token in source.get("tokens") or []:
                    token_id = str(src_token.get("token_id"))
                    train_role = str(src_token.get("train_role"))
                    tgt_token = target_by_id.get(token_id) or target_by_role.get(train_role)
                    has_target = tgt_token is not None
                    if has_target:
                        matched += 1
                    pair_tokens.append(
                        {
                            "token_id": token_id,
                            "train_role": train_role,
                            "source": src_token,
                            "target_center_size_norm": (tgt_token or {}).get("center_size_norm"),
                            "has_target": has_target,
                        }
                    )
                if matched < min_matched_tokens:
                    continue
                pairs.append(
                    {
                        "pair_id": f"{family_id}:{source.get('id')}->{target.get('id')}",
                        "family_id": family_id,
                        "source_id": source.get("id"),
                        "target_id": target.get("id"),
                        "source_canvas": source.get("canvas"),
                        "target_canvas": target.get("canvas"),
                        "tokens": pair_tokens,
                    }
                )
    return pairs


def _role_occurrence_from_token_id(token_id: str) -> tuple[str, int | None]:
    clean = str(token_id or "")
    if "#" not in clean:
        return clean, None
    role, raw_idx = clean.rsplit("#", 1)
    return role, int(raw_idx) if raw_idx.isdigit() else None


def _lookup_core_target(target_tokens: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    by_token_id = {str(t.get("token_id")): t for t in target_tokens}
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_role_occurrence: dict[tuple[str, int], dict[str, Any]] = {}
    for token in target_tokens:
        role = str(token.get("train_role") or "")
        by_role[role].append(token)
        _role, occurrence = _role_occurrence_from_token_id(str(token.get("token_id") or ""))
        if occurrence is not None:
            by_role_occurrence[(role, occurrence)] = token
    unique_by_role = {role: rows[0] for role, rows in by_role.items() if len(rows) == 1}
    return by_token_id, unique_by_role, by_role_occurrence


def build_core_pairs(records: list[dict[str, Any]], families_path: str | Path | None = None, min_matched_tokens: int = 3) -> list[dict[str, Any]]:
    pairs, _stats = build_core_pairs_with_stats(records, families_path, min_matched_tokens=min_matched_tokens)
    return pairs


def build_core_pairs_with_stats(records: list[dict[str, Any]], families_path: str | Path | None = None, min_matched_tokens: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    families = group_records_by_family(records, families_path)
    total_candidate_pairs = 0
    skipped_pairs = 0
    skipped_reason_counts: dict[str, int] = defaultdict(int)
    matched_role_counts: dict[str, int] = defaultdict(int)
    matched_tokens_total = 0
    for family_id, recs in families.items():
        for source in recs:
            for target in recs:
                if source is target or _canvas_size_key(source) == _canvas_size_key(target):
                    continue
                total_candidate_pairs += 1
                tgt_by_id, tgt_unique_by_role, tgt_by_role_occ = _lookup_core_target(target.get("tokens") or [])
                pair_tokens: list[dict[str, Any]] = []
                matched = 0
                for src_token in source.get("tokens") or []:
                    token_id = str(src_token.get("token_id") or "")
                    train_role = str(src_token.get("train_role") or "")
                    _role, occurrence = _role_occurrence_from_token_id(token_id)
                    tgt = tgt_by_id.get(token_id)
                    if tgt is None:
                        tgt = tgt_unique_by_role.get(train_role)
                    if tgt is None and occurrence is not None:
                        tgt = tgt_by_role_occ.get((train_role, occurrence))
                    has_target = tgt is not None
                    if has_target:
                        matched += 1
                        matched_role_counts[train_role] += 1
                    pair_tokens.append(
                        {
                            "token_id": token_id,
                            "train_role": train_role,
                            "source": src_token,
                            "target_center_size_norm": (tgt or {}).get("center_size_norm"),
                            "target_bottom_y_norm": (
                                ((tgt or {}).get("center_size_norm") or [None, None, None, None])[1]
                                + (((tgt or {}).get("center_size_norm") or [None, None, None, None])[3] or 0.0) / 2.0
                            )
                            if has_target and train_role == "legal_group"
                            else None,
                            "has_target": has_target,
                        }
                    )
                if matched < min_matched_tokens:
                    skipped_pairs += 1
                    skipped_reason_counts["fewer_than_3_matched_core_tokens"] += 1
                    continue
                matched_tokens_total += matched
                pairs.append(
                    {
                        "pair_id": f"{family_id}:{source.get('id')}->{target.get('id')}",
                        "family_id": family_id,
                        "source_id": source.get("id"),
                        "target_id": target.get("id"),
                        "source_canvas": source.get("canvas"),
                        "target_canvas": target.get("canvas"),
                        "tokens": pair_tokens,
                    }
                )
    stats = {
        "total_candidate_pairs": total_candidate_pairs,
        "valid_pairs": len(pairs),
        "skipped_pairs": skipped_pairs,
        "skipped_reason_counts": dict(skipped_reason_counts),
        "average_matched_tokens": (matched_tokens_total / len(pairs)) if pairs else 0.0,
        "matched_role_counts": dict(matched_role_counts),
        "pair_count_by_family": dict(sorted((family_id, sum(1 for pair in pairs if str(pair.get("family_id")) == family_id)) for family_id in families)),
    }
    return pairs, stats
