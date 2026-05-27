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

