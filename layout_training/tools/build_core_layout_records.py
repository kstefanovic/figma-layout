"""Build canonical CORE top-level layout records JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from layout_training.records import build_core_record_from_semantic_json


def _resolve_manifest_paths(manifest_path: Path) -> list[Path]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON array of file paths")
    out: list[Path] = []
    for item in data:
        raw = str(item or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        out.append(path)
    return out


def _collect_input_paths(input_dir: str | None, manifest: str | None) -> list[Path]:
    if manifest:
        manifest_path = Path(manifest).resolve()
        return _resolve_manifest_paths(manifest_path)
    if not input_dir:
        raise ValueError("Either --input or --manifest is required")
    base = Path(input_dir)
    if base.is_file():
        return [base.resolve()]
    return sorted(path.resolve() for path in base.rglob("*.json"))


def _iter_semantic_entries(data: object) -> list[tuple[str | None, object]]:
    if isinstance(data, list):
        if not data:
            return []
        # merged_families train-*.json files are arrays of full semantic root objects
        if all(isinstance(item, dict) and isinstance(item.get("children"), list) for item in data):
            return [(str(index), item) for index, item in enumerate(data)]
    return [(None, data)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert semantic JSON files into CORE top-level layout records.")
    parser.add_argument("--input", default=None, help="Directory containing semantic JSON files, or one semantic JSON file.")
    parser.add_argument("--manifest", default=None, help="Optional JSON manifest listing semantic JSON files to include.")
    parser.add_argument("--output", required=True, help="Output records JSONL path.")
    parser.add_argument("--summary", required=True, help="Output summary JSON path.")
    parser.add_argument("--include-raw-json", action="store_true", help="Include full raw JSON in records.")
    args = parser.parse_args(argv)

    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    input_paths = _collect_input_paths(args.input, args.manifest)

    records = []
    skipped = []
    role_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    records_with_background_cluster = 0
    records_with_hero_group = 0
    records_with_brand_group = 0
    records_with_text_main_group = 0
    records_with_legal_group = 0
    text_main_clusters_with_multiple_source_paths = 0
    background_clusters_with_multiple_source_paths = 0
    rotated_hero_count = 0
    legal_group_count = 0
    total_token_count = 0
    for path in input_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            skipped.append({"file": str(path), "error": str(exc)})
            print(f"warning: skipped {path}: {exc}", file=sys.stderr)
            continue
        for entry_suffix, entry in _iter_semantic_entries(data):
            file_id = str(path) if entry_suffix is None else f"{path}::{entry_suffix}"
            try:
                rec = build_core_record_from_semantic_json(entry, file_id=file_id, include_raw_json=args.include_raw_json)
            except ValueError as exc:
                skipped.append({"file": str(path), "entry": entry_suffix, "error": str(exc)})
                print(f"warning: skipped {path} entry {entry_suffix}: {exc}", file=sys.stderr)
                continue
            if not rec.get("tokens"):
                skipped.append({"file": str(path), "entry": entry_suffix, "error": "no_valid_core_tokens"})
                print(f"warning: skipped {path} entry {entry_suffix}: no_valid_core_tokens", file=sys.stderr)
                continue
            if entry_suffix is not None:
                rec["source_file"] = str(path)
                rec["source_entry_index"] = int(entry_suffix)
            records.append(rec)
            token_roles = [str(token.get("train_role") or "") for token in rec.get("tokens") or []]
            role_counts.update(token_roles)
            excluded_counts.update(rec.get("excluded_roles") or [])
            total_token_count += len(rec.get("tokens") or [])
            if "background_cluster" in token_roles:
                records_with_background_cluster += 1
            if "hero_group" in token_roles:
                records_with_hero_group += 1
            if "brand_group" in token_roles:
                records_with_brand_group += 1
            if "text_main_group" in token_roles:
                records_with_text_main_group += 1
            if "legal_group" in token_roles:
                records_with_legal_group += 1
            for token in rec.get("tokens") or []:
                if str(token.get("train_role")) == "text_main_group" and len(token.get("source_paths") or []) > 1:
                    text_main_clusters_with_multiple_source_paths += 1
                if str(token.get("train_role")) == "background_cluster" and len(token.get("source_paths") or []) > 1:
                    background_clusters_with_multiple_source_paths += 1
                if str(token.get("train_role")) == "hero_group" and bool(token.get("is_rotated")):
                    rotated_hero_count += 1
                if str(token.get("train_role")) == "legal_group":
                    legal_group_count += 1

    with output.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "input": str(args.input) if args.input else None,
        "manifest": str(args.manifest) if args.manifest else None,
        "output": str(output),
        "input_file_count": len(input_paths),
        "total_records": len(records) + len(skipped),
        "valid_records": len(records),
        "skipped_records": len(skipped),
        "skipped_record_examples": skipped[:25],
        "count_per_core_role": dict(role_counts),
        "average_token_count": (total_token_count / len(records)) if records else 0.0,
        "records_with_background_cluster": records_with_background_cluster,
        "records_with_hero_group": records_with_hero_group,
        "records_with_brand_group": records_with_brand_group,
        "records_with_text_main_group": records_with_text_main_group,
        "records_with_legal_group": records_with_legal_group,
        "text_main_clusters_with_multiple_source_paths": text_main_clusters_with_multiple_source_paths,
        "background_clusters_with_multiple_source_paths": background_clusters_with_multiple_source_paths,
        "rotated_hero_count": rotated_hero_count,
        "legal_group_count": legal_group_count,
        "records_missing_background_cluster": len(records) - records_with_background_cluster,
        "records_missing_hero_group": len(records) - records_with_hero_group,
        "records_missing_brand_group": len(records) - records_with_brand_group,
        "records_missing_text_main_group": len(records) - records_with_text_main_group,
        "records_missing_legal_group": len(records) - records_with_legal_group,
        "record_count": len(records),
        "skipped_count": len(skipped),
        "excluded_role_counts": dict(excluded_counts),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} record(s) to {output}; summary={summary_path}")
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
