"""Build RALF training pairs by augmenting normal pairs with retrieval context."""

from __future__ import annotations

from typing import Any

from layout_training.pairs import read_jsonl, write_jsonl

from .retrieval import build_record_index, retrieve_for_query


def _retrieved_tokens(record: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for tok in record.get("tokens") or []:
        out.append(
            {
                "token_id": str(tok.get("token_id") or ""),
                "train_role": str(tok.get("train_role") or "unknown_group"),
                "center_size_norm": [float(x) for x in (tok.get("center_size_norm") or [0.5, 0.5, 0.1, 0.1])[:4]],
                "features": {
                    "type": tok.get("type"),
                    "area_ratio": float(tok.get("area_ratio") or 0.0),
                    "has_text": bool(tok.get("has_text")),
                    "has_image": bool(tok.get("has_image")),
                    "has_gradient": bool(tok.get("has_gradient")),
                    "has_star": bool(tok.get("has_star")),
                    "discount_text": bool(tok.get("discount_text")),
                    "instance_count": float(tok.get("instance_count") or 1.0),
                    "descendant_count": float(tok.get("descendant_count") or 0.0),
                    "rotation_deg": float(tok.get("rotation_deg") or 0.0),
                },
            }
        )
    return out


def build_ralf_pairs(
    *,
    records_path: str,
    pairs_path: str,
    output_path: str,
    retrieval_k: int = 5,
    exclude_same_family: bool = False,
    exclude_target_id: bool = False,
) -> dict[str, Any]:
    index = build_record_index(records_path)
    pairs = read_jsonl(pairs_path)
    out_rows: list[dict[str, Any]] = []
    for pair in pairs:
        source_id = str(pair.get("source_id") or "")
        target_id = str(pair.get("target_id") or "")
        family_id = str(pair.get("family_id") or "")
        source_record = index["by_id"].get(source_id)
        if source_record is None:
            continue
        target_canvas = pair.get("target_canvas") or {}
        tw = int(float(target_canvas.get("width") or 1))
        th = int(float(target_canvas.get("height") or 1))
        retrieved = retrieve_for_query(
            query_record=source_record,
            target_width=tw,
            target_height=th,
            index=index,
            k=retrieval_k,
            exclude_target_id=target_id if exclude_target_id else None,
            exclude_family_id=family_id if exclude_same_family else None,
        )
        retrieved_payload = []
        for item in retrieved:
            rec = item["record"]
            retrieved_payload.append(
                {
                    "record_id": str(item["record_id"]),
                    "family_id": str(item["family_id"]),
                    "score": float(item["score"]),
                    "role_jaccard": float(item["role_jaccard"]),
                    "target_aspect_similarity": float(item["target_aspect_similarity"]),
                    "source_aspect_similarity": float(item["source_aspect_similarity"]),
                    "orientation_match": float(item["orientation_match"]),
                    "canvas": rec.get("canvas"),
                    "tokens": _retrieved_tokens(rec),
                }
            )
        row = dict(pair)
        row["retrieved"] = retrieved_payload
        out_rows.append(row)
    write_jsonl(output_path, out_rows)
    return {"pair_count": len(out_rows), "output": output_path}

