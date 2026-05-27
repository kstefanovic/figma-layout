"""Cached runtime inference for RALF model."""

from __future__ import annotations

from functools import lru_cache
import inspect
from pathlib import Path
from typing import Any
import time

from layout_training.records import build_record_from_semantic_json

from .dataset import build_ralf_numeric_features, ralf_collate_fn
from .model import RalfTopLevelLayoutTransformer
from .retrieval import load_retrieval_index_prefer_compact, retrieval_index_status, retrieve_for_query


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for RALF inference.") from exc
    return torch


def _resolve_device(torch, device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


@lru_cache(maxsize=8)
def _load_ckpt(checkpoint_path: str, device: str) -> dict[str, Any]:
    torch = _torch()
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"RALF checkpoint not found: {path}")
    ckpt = torch.load(str(path), map_location="cpu")
    cfg = ckpt["config"]
    model_args = set(inspect.signature(RalfTopLevelLayoutTransformer).parameters)
    model_cfg = {k: v for k, v in cfg.items() if k in model_args}
    model = RalfTopLevelLayoutTransformer(**model_cfg)
    model.load_state_dict(ckpt["model_state"])
    dev = _resolve_device(torch, device)
    model.to(dev)
    model.eval()
    return {
        "checkpoint_path": str(path),
        "device": str(dev),
        "model": model,
        "config": cfg,
        "role_vocab": ckpt["role_vocab"],
        "type_vocab": ckpt["type_vocab"],
    }


def load_ralf_checkpoint(checkpoint_path: str, device: str = "auto"):
    return _load_ckpt(checkpoint_path, device)


def get_ralf_retrieval_index_status(index_path: str) -> dict[str, Any]:
    return retrieval_index_status(index_path)


def _target_canvas(w: int, h: int) -> dict[str, Any]:
    from layout_training.geometry import orientation

    return {"width": w, "height": h, "aspect": w / max(h, 1e-6), "orientation": orientation(float(w), float(h))}


def _build_retrieved_payload(query: dict[str, Any], target_w: int, target_h: int, index: dict[str, Any], k: int):
    fam = str(query.get("source_file") or "")
    if "__" in fam:
        fam = fam.split("__", 1)[0]
    rows = retrieve_for_query(
        query_record=query,
        target_width=target_w,
        target_height=target_h,
        index=index,
        k=k,
        exclude_target_id=None,
        exclude_family_id=fam,
    )
    retrieved = []
    for row in rows:
        rec = row["record"]
        toks = []
        for t in rec.get("tokens") or []:
            toks.append(
                {
                    "token_id": str(t.get("token_id") or ""),
                    "train_role": str(t.get("train_role") or "unknown_group"),
                    "center_size_norm": [float(x) for x in (t.get("center_size_norm") or [0.5, 0.5, 0.1, 0.1])[:4]],
                    "features": {
                        "type": t.get("type"),
                        "area_ratio": float(t.get("area_ratio") or 0.0),
                        "has_text": bool(t.get("has_text")),
                        "has_image": bool(t.get("has_image")),
                        "has_gradient": bool(t.get("has_gradient")),
                        "has_star": bool(t.get("has_star")),
                        "discount_text": bool(t.get("discount_text")),
                        "instance_count": float(t.get("instance_count") or 1.0),
                        "descendant_count": float(t.get("descendant_count") or 0.0),
                        "rotation_deg": float(t.get("rotation_deg") or 0.0),
                    },
                }
            )
        retrieved.append(
            {
                "record_id": str(row["record_id"]),
                "family_id": str(row["family_id"]),
                "score": float(row["score"]),
                "canvas": rec.get("canvas"),
                "tokens": toks,
                "role_jaccard": float(row["role_jaccard"]),
                "target_aspect_similarity": float(row["target_aspect_similarity"]),
            }
        )
    return retrieved


def _source_feature_row(pair: dict[str, Any], token: dict[str, Any], role_vocab: dict[str, int], type_vocab: dict[str, int]) -> dict[str, Any]:
    return {
        "num": build_ralf_numeric_features(
            token,
            canvas=pair.get("source_canvas"),
            target_canvas=pair.get("target_canvas"),
            retrieval_score=0.0,
        ),
        "center": [float(x) for x in (token.get("center_size_norm") or [0.5, 0.5, 0.1, 0.1])[:4]],
        "role_id": role_vocab.get(str(token.get("train_role") or "unknown_group"), 0),
        "type_id": type_vocab.get(str(token.get("type") or "unknown"), 0),
        "target": [0.0, 0.0, 0.0, 0.0],
        "has_target": False,
        "train_role": str(token.get("train_role") or "unknown_group"),
        "token_id": str(token.get("token_id") or ""),
    }


def _ret_feature_row(pair: dict[str, Any], ret: dict[str, Any], tok: dict[str, Any], role_vocab: dict[str, int], type_vocab: dict[str, int]) -> dict[str, Any]:
    return {
        "num": build_ralf_numeric_features(
            tok,
            canvas=ret.get("canvas"),
            target_canvas=pair.get("target_canvas"),
            retrieval_score=float(ret.get("score") or 0.0),
        ),
        "role_id": role_vocab.get(str(tok.get("train_role") or "unknown_group"), 0),
        "type_id": type_vocab.get(str((tok.get("features") or {}).get("type") or "unknown"), 0),
    }


def predict_ralf_top_level_layout_json(
    semantic_json: dict,
    target_width: int,
    target_height: int,
    checkpoint_path: str,
    records_path: str,
    index_path: str | None = None,
    retrieval_k: int = 5,
    device: str = "auto",
) -> dict:
    bundle = load_ralf_checkpoint(checkpoint_path, device)
    index = load_retrieval_index_prefer_compact(index_path or "", records_path) if index_path else load_retrieval_index_prefer_compact("", records_path)
    role_vocab = bundle["role_vocab"]
    type_vocab = bundle["type_vocab"]
    model = bundle["model"]
    warnings: list[str] = []
    query_record = build_record_from_semantic_json(semantic_json, file_id="predict_input")
    if index.get("index_type") == "jsonl_cached":
        warnings.append("Compact RALF retrieval index missing; using cached top_level_records.jsonl fallback.")
    retrieval_t0 = time.perf_counter()
    retrieved = _build_retrieved_payload(query_record, target_width, target_height, index, retrieval_k)
    retrieval_time_ms = (time.perf_counter() - retrieval_t0) * 1000.0

    pair = {
        "source_canvas": query_record["canvas"],
        "target_canvas": _target_canvas(target_width, target_height),
    }
    src_rows = [_source_feature_row(pair, t, role_vocab, type_vocab) for t in query_record.get("tokens") or []]
    ret_rows = []
    for ret in retrieved:
        rtoks = [_ret_feature_row(pair, ret, tok, role_vocab, type_vocab) for tok in ret.get("tokens") or []]
        ret_rows.append({"score": float(ret.get("score") or 0.0), "tokens": rtoks})
    batch = ralf_collate_fn([{"source_tokens": src_rows, "retrieved": ret_rows, "pair_id": "predict"}])

    torch = _torch()
    dev = torch.device(bundle["device"])
    batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
    with torch.no_grad():
        pred = model(
            batch["src_num"],
            batch["src_role_ids"],
            batch["src_type_ids"],
            batch["src_mask"],
            batch["ret_num"],
            batch["ret_role_ids"],
            batch["ret_type_ids"],
            batch["ret_mask"],
            batch["ret_scores"],
            batch["src_center"],
        )[0].cpu().tolist()

    predictions = {}
    tokens = query_record.get("tokens") or []
    for i, token in enumerate(tokens):
        predictions[str(token.get("token_id") or "")] = [float(x) for x in pred[i]]

    from layout_training.model.postprocess import apply_predictions_to_json

    final_json = apply_predictions_to_json(
        semantic_json,
        query_record,
        predictions,
        float(target_width),
        float(target_height),
        warnings=warnings,
    )
    debug = {
        "engine": "ralf_top_level_layout_transformer_v1",
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "records_path": str(Path(records_path).expanduser().resolve()),
        "retrieval_index_path": str(Path(index_path).expanduser().resolve()) if index_path else None,
        "retrieval_index_type": str(index.get("index_type") or "unknown"),
        "retrieval_time_ms": float(retrieval_time_ms),
        "retrieval_k": int(retrieval_k),
        "retrieved_examples": [
            {"id": r.get("record_id"), "score": float(r.get("score") or 0.0), "canvas": r.get("canvas")}
            for r in retrieved
        ],
        "token_count": len(tokens),
        "predicted_token_count": len(predictions),
    }
    return {"final_json": final_json, "warnings": warnings, "debug": debug}
