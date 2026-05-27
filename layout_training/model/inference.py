"""Reusable cached inference API for top-level layout prediction."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

from layout_training.records import build_record_from_semantic_json

from .dataset import collate_fn, token_numeric_features
from .model import TopLevelLayoutTransformer
from .postprocess import apply_predictions_to_json
from .ralf_inference import blend_model_and_retrieval_predictions
from .retrieval import build_retrieval_index, build_retrieval_role_priors, retrieve_similar_layouts


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for top-level layout inference.") from exc
    return torch


def _resolve_device(torch, device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


@lru_cache(maxsize=8)
def _load_cached(checkpoint_path: str, device: str = "auto") -> dict[str, Any]:
    torch = _torch()
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Top-level layout checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu")
    config = checkpoint["model_config"]
    model = TopLevelLayoutTransformer(**config)
    model.load_state_dict(checkpoint["model_state"])
    dev = _resolve_device(torch, device)
    model.to(dev)
    model.eval()
    return {
        "checkpoint_path": str(path),
        "device": str(dev),
        "model": model,
        "role_vocab": checkpoint["role_vocab"],
        "type_vocab": checkpoint["type_vocab"],
        "model_config": config,
    }


def load_top_level_layout_checkpoint(checkpoint_path: str, device: str = "auto") -> object:
    """Load (or return cached) checkpoint bundle keyed by path+device."""
    return _load_cached(checkpoint_path, device)


@lru_cache(maxsize=8)
def _load_retrieval_index_cached(records_path: str) -> object:
    return build_retrieval_index(records_path)


def _target_canvas(width: float, height: float) -> dict[str, Any]:
    from layout_training.geometry import orientation

    return {"width": width, "height": height, "aspect": width / height, "orientation": orientation(width, height)}


def _prediction_rows(record: dict[str, Any], target_width: float, target_height: float, role_vocab: dict[str, int], type_vocab: dict[str, int]) -> list[dict[str, Any]]:
    pair = {
        "pair_id": "predict",
        "source_canvas": record["canvas"],
        "target_canvas": _target_canvas(target_width, target_height),
    }
    rows: list[dict[str, Any]] = []
    for token in record.get("tokens") or []:
        rows.append(
            {
                "x_num": token_numeric_features(pair, {"source": token}),
                "source_center": [float(x) for x in (token.get("center_size_norm") or [0.5, 0.5, 0.1, 0.1])[:4]],
                "role_id": role_vocab.get(str(token.get("train_role") or "unknown_group"), 0),
                "type_id": type_vocab.get(str(token.get("type") or "unknown"), 0),
                "target": [0.0, 0.0, 0.0, 0.0],
                "has_target": False,
                "train_role": str(token.get("train_role") or "unknown_group"),
            }
        )
    return rows


def predict_top_level_layout_json(
    semantic_json: Any,
    target_width: int,
    target_height: int,
    checkpoint_path: str,
    device: str = "auto",
    retrieval_enabled: bool = True,
    retrieval_records_path: str | None = None,
    retrieval_k: int = 5,
    retrieval_blend: bool = True,
) -> dict:
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_width and target_height must be positive integers")

    bundle = load_top_level_layout_checkpoint(checkpoint_path, device)
    torch = _torch()
    model = bundle["model"]
    role_vocab = bundle["role_vocab"]
    type_vocab = bundle["type_vocab"]
    dev = torch.device(bundle["device"])
    warnings: list[str] = []

    source_json = copy.deepcopy(semantic_json)
    record = build_record_from_semantic_json(source_json, file_id="prediction_input", include_raw_json=False)
    rows = _prediction_rows(record, float(target_width), float(target_height), role_vocab, type_vocab)
    batch = collate_fn([{"pair_id": "predict", "tokens": rows}])
    batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
    with torch.no_grad():
        pred = model(batch["x_num"], batch["role_ids"], batch["type_ids"], batch["mask"], batch["source_center"])[0].cpu().tolist()

    model_predictions_by_token_id: dict[str, list[float]] = {}
    for i, token in enumerate(record.get("tokens") or []):
        model_predictions_by_token_id[str(token.get("token_id"))] = [float(x) for x in pred[i]]

    final_predictions_by_token_id = dict(model_predictions_by_token_id)
    retrieved_examples: list[dict[str, Any]] = []
    retrieval_priors_count = 0

    if retrieval_enabled:
        try:
            default_records = Path(__file__).resolve().parents[1] / "data" / "layout_records" / "top_level_records.jsonl"
            records_path = str(Path(retrieval_records_path).expanduser().resolve()) if retrieval_records_path else str(default_records.resolve())
            index = _load_retrieval_index_cached(records_path)
            retrieved = retrieve_similar_layouts(
                query_record=record,
                target_width=int(target_width),
                target_height=int(target_height),
                index=index,
                k=int(retrieval_k),
            )
            retrieval_priors = build_retrieval_role_priors(record, retrieved)
            retrieval_priors_count = len(retrieval_priors)
            retrieved_examples = [
                {
                    "id": str((item.get("record") or {}).get("id")),
                    "score": float(item.get("score") or 0.0),
                    "canvas": (item.get("record") or {}).get("canvas"),
                    "role_jaccard": float(item.get("role_jaccard") or 0.0),
                    "target_aspect_similarity": float(item.get("target_aspect_similarity") or 0.0),
                }
                for item in retrieved
            ]
            if retrieval_blend and retrieval_priors:
                final_predictions_by_token_id = blend_model_and_retrieval_predictions(
                    model_predictions=model_predictions_by_token_id,
                    retrieval_priors=retrieval_priors,
                    tokens=record.get("tokens") or [],
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"retrieval_failed:{exc}")

    final_json = apply_predictions_to_json(
        source_json,
        record,
        final_predictions_by_token_id,
        float(target_width),
        float(target_height),
        warnings=warnings,
    )
    debug = {
        "engine": "top_level_layout_transformer_v1",
        "checkpoint": str(bundle["checkpoint_path"]),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "token_count": len(record.get("tokens") or []),
        "predicted_token_count": len(final_predictions_by_token_id),
        "device": bundle["device"],
        "retrieval_enabled": bool(retrieval_enabled),
        "retrieval_k": int(retrieval_k),
        "retrieval_records_path": str(Path(retrieval_records_path).expanduser().resolve())
        if retrieval_records_path
        else str((Path(__file__).resolve().parents[1] / "data" / "layout_records" / "top_level_records.jsonl").resolve()),
        "retrieved_examples": retrieved_examples,
        "retrieval_priors_count": retrieval_priors_count,
        "retrieval_blend": bool(retrieval_blend),
    }
    return {"final_json": final_json, "debug": debug, "warnings": warnings}
