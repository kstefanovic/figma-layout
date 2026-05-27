"""Cached inference API for the simplified CORE layout model."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

from layout_training.geometry import bbox_union, get_visual_bounds
from layout_training.records import build_core_record_from_semantic_json

from .dataset import build_role_vocab, collate_fn, token_numeric_features
from .model import CoreTopLevelLayoutTransformer
from .postprocess import apply_core_predictions_to_json


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for CORE top-level layout inference.") from exc
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
        raise FileNotFoundError(f"CORE top-level layout checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu")
    config = checkpoint["model_config"]
    model = CoreTopLevelLayoutTransformer(**config)
    model.load_state_dict(checkpoint["model_state"])
    dev = _resolve_device(torch, device)
    model.to(dev)
    model.eval()
    return {
        "checkpoint_path": str(path),
        "device": str(dev),
        "model": model,
        "role_vocab": checkpoint.get("role_vocab") or build_role_vocab(),
        "type_vocab": checkpoint.get("type_vocab") or {"<pad>": 0},
        "model_config": config,
    }


def _target_canvas(width: float, height: float) -> dict[str, Any]:
    from layout_training.geometry import orientation

    return {"width": width, "height": height, "aspect": width / height, "orientation": orientation(width, height)}


def predict_core_top_level_layout_json(
    semantic_json: Any,
    target_width: int,
    target_height: int,
    checkpoint_path: str,
    device: str = "auto",
) -> dict[str, Any]:
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_width and target_height must be positive integers")

    bundle = _load_cached(checkpoint_path, device)
    torch = _torch()
    model = bundle["model"]
    role_vocab = bundle["role_vocab"]
    dev = torch.device(bundle["device"])
    warnings: list[str] = []

    source_json = copy.deepcopy(semantic_json)
    record = build_core_record_from_semantic_json(source_json, file_id="prediction_input", include_raw_json=False)
    if not record.get("tokens"):
        raise ValueError("No valid core tokens found for CORE top-level layout prediction input")
    pair = {"pair_id": "predict", "source_canvas": record["canvas"], "target_canvas": _target_canvas(float(target_width), float(target_height))}
    rows = []
    for token in record.get("tokens") or []:
        rows.append(
            {
                "x_num": token_numeric_features(pair, {"source": token}),
                "source_center": [float(x) for x in (token.get("center_size_norm") or [0.5, 0.5, 0.1, 0.1])[:4]],
                "role_id": role_vocab.get(str(token.get("train_role") or ""), 0),
                "type_id": 0,
                "target": [0.0, 0.0, 0.0, 0.0],
                "target_bottom_y": 0.0,
                "has_target": False,
                "train_role": str(token.get("train_role") or ""),
            }
        )
    batch = collate_fn([{"pair_id": "predict", "tokens": rows}])
    batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
    with torch.no_grad():
        pred = model(batch["x_num"], batch["role_ids"], batch["mask"], batch["source_center"])[0].cpu().tolist()
    predictions_by_token_id = {
        str(token.get("token_id")): [float(x) for x in pred[idx]]
        for idx, token in enumerate(record.get("tokens") or [])
    }
    postprocess_actions: list[dict[str, Any]] = []
    final_json = apply_core_predictions_to_json(
        source_json,
        record,
        predictions_by_token_id,
        float(target_width),
        float(target_height),
        warnings=warnings,
        debug_actions=postprocess_actions,
    )
    action_by_token_id = {str(action.get("token_id") or ""): action for action in postprocess_actions if isinstance(action, dict)}
    predicted_tokens_debug: list[dict[str, Any]] = []
    skipped_unmatched_tokens: list[dict[str, Any]] = []
    root = final_json[0] if isinstance(final_json, list) and final_json and isinstance(final_json[0], dict) else final_json
    children = root.get("children") if isinstance(root, dict) else None
    for token in record.get("tokens") or []:
        token_id = str(token.get("token_id") or "")
        pred_values = predictions_by_token_id.get(token_id)
        action = action_by_token_id.get(token_id)
        token_debug = {
            "token_id": token_id,
            "train_role": str(token.get("train_role") or ""),
            "source_paths": list(token.get("source_paths") or []),
            "pred_center_size_norm": pred_values,
        }
        if action is not None:
            token_debug["postprocess_action"] = action
            token_debug["final_applied_bounds"] = action.get("final_applied_bounds")
        else:
            skipped_unmatched_tokens.append(
                {
                    "token_id": token_id,
                    "train_role": str(token.get("train_role") or ""),
                    "source_paths": list(token.get("source_paths") or []),
                    "reason": "no_postprocess_action",
                }
            )
        predicted_tokens_debug.append(token_debug)
    return {
        "final_json": final_json,
        "warnings": warnings,
        "debug": {
            "engine": "core_top_level_layout_transformer_v1",
            "checkpoint": str(bundle["checkpoint_path"]),
            "device": bundle["device"],
            "target_width": int(target_width),
            "target_height": int(target_height),
            "predicted_tokens": predicted_tokens_debug,
            "postprocess_actions": postprocess_actions,
            "skipped_unmatched_tokens": skipped_unmatched_tokens,
        },
    }
