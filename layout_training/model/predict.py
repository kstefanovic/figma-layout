"""Prediction helpers for top-level layout adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inference import predict_top_level_layout_json


def parse_target_size(value: str) -> tuple[float, float]:
    clean = value.lower().replace("х", "x")
    if "x" not in clean:
        raise ValueError("--target-size must look like 600x1024")
    w, h = clean.split("x", 1)
    return float(w), float(h)


def predict_json(
    *,
    checkpoint_path: str,
    input_json: Any,
    target_width: float,
    target_height: float,
    device: str = "auto",
    retrieval_enabled: bool = True,
    retrieval_records_path: str | None = None,
    retrieval_k: int = 5,
    retrieval_blend: bool = True,
) -> tuple[Any, dict[str, Any]]:
    result = predict_top_level_layout_json(
        semantic_json=input_json,
        target_width=int(target_width),
        target_height=int(target_height),
        checkpoint_path=checkpoint_path,
        device=device,
        retrieval_enabled=retrieval_enabled,
        retrieval_records_path=retrieval_records_path,
        retrieval_k=retrieval_k,
        retrieval_blend=retrieval_blend,
    )
    return result["final_json"], result["debug"]


def predict_file(
    checkpoint_path: str,
    input_path: str,
    target_size: str,
    output_path: str,
    device: str = "auto",
    retrieval_enabled: bool = True,
    retrieval_records_path: str | None = None,
    retrieval_k: int = 5,
    retrieval_blend: bool = True,
) -> dict[str, Any]:
    target_w, target_h = parse_target_size(target_size)
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    final_json, debug = predict_json(
        checkpoint_path=checkpoint_path,
        input_json=data,
        target_width=target_w,
        target_height=target_h,
        device=device,
        retrieval_enabled=retrieval_enabled,
        retrieval_records_path=retrieval_records_path,
        retrieval_k=retrieval_k,
        retrieval_blend=retrieval_blend,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out), "debug": debug}
