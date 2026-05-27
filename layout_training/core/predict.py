"""Prediction helpers for the CORE layout model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inference import predict_core_top_level_layout_json


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
) -> tuple[Any, dict[str, Any]]:
    result = predict_core_top_level_layout_json(
        semantic_json=input_json,
        target_width=int(target_width),
        target_height=int(target_height),
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return result["final_json"], result["debug"]


def predict_file(
    checkpoint_path: str,
    input_path: str,
    target_size: str,
    output_path: str,
    device: str = "auto",
) -> dict[str, Any]:
    target_w, target_h = parse_target_size(target_size)
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = predict_core_top_level_layout_json(
        semantic_json=data,
        target_width=int(target_w),
        target_height=int(target_h),
        checkpoint_path=checkpoint_path,
        device=device,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result["final_json"], ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out), "debug": result["debug"], "warnings": result["warnings"]}
