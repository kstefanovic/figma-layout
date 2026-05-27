"""CLI-facing prediction wrapper for trained RALF model."""

from __future__ import annotations

import json
from pathlib import Path

from .inference import predict_ralf_top_level_layout_json


def parse_target_size(value: str) -> tuple[int, int]:
    clean = value.lower().replace("х", "x")
    if "x" not in clean:
        raise ValueError("--target-size must look like 600x1024")
    w, h = clean.split("x", 1)
    return int(float(w)), int(float(h))


def predict_file(
    *,
    checkpoint: str,
    input_path: str,
    target_size: str,
    records: str,
    retrieval_k: int,
    output_path: str,
    device: str = "auto",
) -> dict:
    tw, th = parse_target_size(target_size)
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = predict_ralf_top_level_layout_json(
        semantic_json=data,
        target_width=tw,
        target_height=th,
        checkpoint_path=checkpoint,
        records_path=records,
        retrieval_k=retrieval_k,
        device=device,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result["final_json"], ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out), "debug": result.get("debug", {}), "warnings": result.get("warnings", [])}

