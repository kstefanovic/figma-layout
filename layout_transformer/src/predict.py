"""Predict target semantic role bounds for one clean semantic JSON."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from .extract import (
    copy_json_with_predicted_bounds,
    flatten_semantic_nodes,
    get_bbox_norm,
    get_canvas_size,
)
from .model import LayoutTransformer
from .postprocess import postprocess_layout
from .prototype_index import DEFAULT_PROTOTYPES_PATH, build_prototypes, load_prototypes, save_prototypes, select_target_prototype_match
from .prototype_postprocess import apply_prototype_postprocess
from .roles import NUM_ROLES, ROLE_TO_ID, TRAIN_ROLES

RAW_NAME_RE = re.compile(r"^(?:\d+|Group\s+\d+|Group\s+#+)$", re.IGNORECASE)


class StructuralLayoutTransformerService:
    """Loaded structural layout transformer plus deterministic postprocess."""

    def __init__(self, checkpoint: str | Path, device: str | None = None) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        checkpoint_payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        checkpoint_roles = checkpoint_payload.get("roles")
        if checkpoint_roles is not None and list(checkpoint_roles) != TRAIN_ROLES:
            raise ValueError(
                f"{self.checkpoint} was trained for roles {checkpoint_roles!r}; expected {TRAIN_ROLES!r}"
            )
        model_kwargs = dict(checkpoint_payload["model_kwargs"])
        model_kwargs["num_roles"] = NUM_ROLES
        self.model = LayoutTransformer(**model_kwargs).to(self.device)
        self.model.load_state_dict(checkpoint_payload["model_state"])
        self.model.eval()
        self.model_roles = list(TRAIN_ROLES)
        self.prototypes = self._load_or_build_prototypes()
        self.last_report: dict[str, Any] = {
            "transformed_children_count": 0,
            "floating_roles_placed": [],
            "font_size_fitted": 0,
            "warnings": [],
        }

    def predict(self, source_json: dict[str, Any], target_width: int | float, target_height: int | float) -> dict[str, Any]:
        final_json, report = predict_structural_layout(
            model=self.model,
            device=self.device,
            source_json=source_json,
            target_width=target_width,
            target_height=target_height,
            prototypes=self.prototypes,
            return_report=True,
        )
        self.last_report = report
        return final_json

    def _load_or_build_prototypes(self) -> list[dict[str, Any]]:
        try:
            prototypes = load_prototypes(DEFAULT_PROTOTYPES_PATH)
            if prototypes and not any(isinstance(p.get("role_bboxes"), dict) and p.get("role_bboxes") for p in prototypes):
                return self._build_and_save_prototypes()
            return prototypes
        except FileNotFoundError:
            return self._build_and_save_prototypes()
        except Exception as exc:
            print(f"Warning: failed loading layout prototypes from {DEFAULT_PROTOTYPES_PATH}: {exc}")
            return []

    def _build_and_save_prototypes(self) -> list[dict[str, Any]]:
        input_dir = Path("layout_transformer/data/clean_families")
        if not input_dir.exists():
            return []
        prototypes = build_prototypes(input_dir)
        try:
            save_prototypes(prototypes, DEFAULT_PROTOTYPES_PATH)
        except OSError as exc:
            print(f"Warning: failed saving layout prototypes to {DEFAULT_PROTOTYPES_PATH}: {exc}")
        return prototypes


def predict_structural_layout(
    *,
    model: LayoutTransformer,
    device: str,
    source_json: dict[str, Any],
    target_width: int | float,
    target_height: int | float,
    prototypes: list[dict[str, Any]] | None = None,
    return_report: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Predict structural parent roles, then deterministically postprocess children/floating roles."""
    target_w = float(target_width)
    target_h = float(target_height)
    if target_w <= 0 or target_h <= 0:
        raise ValueError("target_width and target_height must be positive")
    source_w, source_h = get_canvas_size(source_json)
    prototype_match = select_target_prototype_match(source_json, target_w, target_h, prototypes or [])
    prototype = prototype_match.get("prototype") if isinstance(prototype_match, dict) else None
    if prototype is not None and _is_strict_prototype_match(prototype_match):
        output_json = copy_json_with_predicted_bounds(source_json, {}, target_w, target_h)
        output_json, report = apply_prototype_postprocess(
            source_json=source_json,
            output_json=output_json,
            target_w=target_w,
            target_h=target_h,
            prototype=prototype,
            prototype_match=prototype_match,
            return_report=True,
        )
        validate_predicted_layout(output_json, int(round(target_w)), int(round(target_h)))
        if return_report:
            return output_json, report
        return output_json

    source_bboxes = torch.zeros((1, NUM_ROLES, 4), dtype=torch.float32)
    role_mask = torch.zeros((1, NUM_ROLES), dtype=torch.float32)
    nodes = flatten_semantic_nodes(source_json)
    for role in TRAIN_ROLES:
        node = nodes.get(role)
        if node is None:
            continue
        role_id = ROLE_TO_ID[role]
        source_bboxes[0, role_id] = torch.tensor(get_bbox_norm(node, source_w, source_h), dtype=torch.float32)
        role_mask[0, role_id] = 1.0

    with torch.no_grad():
        pred = model(
            torch.arange(NUM_ROLES, dtype=torch.long, device=device),
            source_bboxes.to(device),
            torch.tensor([[source_w, source_h]], dtype=torch.float32, device=device),
            torch.tensor([[target_w, target_h]], dtype=torch.float32, device=device),
            role_mask.to(device),
        )[0].cpu()

    pred_role_bboxes = {
        role: pred[ROLE_TO_ID[role]].tolist()
        for role in TRAIN_ROLES
        if role_mask[0, ROLE_TO_ID[role]].item() > 0
    }
    output_json = copy_json_with_predicted_bounds(source_json, pred_role_bboxes, target_w, target_h)
    if prototype is not None:
        output_json, report = apply_prototype_postprocess(
            source_json=source_json,
            output_json=output_json,
            target_w=target_w,
            target_h=target_h,
            prototype=prototype,
            prototype_match=prototype_match,
            return_report=True,
        )
    else:
        output_json, report = postprocess_layout(
            source_json=source_json,
            output_json=output_json,
            target_w=target_w,
            target_h=target_h,
            return_report=True,
        )
    validate_predicted_layout(output_json, int(round(target_w)), int(round(target_h)))
    if return_report:
        return output_json, report
    return output_json


def _is_strict_prototype_match(match: dict[str, Any] | None) -> bool:
    if not isinstance(match, dict):
        return False
    aspect_diff = _num(match.get("aspect_diff"), 999.0)
    width_diff = _num(match.get("width_diff_ratio"), 999.0)
    height_diff = _num(match.get("height_diff_ratio"), 999.0)
    exact_size = bool(match.get("exact_size"))
    return aspect_diff < 0.05 and (exact_size or (width_diff < 0.10 and height_diff < 0.10))


def _num(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def validate_predicted_layout(final_json: dict[str, Any], target_width: int, target_height: int) -> None:
    """Validate production output before returning it to the plugin."""
    width, height = get_canvas_size(final_json)
    if int(round(width)) != int(target_width) or int(round(height)) != int(target_height):
        raise ValueError(
            f"root bounds size {width}x{height} does not match target {target_width}x{target_height}"
        )

    nodes = flatten_semantic_nodes(final_json)
    missing = [role for role in TRAIN_ROLES if role not in nodes]
    if missing:
        raise ValueError(f"predicted JSON is missing required roles: {missing}")

    raw_names: list[str] = []
    invalid_bounds: list[str] = []
    for node in _walk_dict_nodes(final_json):
        name = str(node.get("name") or "")
        if RAW_NAME_RE.fullmatch(name.strip()):
            raw_names.append(name)
        bounds = node.get("bounds")
        if isinstance(bounds, dict):
            for key in ("x", "y", "width", "height"):
                value = bounds.get(key)
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    invalid_bounds.append(f"{name}.{key}={value!r}")
                    continue
                if not math.isfinite(number):
                    invalid_bounds.append(f"{name}.{key}={value!r}")

    if raw_names:
        preview = raw_names[:20]
        raise ValueError(f"predicted JSON contains raw layer names: {preview}")
    if invalid_bounds:
        preview = invalid_bounds[:20]
        raise ValueError(f"predicted JSON contains invalid bounds: {preview}")


def _walk_dict_nodes(node: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def walk(item: Any) -> None:
        if not isinstance(item, dict):
            return
        out.append(item)
        for child in item.get("children") or []:
            walk(child)

    walk(node)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-width", required=True, type=float)
    parser.add_argument("--target-height", required=True, type=float)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    service = StructuralLayoutTransformerService(args.checkpoint, args.device)
    source = _load_source(args.source_json, args.source_index)
    output_json = service.predict(source, args.target_width, args.target_height)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    print(f"Predicted structural roles: {len(service.model_roles)}")
    print(f"transformed_children_count: {service.last_report.get('transformed_children_count', 0)}")
    print(f"text_alignment: {service.last_report.get('text_alignment')}")
    print(f"text_alignment_applied: {service.last_report.get('text_alignment_applied', 0)}")
    print(f"headline_children_aligned: {service.last_report.get('headline_children_aligned', 0)}")
    print(f"font_size_fitted: {service.last_report.get('font_size_fitted', 0)}")
    print(f"floating_roles_placed: {service.last_report.get('floating_roles_placed', [])}")
    print(f"warnings: {service.last_report.get('warnings', [])}")
    print(f"Wrote: {args.out}")


def _load_source(path: Path, source_index: int) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data or not isinstance(data[0], dict):
            raise ValueError("source JSON list must contain at least one frame object")
        if source_index < 0 or source_index >= len(data):
            raise IndexError(f"--source-index {source_index} is out of range for {len(data)} source frames")
        return data[source_index]
    if isinstance(data, dict):
        return data
    raise ValueError("source JSON must contain a frame object or a list of frame objects")


if __name__ == "__main__":
    main()
