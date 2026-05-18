"""Predict a rich semantic target layout with the V2 multi-model system."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from .models.child_model import ChildLayoutTransformer
from .models.floating_model import FloatingLayoutTransformer
from .models.parent_model import ParentLayoutTransformer
from .prototypes import TEXT_STYLE_FIELDS, load_or_build_prototypes, select_prototype
from .rich_utils import (
    apply_relative_bbox,
    bounds_of,
    clamp_canvas_bbox,
    clamp_relative_bbox,
    clone_frame,
    denormalize_bbox,
    flatten_role_nodes,
    get_canvas_size,
    is_text_node,
    load_one_frame,
    node_flags,
    normalized_bbox,
    relative_bbox,
    safe_float,
    set_bounds,
    walk_nodes,
)
from .schema import (
    ALIGN_H_TO_ID,
    ALIGN_V_TO_ID,
    CHILD_PARENT,
    CHILD_ROLES,
    FLOATING_ROLES,
    ID_TO_ALIGN_H,
    ID_TO_ALIGN_V,
    PARENT_ROLES,
    UNKNOWN_NODE_TYPE,
    orientation_id,
)


MODEL_CLASSES = {
    "parent": ParentLayoutTransformer,
    "child": ChildLayoutTransformer,
    "floating": FloatingLayoutTransformer,
}


class LayoutTransformerV2Service:
    """Loaded three-model V2 layout transformer service."""

    def __init__(
        self,
        *,
        parent_checkpoint: str | Path,
        child_checkpoint: str | Path,
        floating_checkpoint: str | Path,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.parent_checkpoint = Path(parent_checkpoint)
        self.child_checkpoint = Path(child_checkpoint)
        self.floating_checkpoint = Path(floating_checkpoint)
        self.parent_model, self.parent_meta = load_model(self.parent_checkpoint, "parent", self.device)
        self.child_model, self.child_meta = load_model(self.child_checkpoint, "child", self.device)
        self.floating_model, self.floating_meta = load_model(self.floating_checkpoint, "floating", self.device)
        self.prototypes = load_or_build_prototypes()
        self.model_roles = list(PARENT_ROLES) + list(CHILD_ROLES) + list(FLOATING_ROLES)
        self.last_report: dict[str, Any] = {}

    def predict(self, source_json: dict[str, Any], target_width: int | float, target_height: int | float) -> dict[str, Any]:
        final_json, report = predict_with_loaded_models(
            source_json=source_json,
            target_width=float(target_width),
            target_height=float(target_height),
            parent_model=self.parent_model,
            parent_meta=self.parent_meta,
            child_model=self.child_model,
            child_meta=self.child_meta,
            floating_model=self.floating_model,
            floating_meta=self.floating_meta,
            prototypes=self.prototypes,
            device=self.device,
            return_report=True,
        )
        self.last_report = {
            **report,
            "postprocess_mode": "layout_transformer_v2_multi_model",
            "checkpoints": {
                "parent": str(self.parent_checkpoint),
                "child": str(self.child_checkpoint),
                "floating": str(self.floating_checkpoint),
            },
        }
        return final_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-width", required=True, type=float)
    parser.add_argument("--target-height", required=True, type=float)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--child-checkpoint", required=True, type=Path)
    parser.add_argument("--floating-checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    source = load_one_frame(args.source_json, args.source_index)
    final_json = predict_layout(
        source_json=source,
        target_width=args.target_width,
        target_height=args.target_height,
        parent_checkpoint=args.parent_checkpoint,
        child_checkpoint=args.child_checkpoint,
        floating_checkpoint=args.floating_checkpoint,
        device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {args.out}")


def predict_layout(
    *,
    source_json: dict[str, Any],
    target_width: float,
    target_height: float,
    parent_checkpoint: Path,
    child_checkpoint: Path,
    floating_checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    parent_model, parent_meta = load_model(parent_checkpoint, "parent", device)
    child_model, child_meta = load_model(child_checkpoint, "child", device)
    floating_model, floating_meta = load_model(floating_checkpoint, "floating", device)
    final_json, _report = predict_with_loaded_models(
        source_json=source_json,
        target_width=target_width,
        target_height=target_height,
        parent_model=parent_model,
        parent_meta=parent_meta,
        child_model=child_model,
        child_meta=child_meta,
        floating_model=floating_model,
        floating_meta=floating_meta,
        device=device,
        return_report=True,
    )
    return final_json


def predict_with_loaded_models(
    *,
    source_json: dict[str, Any],
    target_width: float,
    target_height: float,
    parent_model: torch.nn.Module,
    parent_meta: dict[str, Any],
    child_model: torch.nn.Module,
    child_meta: dict[str, Any],
    floating_model: torch.nn.Module,
    floating_meta: dict[str, Any],
    prototypes: list[dict[str, Any]] | None = None,
    device: str,
    return_report: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    source_w, source_h = get_canvas_size(source_json)
    target_w, target_h = float(target_width), float(target_height)
    if target_w <= 0 or target_h <= 0:
        raise ValueError("target_width and target_height must be positive")
    output = clone_frame(source_json)
    output.setdefault("bounds", {}).update({"x": 0.0, "y": 0.0, "width": target_w, "height": target_h})
    output["clipsContent"] = True

    source_nodes = flatten_role_nodes(source_json)
    out_nodes = flatten_role_nodes(output)
    predicted_bounds: dict[str, dict[str, float]] = {}
    predicted_roles: list[str] = []
    prototype = select_prototype(
        prototypes or [],
        source_json=source_json,
        target_width=target_w,
        target_height=target_h,
    )
    corrections = {
        "background_shape": False,
        "text_style_from_prototype": False,
        "floating_from_prototype": False,
        "child_relative_from_prototype": False,
    }

    for role in PARENT_ROLES:
        if role not in source_nodes or role not in out_nodes:
            continue
        pred = predict_one(parent_model, parent_meta, role, source_nodes[role], source_nodes, source_w, source_h, target_w, target_h, device)
        bbox = clamp_canvas_bbox(pred["bbox"].tolist())
        abs_bounds = denormalize_bbox(bbox, target_w, target_h)
        set_bounds(out_nodes[role], abs_bounds)
        predicted_bounds[role] = abs_bounds
        predicted_roles.append(role)
        out_nodes[role]["visible"] = bool(torch.sigmoid(pred["visibility"]).item() >= 0.5)

    if prototype is not None and target_h > target_w * 1.05:
        proto_bbox = _prototype_bbox(prototype, "background_shape")
        if proto_bbox is not None and "background_shape" in out_nodes:
            abs_bounds = denormalize_bbox(proto_bbox, target_w, target_h)
            set_bounds(out_nodes["background_shape"], abs_bounds)
            predicted_bounds["background_shape"] = abs_bounds
            corrections["background_shape"] = True

    for role in CHILD_ROLES:
        if role not in source_nodes or role not in out_nodes:
            continue
        pred = predict_one(child_model, child_meta, role, source_nodes[role], source_nodes, source_w, source_h, target_w, target_h, device)
        parent_role = CHILD_PARENT.get(role)
        parent_bounds = predicted_bounds.get(parent_role)
        if parent_bounds is None and parent_role in out_nodes:
            parent_bounds = bounds_of(out_nodes[parent_role])
        if parent_bounds and parent_bounds["width"] > 0 and parent_bounds["height"] > 0:
            proto_rel = _prototype_relative_bbox(prototype, role)
            if proto_rel is not None and role in {"headline", "subheadline_delivery_time"}:
                rel = clamp_relative_bbox(proto_rel)
                corrections["child_relative_from_prototype"] = True
            else:
                rel = clamp_relative_bbox(pred["relative_bbox"].tolist())
            abs_bounds = apply_relative_bbox(parent_bounds, rel)
            set_bounds(out_nodes[role], abs_bounds)
            predicted_bounds[role] = abs_bounds
            predicted_roles.append(role)
        apply_child_text_prediction(out_nodes[role], pred, target_w, target_h)

    for role in FLOATING_ROLES:
        if role not in source_nodes or role not in out_nodes:
            continue
        pred = predict_one(floating_model, floating_meta, role, source_nodes[role], source_nodes, source_w, source_h, target_w, target_h, device)
        bbox = clamp_canvas_bbox(pred["bbox"].tolist())
        proto_bbox = _prototype_bbox(prototype, role)
        if proto_bbox is not None:
            bbox = proto_bbox
            corrections["floating_from_prototype"] = True
        abs_bounds = denormalize_bbox(bbox, target_w, target_h)
        set_bounds(out_nodes[role], abs_bounds)
        predicted_bounds[role] = abs_bounds
        predicted_roles.append(role)
        out_nodes[role]["visible"] = bool(torch.sigmoid(pred["visibility"]).item() >= 0.5)

    if prototype is not None:
        corrections["text_style_from_prototype"] = apply_prototype_text_styles(output, prototype)
    apply_deterministic_rules(output, target_w, target_h)
    report = {
        "predicted_roles": predicted_roles,
        "prototype_id": prototype.get("prototype_id") if prototype else None,
        "prototype_match_score": prototype.get("match_score") if prototype else None,
        "corrections_applied": corrections,
        "source_canvas": {"width": source_w, "height": source_h, "aspect": source_w / source_h},
        "target_canvas": {"width": target_w, "height": target_h, "aspect": target_w / target_h},
        "parent_roles": list(PARENT_ROLES),
        "child_roles": list(CHILD_ROLES),
        "floating_roles": list(FLOATING_ROLES),
    }
    if return_report:
        return output, report
    return output


def _prototype_bbox(prototype: dict[str, Any] | None, role: str) -> list[float] | None:
    if prototype is None:
        return None
    bbox = (prototype.get("role_bboxes") or {}).get(role)
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return [float(value) for value in bbox]


def _prototype_relative_bbox(prototype: dict[str, Any] | None, role: str) -> list[float] | None:
    if prototype is None:
        return None
    bbox = (prototype.get("child_relative_bboxes") or {}).get(role)
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return [float(value) for value in bbox]


def blend_bbox(prototype_bbox: list[float], model_bbox: list[float], *, prototype_weight: float) -> list[float]:
    model_weight = 1.0 - prototype_weight
    return [
        prototype_weight * float(proto_value) + model_weight * float(model_value)
        for proto_value, model_value in zip(prototype_bbox, model_bbox)
    ]


def apply_prototype_text_styles(output: dict[str, Any], prototype: dict[str, Any]) -> bool:
    nodes = flatten_role_nodes(output)
    styles = prototype.get("text_styles") or {}
    applied = False
    for role in ("headline", "subheadline_delivery_time", "legal_text", "age_badge"):
        node = nodes.get(role)
        style = styles.get(role)
        if not node or not isinstance(style, dict):
            continue
        for field in TEXT_STYLE_FIELDS:
            if field in style:
                node[field] = json.loads(json.dumps(style[field], ensure_ascii=False))
                applied = True
    return applied


def load_model(checkpoint: Path, dataset_type: str, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("dataset_type") != dataset_type:
        raise ValueError(f"{checkpoint} contains {payload.get('dataset_type')!r}, expected {dataset_type!r}")
    model = MODEL_CLASSES[dataset_type](**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def predict_one(
    model: torch.nn.Module,
    meta: dict[str, Any],
    role: str,
    node: dict[str, Any],
    source_nodes: dict[str, dict[str, Any]],
    source_w: float,
    source_h: float,
    target_w: float,
    target_h: float,
    device: str,
) -> dict[str, torch.Tensor]:
    role_to_id = meta["role_to_id"]
    node_type_to_id = meta["node_type_to_id"]
    parent = source_nodes.get(CHILD_PARENT.get(role, ""))
    rel = relative_bbox(node, parent) if parent is not None else [0.0, 0.0, 0.0, 0.0]
    node_type = str(node.get("type") or UNKNOWN_NODE_TYPE).lower()
    source_area = max(source_w * source_h, 1.0)
    batch = {
        "role_id": torch.tensor([role_to_id[role]], dtype=torch.long, device=device),
        "node_type_id": torch.tensor([node_type_to_id.get(node_type, node_type_to_id[UNKNOWN_NODE_TYPE])], dtype=torch.long, device=device),
        "source_canvas": torch.tensor([[source_w, source_h, source_w / source_h]], dtype=torch.float32, device=device),
        "target_canvas": torch.tensor([[target_w, target_h, target_w / target_h]], dtype=torch.float32, device=device),
        "source_orientation": torch.tensor([orientation_id(source_w, source_h)], dtype=torch.long, device=device),
        "target_orientation": torch.tensor([orientation_id(target_w, target_h)], dtype=torch.long, device=device),
        "source_bbox": torch.tensor([normalized_bbox(node, source_w, source_h)], dtype=torch.float32, device=device),
        "source_relative_bbox": torch.tensor([rel], dtype=torch.float32, device=device),
        "flags": torch.tensor([node_flags(node)], dtype=torch.float32, device=device),
        "font_size": torch.tensor([float(node.get("fontSize") or 0.0) / math.sqrt(source_area)], dtype=torch.float32, device=device),
        "align_h": torch.tensor([ALIGN_H_TO_ID.get(str(node.get("textAlignHorizontal") or "UNKNOWN").upper(), ALIGN_H_TO_ID["UNKNOWN"])], dtype=torch.long, device=device),
        "align_v": torch.tensor([ALIGN_V_TO_ID.get(str(node.get("textAlignVertical") or "UNKNOWN").upper(), ALIGN_V_TO_ID["UNKNOWN"])], dtype=torch.long, device=device),
    }
    with torch.no_grad():
        return {key: value[0].detach().cpu() for key, value in model(**batch).items()}


def apply_child_text_prediction(node: dict[str, Any], pred: dict[str, torch.Tensor], target_w: float, target_h: float) -> None:
    if not is_text_node(node):
        return
    font_norm = max(0.0, float(pred["font_size"].item()))
    if font_norm > 0:
        node["fontSize"] = font_norm * math.sqrt(max(target_w * target_h, 1.0))
    node["textAlignHorizontal"] = ID_TO_ALIGN_H.get(int(pred["align_h"].argmax().item()), node.get("textAlignHorizontal", "CENTER"))
    node["textAlignVertical"] = ID_TO_ALIGN_V.get(int(pred["align_v"].argmax().item()), node.get("textAlignVertical", "CENTER"))


def apply_deterministic_rules(output: dict[str, Any], target_w: float, target_h: float) -> None:
    text_align = "CENTER" if target_h > target_w * 1.05 else "LEFT" if target_w > target_h * 1.05 else "CENTER"
    for node in walk_nodes(output):
        if node.get("name") in {"headline_group", "brand_group"}:
            node["clipsContent"] = False
        if is_text_node(node):
            node["textAlignHorizontal"] = text_align


if __name__ == "__main__":
    main()

