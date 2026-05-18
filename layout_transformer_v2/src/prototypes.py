"""Prototype retrieval for V2 rich layout postprocess."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .rich_utils import flatten_role_nodes, get_canvas_size, load_frames, normalized_bbox, relative_bbox, safe_float
from .schema import CHILD_PARENT, CHILD_ROLES, FLOATING_ROLES, PARENT_ROLES, orientation_id

DEFAULT_PROTOTYPES_PATH = Path("layout_transformer_v2/data/prototypes/layout_prototypes.json")
DEFAULT_RICH_FAMILIES_DIR = Path("layout_transformer/data/clean_families_rich")
STRUCTURAL_ROLES = ["hero_image", "background_shape", "brand_group", "headline_group", "legal_text"]
TEXT_STYLE_ROLES = ["headline", "subheadline_delivery_time", "legal_text", "age_badge"]
TEXT_STYLE_FIELDS = [
    "fontSize",
    "fontName",
    "textAlignHorizontal",
    "textAlignVertical",
    "textAutoResize",
    "lineHeight",
    "letterSpacing",
    "fills",
    "opacity",
    "textCase",
    "textDecoration",
]


def load_or_build_prototypes(
    path: Path = DEFAULT_PROTOTYPES_PATH,
    rich_families_dir: Path = DEFAULT_RICH_FAMILIES_DIR,
) -> list[dict[str, Any]]:
    if path.exists():
        return load_prototypes(path)
    prototypes = build_prototypes(rich_families_dir)
    save_prototypes(prototypes, path)
    return prototypes


def load_prototypes(path: Path = DEFAULT_PROTOTYPES_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        rows = data.get("prototypes")
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a prototype list")
    return [row for row in rows if isinstance(row, dict)]


def save_prototypes(prototypes: list[dict[str, Any]], path: Path = DEFAULT_PROTOTYPES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "prototypes": prototypes}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_prototypes(rich_families_dir: Path = DEFAULT_RICH_FAMILIES_DIR) -> list[dict[str, Any]]:
    paths = sorted(rich_families_dir.glob("*_clean_fixed_semantic_rich.json"))
    if not paths:
        raise FileNotFoundError(f"no rich semantic family JSONs found in {rich_families_dir}")
    prototypes: list[dict[str, Any]] = []
    for path in paths:
        frames = load_frames(path)
        family_key = path.stem.replace("_clean_fixed_semantic_rich", "")
        for idx, frame in enumerate(frames):
            try:
                prototypes.append(frame_to_prototype(frame, prototype_id=f"{family_key}:{idx}", source_file=path.name))
            except ValueError:
                continue
    return prototypes


def frame_to_prototype(frame: dict[str, Any], *, prototype_id: str, source_file: str) -> dict[str, Any]:
    width, height = get_canvas_size(frame)
    nodes = flatten_role_nodes(frame)
    bboxes = {}
    rel_bboxes = {}
    text_styles = {}
    text_content = {}
    role_types = {}
    for role, node in nodes.items():
        bboxes[role] = normalized_bbox(node, width, height)
        role_types[role] = str(node.get("type") or "")
        if role in CHILD_ROLES:
            parent = nodes.get(CHILD_PARENT.get(role, ""))
            if parent is not None:
                rel_bboxes[role] = relative_bbox(node, parent)
        if role in TEXT_STYLE_ROLES:
            style = {field: copy.deepcopy(node[field]) for field in TEXT_STYLE_FIELDS if field in node}
            if style:
                text_styles[role] = style
            if isinstance(node.get("characters"), str):
                text_content[role] = _normalize_text(node["characters"])
    return {
        "prototype_id": prototype_id,
        "source_file": source_file,
        "frame_name": frame.get("name"),
        "width": width,
        "height": height,
        "aspect": width / height,
        "orientation_id": orientation_id(width, height),
        "role_bboxes": bboxes,
        "child_relative_bboxes": rel_bboxes,
        "text_styles": text_styles,
        "text_content": text_content,
        "role_types": role_types,
    }


def select_prototype(
    prototypes: list[dict[str, Any]],
    *,
    source_json: dict[str, Any],
    target_width: float,
    target_height: float,
) -> dict[str, Any] | None:
    if not prototypes:
        return None
    source_w, source_h = get_canvas_size(source_json)
    source_nodes = flatten_role_nodes(source_json)
    source_bboxes = {
        role: normalized_bbox(source_nodes[role], source_w, source_h)
        for role in STRUCTURAL_ROLES
        if role in source_nodes
    }
    source_text = _source_text_signature(source_nodes)
    target_aspect = target_width / max(target_height, 1.0)
    target_orientation = orientation_id(target_width, target_height)
    scored = [
        (_prototype_score(proto, target_width, target_height, target_aspect, target_orientation, source_bboxes, source_text), proto)
        for proto in prototypes
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    score, proto = scored[0]
    selected = dict(proto)
    selected["match_score"] = float(score)
    return selected


def _prototype_score(
    proto: dict[str, Any],
    target_w: float,
    target_h: float,
    target_aspect: float,
    target_orientation: int,
    source_bboxes: dict[str, list[float]],
    source_text: str,
) -> float:
    width = safe_float(proto.get("width"))
    height = safe_float(proto.get("height"))
    exact_size = abs(width - target_w) <= 0.5 and abs(height - target_h) <= 0.5
    aspect = safe_float(proto.get("aspect"), width / max(height, 1.0))
    orientation_match = int(proto.get("orientation_id", -1)) == target_orientation
    aspect_score = max(0.0, 1.0 - abs(aspect - target_aspect) / max(target_aspect, 1e-6))
    structural_score = _structural_similarity(source_bboxes, proto.get("role_bboxes") or {})
    text_score = _text_similarity(source_text, " ".join((proto.get("text_content") or {}).values()))
    return (
        (1000.0 if exact_size else 0.0)
        + (100.0 if orientation_match else 0.0)
        + 25.0 * aspect_score
        + 10.0 * text_score
        + structural_score
    )


def _structural_similarity(source_bboxes: dict[str, list[float]], proto_bboxes: dict[str, Any]) -> float:
    if not source_bboxes:
        return 0.0
    distances = []
    for role, source_box in source_bboxes.items():
        proto_box = proto_bboxes.get(role)
        if not isinstance(proto_box, list) or len(proto_box) != 4:
            continue
        distances.append(sum(abs(float(a) - float(b)) for a, b in zip(source_box, proto_box)) / 4.0)
    if not distances:
        return 0.0
    return max(0.0, 1.0 - sum(distances) / len(distances))


def _source_text_signature(nodes: dict[str, dict[str, Any]]) -> str:
    pieces = []
    for role in ("headline", "subheadline_delivery_time", "legal_text"):
        node = nodes.get(role)
        if node and isinstance(node.get("characters"), str):
            pieces.append(node["characters"])
    return _normalize_text(" ".join(pieces))


def _text_similarity(a: str, b: str) -> float:
    a_tokens = set(_normalize_text(a).split())
    b_tokens = set(_normalize_text(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().replace("\n", " ").split())

