"""Build and select target layout prototypes for Layout Transformer postprocess."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from .roles import FLOATING_ROLES, TRAIN_ROLES

CHILD_PARENT = {
    "headline": "headline_group",
    "subheadline_delivery_time": "headline_group",
    "logo": "brand_group",
    "logo_back": "logo",
    "logo_fore": "logo",
    "brand_name_first_part_1": "brand_group",
    "brand_name_first_part_2": "brand_group",
    "brand_name_second": "brand_group",
}
CHILD_ROLES = tuple(CHILD_PARENT)
STRUCTURAL_ROLES = tuple(TRAIN_ROLES)
FLOATING_PROTO_ROLES = tuple(FLOATING_ROLES)
DEFAULT_PROTOTYPES_PATH = Path("layout_transformer/data/prototypes/layout_prototypes.json")


def build_prototypes(input_dir: Path) -> list[dict[str, Any]]:
    prototypes: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*_clean_fixed_semantic.json")):
        data = _load_json(path)
        frames = _coerce_frames(data)
        family_id = path.stem
        for index, frame in enumerate(frames):
            canvas = _bounds(frame)
            if canvas["width"] <= 0 or canvas["height"] <= 0:
                continue
            nodes = _flatten_semantic_nodes(frame)
            prototype = {
                "prototype_id": f"{family_id}:{index}",
                "family_id": family_id,
                "source_file": path.name,
                "frame_index": index,
                "name": frame.get("name"),
                "canvas": {
                    "width": canvas["width"],
                    "height": canvas["height"],
                    "aspect": canvas["width"] / canvas["height"],
                    "orientation": _orientation(canvas["width"], canvas["height"]),
                },
                "structural_bboxes": {},
                "role_bboxes": {},
                "child_relative_bboxes": {},
                "floating_bboxes": {},
                "text_styles": {},
            }
            for role, node in nodes.items():
                prototype["role_bboxes"][role] = _norm_bbox(_bounds(node), canvas)
            for role in STRUCTURAL_ROLES:
                node = nodes.get(role)
                if node is not None:
                    prototype["structural_bboxes"][role] = _norm_bbox(_bounds(node), canvas)
            for role, parent_role in CHILD_PARENT.items():
                child = nodes.get(role)
                parent = nodes.get(parent_role)
                if child is not None and parent is not None:
                    rel = _relative_bbox(_bounds(child), _bounds(parent))
                    if rel is not None:
                        prototype["child_relative_bboxes"][role] = rel
                        prototype["text_styles"][role] = _text_style(
                            child, role=role, parent_bounds=_bounds(parent)
                        )
            for role in FLOATING_PROTO_ROLES:
                node = nodes.get(role)
                if node is not None:
                    prototype["floating_bboxes"][role] = _norm_bbox(_bounds(node), canvas)
                    prototype["text_styles"][role] = _text_style(node, role=role, parent_bounds=canvas)
            legal = nodes.get("legal_text")
            if legal is not None:
                prototype["text_styles"]["legal_text"] = _text_style(
                    legal, role="legal_text", parent_bounds=canvas
                )
                lb = _bounds(legal)
                bg = nodes.get("background_shape")
                rel_bg = _relative_bbox(lb, _bounds(bg)) if bg is not None else None
                if rel_bg is not None:
                    prototype["legal_text_relative"] = {"anchor": "background_shape", **rel_bg}
                else:
                    rel_canvas = _relative_bbox(lb, canvas)
                    if rel_canvas is not None:
                        prototype["legal_text_relative"] = {"anchor": "canvas", **rel_canvas}
            prototypes.append(prototype)
    return prototypes


def save_prototypes(prototypes: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"version": 1, "prototypes": prototypes}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_prototypes(path: Path = DEFAULT_PROTOTYPES_PATH) -> list[dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("prototypes"), list):
        return [item for item in data["prototypes"] if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError(f"prototype file has invalid shape: {path}")


def default_clean_families_dir() -> Path:
    """Directory of ``*_clean_fixed_semantic.json`` frames used to build prototypes."""
    return Path(__file__).resolve().parent.parent / "data" / "clean_families"


def load_prototype_semantic_frame(
    prototype: dict[str, Any],
    *,
    clean_families_dir: Path | None = None,
) -> dict[str, Any]:
    """Return a deep copy of the full semantic frame for this prototype (index + file)."""
    if not isinstance(prototype, dict):
        raise TypeError("prototype must be a dict")
    source_file = prototype.get("source_file")
    if not isinstance(source_file, str) or not source_file.strip():
        raise ValueError("prototype missing source_file")
    try:
        frame_index = int(prototype.get("frame_index", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("prototype has invalid frame_index") from exc
    base = clean_families_dir or default_clean_families_dir()
    path = base / source_file
    if not path.is_file():
        raise FileNotFoundError(f"prototype source frame file not found: {path}")
    data = _load_json(path)
    frames = _coerce_frames(data)
    if frame_index < 0 or frame_index >= len(frames):
        raise IndexError(f"prototype frame_index {frame_index} out of range for {path} ({len(frames)} frames)")
    return copy.deepcopy(frames[frame_index])


def select_target_prototype(
    source_json: dict[str, Any],
    target_w: float,
    target_h: float,
    prototypes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    match = select_target_prototype_match(source_json, target_w, target_h, prototypes)
    return None if match is None else match["prototype"]


def select_target_prototype_match(
    source_json: dict[str, Any],
    target_w: float,
    target_h: float,
    prototypes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not prototypes:
        return None
    source_family = _infer_source_family(source_json, prototypes)
    target_aspect = target_w / target_h
    target_orientation = _orientation(target_w, target_h)

    def score(proto: dict[str, Any]) -> tuple[float, float, str]:
        parts = _prototype_match_parts(proto, target_w, target_h, target_aspect, target_orientation, source_family)
        return (parts["score"], parts["aspect_diff"], str(proto.get("prototype_id") or ""))

    proto = min(prototypes, key=score)
    parts = _prototype_match_parts(proto, target_w, target_h, target_aspect, target_orientation, source_family)
    return {
        "prototype": proto,
        "prototype_id": proto.get("prototype_id"),
        "score": parts["score"],
        "aspect_diff": parts["aspect_diff"],
        "width_diff_ratio": parts["width_diff_ratio"],
        "height_diff_ratio": parts["height_diff_ratio"],
        "exact_size": parts["exact_size"],
        "source_family": source_family,
    }


def _prototype_match_parts(
    proto: dict[str, Any],
    target_w: float,
    target_h: float,
    target_aspect: float,
    target_orientation: str,
    source_family: str | None,
) -> dict[str, float | bool]:
    canvas = proto.get("canvas") if isinstance(proto.get("canvas"), dict) else {}
    aspect = _num(canvas.get("aspect"), 1.0)
    orientation_penalty = 0.0 if canvas.get("orientation") == target_orientation else 5.0
    family_penalty = 0.0 if source_family and proto.get("family_id") == source_family else 2.0
    aspect_diff = abs(aspect - target_aspect)
    aspect_penalty = abs(math.log(max(0.01, aspect) / max(0.01, target_aspect)))
    proto_w = _num(canvas.get("width"), target_w)
    proto_h = _num(canvas.get("height"), target_h)
    width_diff_ratio = abs(proto_w - target_w) / max(1.0, target_w)
    height_diff_ratio = abs(proto_h - target_h) / max(1.0, target_h)
    # Prefer exact-ish dimensions after aspect/style match; it stabilizes 640x720 templates.
    dim_penalty = (width_diff_ratio + height_diff_ratio) * 0.1
    return {
        "score": orientation_penalty + family_penalty + aspect_penalty + dim_penalty,
        "aspect_diff": aspect_diff,
        "width_diff_ratio": width_diff_ratio,
        "height_diff_ratio": height_diff_ratio,
        "exact_size": abs(proto_w - target_w) < 0.5 and abs(proto_h - target_h) < 0.5,
    }


def _infer_source_family(source_json: dict[str, Any], prototypes: list[dict[str, Any]]) -> str | None:
    source_sig = _structural_signature(source_json)
    if not source_sig:
        return None
    source_canvas = _bounds(source_json)
    source_aspect = source_canvas["width"] / source_canvas["height"] if source_canvas["height"] > 0 else 1.0
    source_orientation = _orientation(source_canvas["width"], source_canvas["height"])

    best: tuple[float, str] | None = None
    for proto in prototypes:
        canvas = proto.get("canvas") if isinstance(proto.get("canvas"), dict) else {}
        sig = proto.get("structural_bboxes") if isinstance(proto.get("structural_bboxes"), dict) else {}
        if not sig:
            continue
        distance = _signature_distance(source_sig, sig)
        aspect = _num(canvas.get("aspect"), source_aspect)
        distance += abs(math.log(max(0.01, aspect) / max(0.01, source_aspect))) * 0.25
        if canvas.get("orientation") != source_orientation:
            distance += 0.75
        family = str(proto.get("family_id") or "")
        if not family:
            continue
        if best is None or distance < best[0]:
            best = (distance, family)
    return None if best is None else best[1]


def _structural_signature(frame: dict[str, Any]) -> dict[str, dict[str, float]]:
    canvas = _bounds(frame)
    if canvas["width"] <= 0 or canvas["height"] <= 0:
        return {}
    nodes = _flatten_semantic_nodes(frame)
    out: dict[str, dict[str, float]] = {}
    for role in STRUCTURAL_ROLES:
        node = nodes.get(role)
        if node is not None:
            out[role] = _norm_bbox(_bounds(node), canvas)
    return out


def _signature_distance(a: dict[str, dict[str, float]], b: dict[str, Any]) -> float:
    total = 0.0
    count = 0
    for role in STRUCTURAL_ROLES:
        ab = a.get(role)
        bb = b.get(role)
        if not isinstance(ab, dict) or not isinstance(bb, dict):
            continue
        for key in ("x", "y", "width", "height"):
            total += (_num(ab.get(key), 0.0) - _num(bb.get(key), 0.0)) ** 2
            count += 1
    return total / max(1, count)


def _flatten_semantic_nodes(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        name = node.get("name")
        if isinstance(name, str):
            current = out.get(name)
            if current is None or _area(_bounds(node)) > _area(_bounds(current)):
                out[name] = node
        for child in node.get("children") or []:
            walk(child)

    walk(frame)
    return out


def _relative_bbox(child: dict[str, float], parent: dict[str, float]) -> dict[str, float] | None:
    if parent["width"] <= 0 or parent["height"] <= 0:
        return None
    return {
        "x": (child["x"] - parent["x"]) / parent["width"],
        "y": (child["y"] - parent["y"]) / parent["height"],
        "width": child["width"] / parent["width"],
        "height": child["height"] / parent["height"],
    }


def _norm_bbox(bounds: dict[str, float], canvas: dict[str, float]) -> dict[str, float]:
    return {
        "x": (bounds["x"] - canvas["x"]) / canvas["width"],
        "y": (bounds["y"] - canvas["y"]) / canvas["height"],
        "width": bounds["width"] / canvas["width"],
        "height": bounds["height"] / canvas["height"],
    }


def default_font_name_for_role(role: str | None) -> dict[str, str]:
    """Fallback font when clean semantic JSON omits ``fontName`` (common for training exports)."""
    if role == "headline":
        return {"family": "YS Display Compressed", "style": "Heavy"}
    if role == "subheadline_delivery_time":
        return {"family": "YS Text", "style": "Medium"}
    if role == "legal_text":
        return {"family": "YS Text", "style": "Regular"}
    return {"family": "YS Text", "style": "Regular"}


def _text_style(
    node: dict[str, Any],
    *,
    role: str | None = None,
    parent_bounds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Full text style record for prototype ``text_styles`` (required keys for plugin / postprocess).

    Copies fields present on the semantic node (and optional nested ``style``). Any missing required
    keys are filled with layout-informed defaults so exports without typography still get a usable
    template.
    """
    src: dict[str, Any] = dict(node)
    nested = node.get("style")
    if isinstance(nested, dict):
        for key in (
            "fontSize",
            "fontName",
            "textAlignHorizontal",
            "textAlignVertical",
            "textAutoResize",
            "lineHeight",
            "letterSpacing",
            "fills",
            "opacity",
        ):
            if key in nested and key not in src:
                src[key] = nested[key]

    out: dict[str, Any] = {}
    font_size = src.get("fontSize")
    if isinstance(font_size, (int, float)) and math.isfinite(float(font_size)):
        out["fontSize"] = float(font_size)
    font_name = src.get("fontName")
    if isinstance(font_name, dict) and font_name.get("family") and font_name.get("style"):
        out["fontName"] = {"family": str(font_name["family"]), "style": str(font_name["style"])}
    tar = src.get("textAutoResize")
    if isinstance(tar, str) and tar.strip():
        out["textAutoResize"] = tar.strip()
    for key in ("textAlignHorizontal", "textAlignVertical"):
        value = src.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    for key in ("lineHeight", "letterSpacing"):
        if key not in src:
            continue
        val = src[key]
        if isinstance(val, (dict, list)):
            out[key] = copy.deepcopy(val)
        else:
            out[key] = val
    if "fills" in src and src["fills"] is not None:
        out["fills"] = copy.deepcopy(src["fills"])
    opacity = src.get("opacity")
    if isinstance(opacity, (int, float)) and math.isfinite(float(opacity)):
        out["opacity"] = float(opacity)

    child_bounds = _bounds(node)
    if "fontSize" not in out:
        out["fontSize"] = _inferred_font_size_from_bounds(role, child_bounds)
    if "fontName" not in out:
        out["fontName"] = default_font_name_for_role(role)
    if "textAlignHorizontal" not in out:
        if parent_bounds is not None and parent_bounds.get("width", 0.0) > 0:
            out["textAlignHorizontal"] = _infer_text_align_horizontal(child_bounds, parent_bounds)
        else:
            out["textAlignHorizontal"] = "LEFT"
    if "textAlignVertical" not in out:
        out["textAlignVertical"] = "TOP"
    if "textAutoResize" not in out:
        out["textAutoResize"] = "NONE"
    fs = float(out["fontSize"])
    if "lineHeight" not in out:
        out["lineHeight"] = {"unit": "PIXELS", "value": max(1.0, round(fs * 1.15))}
    if "letterSpacing" not in out:
        out["letterSpacing"] = {"unit": "PIXELS", "value": 0.0}
    if "fills" not in out:
        out["fills"] = [{"type": "SOLID", "color": {"r": 1.0, "g": 1.0, "b": 1.0}}]
    if "opacity" not in out:
        op = node.get("opacity")
        if isinstance(op, (int, float)) and math.isfinite(float(op)):
            out["opacity"] = float(op)
        else:
            out["opacity"] = 1.0
    return out


def _inferred_font_size_from_bounds(role: str | None, child_bounds: dict[str, float]) -> float:
    h = max(0.0, child_bounds.get("height", 0.0))
    if h <= 0:
        return 16.0
    if role == "headline":
        factor = 0.48
    elif role == "subheadline_delivery_time":
        factor = 0.44
    elif role == "legal_text":
        factor = 0.28
    else:
        factor = 0.34
    return max(8.0, min(128.0, h * factor))


def _infer_text_align_horizontal(child: dict[str, float], parent: dict[str, float]) -> str:
    pw = parent.get("width", 0.0)
    if pw <= 0:
        return "LEFT"
    cx = child.get("x", 0.0) + max(0.0, child.get("width", 0.0)) / 2.0
    pc = parent.get("x", 0.0) + pw / 2.0
    if abs(cx - pc) < pw * 0.08:
        return "CENTER"
    return "LEFT"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_frames(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("frames", "banners", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    return []


def _bounds(node: dict[str, Any] | None) -> dict[str, float]:
    raw = node.get("bounds") if isinstance(node, dict) else None
    if not isinstance(raw, dict):
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    return {
        "x": _num(raw.get("x"), 0.0),
        "y": _num(raw.get("y"), 0.0),
        "width": _num(raw.get("width"), 0.0),
        "height": _num(raw.get("height"), 0.0),
    }


def inferred_text_font_size_for_role(role: str | None, node: dict[str, Any] | None) -> float:
    """Nominal ``fontSize`` from text bounds when JSON omits explicit typography."""
    if not isinstance(node, dict):
        return _inferred_font_size_from_bounds(role, {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0})
    return _inferred_font_size_from_bounds(role, _bounds(node))


def _area(bounds: dict[str, float]) -> float:
    return max(0.0, bounds["width"]) * max(0.0, bounds["height"])


def _orientation(width: float, height: float) -> str:
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def _num(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("layout_transformer/data/clean_families"))
    parser.add_argument("--out", type=Path, default=DEFAULT_PROTOTYPES_PATH)
    args = parser.parse_args()
    prototypes = build_prototypes(args.input_dir)
    save_prototypes(prototypes, args.out)
    print(f"prototypes: {len(prototypes)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
