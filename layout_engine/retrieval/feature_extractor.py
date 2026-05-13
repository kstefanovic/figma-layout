from __future__ import annotations

import math
from typing import Any


ORIENTATIONS = ["portrait", "balanced", "landscape", "wide", "super_wide"]
ASPECT_BUCKETS = ["portrait_tall", "portrait", "balanced", "landscape", "wide", "super_wide"]
ARCHETYPES = [
    "top_image_bottom_panel",
    "left_panel_right_image",
    "right_panel_left_image",
    "full_image_overlay",
    "mixed",
]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_bounds(node: dict) -> dict:
    b = node.get("bounds") if isinstance(node, dict) else None
    return b if isinstance(b, dict) else {}


def normalize_box(box: dict | list[float] | tuple[float, ...], root_w: float, root_h: float) -> list[float]:
    root_w = max(float(root_w or 1), 1e-6)
    root_h = max(float(root_h or 1), 1e-6)
    if isinstance(box, (list, tuple)):
        x, y, w, h = (list(box) + [0, 0, 0, 0])[:4]
    else:
        x = _num(box.get("x"))
        y = _num(box.get("y"))
        w = _num(box.get("width"))
        h = _num(box.get("height"))
    return [float(x) / root_w, float(y) / root_h, float(w) / root_w, float(h) / root_h]


def denormalize_box(norm_box: list[float] | tuple[float, ...], target_w: float, target_h: float) -> dict:
    vals = (list(norm_box) + [0, 0, 0, 0])[:4]
    return {
        "x": round(float(vals[0]) * target_w, 2),
        "y": round(float(vals[1]) * target_h, 2),
        "width": round(float(vals[2]) * target_w, 2),
        "height": round(float(vals[3]) * target_h, 2),
    }


def get_orientation(width: float, height: float) -> str:
    aspect = float(width or 1) / max(float(height or 1), 1e-6)
    if aspect < 0.75:
        return "portrait"
    if aspect < 1.4:
        return "balanced"
    if aspect < 2.5:
        return "landscape"
    if aspect < 4.0:
        return "wide"
    return "super_wide"


def get_aspect_bucket(width: float, height: float) -> str:
    aspect = float(width or 1) / max(float(height or 1), 1e-6)
    if aspect < 0.65:
        return "portrait_tall"
    if aspect < 0.85:
        return "portrait"
    if aspect < 1.4:
        return "balanced"
    if aspect < 2.5:
        return "landscape"
    if aspect < 4.0:
        return "wide"
    return "super_wide"


def get_all_text(node: dict) -> str:
    pieces: list[str] = []

    def walk(item: Any) -> None:
        if not isinstance(item, dict):
            return
        for key in ("characters", "text", "content"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                pieces.append(v.strip())
                break
        for child in item.get("children") or []:
            walk(child)

    walk(node)
    return " ".join(pieces)


def flatten_nodes(root: dict) -> list[dict]:
    out: list[dict] = []

    def walk(node: Any, depth: int, parent_id: str | None, path: str) -> None:
        if not isinstance(node, dict):
            return
        out.append(
            {
                "node": node,
                "depth": depth,
                "parent_id": parent_id,
                "path": str(node.get("path") or path),
                "bounds": get_bounds(node),
                "type": str(node.get("type") or "").lower(),
                "text": get_all_text(node),
            }
        )
        for i, child in enumerate(node.get("children") or []):
            child_path = f"{path}/{i}" if path else str(i)
            walk(child, depth + 1, str(node.get("id") or ""), child_path)

    walk(root, 0, None, "")
    return out


def _role_name(node: dict) -> str:
    return str(node.get("name") or "").strip().lower().replace("-", "_")


def _area(node: dict) -> float:
    b = get_bounds(node)
    return max(0.0, _num(b.get("width"))) * max(0.0, _num(b.get("height")))


def find_role_node(banner: dict, role: str) -> dict | None:
    matches: list[dict] = []
    for item in flatten_nodes(banner):
        node = item["node"]
        name = _role_name(node)
        if role == "hero_image":
            ok = name == "hero_image" or name.startswith("hero_image_") or name == "image_zone"
        elif role == "background_shape":
            ok = name == "background_shape" or name.startswith("background_shape")
        else:
            ok = name == role or name.startswith(role + "_")
        if ok:
            matches.append(node)
    if not matches:
        return None
    return max(matches, key=_area)


def _union_norm_boxes(boxes: list[list[float]]) -> list[float] | None:
    valid = [b for b in boxes if b and b[2] > 0 and b[3] > 0]
    if not valid:
        return None
    x0 = min(b[0] for b in valid)
    y0 = min(b[1] for b in valid)
    x1 = max(b[0] + b[2] for b in valid)
    y1 = max(b[1] + b[3] for b in valid)
    return [x0, y0, x1 - x0, y1 - y0]


def get_text_zone_box(banner: dict) -> list[float] | None:
    rb = get_bounds(banner)
    rw, rh = _num(rb.get("width")), _num(rb.get("height"))
    boxes = []
    for role in ("brand_group", "headline_group", "legal_text"):
        node = find_role_node(banner, role)
        if node:
            boxes.append(normalize_box(get_bounds(node), rw, rh))
    return _union_norm_boxes(boxes)


def detect_visual_archetype(
    hero_box: list[float] | None,
    bg_box: list[float] | None,
    text_zone_box: list[float] | None,
    width: float,
    height: float,
) -> str:
    if not hero_box or not bg_box:
        return "mixed"
    hx, hy, hw, hh = hero_box
    bx, by, bw, bh = bg_box
    hcx, hcy = hx + hw / 2, hy + hh / 2
    bcx, bcy = bx + bw / 2, by + bh / 2
    if hcy < 0.45 and bcy > 0.45:
        return "top_image_bottom_panel"
    if hcx > 0.55 and bcx < 0.50:
        return "left_panel_right_image"
    if hcx < 0.45 and bcx > 0.50:
        return "right_panel_left_image"
    if hw * hh > 0.70:
        return "full_image_overlay"
    return "mixed"


def _one_hot(value: str, values: list[str]) -> list[float]:
    return [1.0 if value == v else 0.0 for v in values]


def _box_or_zeros(box: list[float] | None) -> list[float]:
    return list(box) if box else [0.0, 0.0, 0.0, 0.0]


def _feature_vector(
    width: float,
    height: float,
    orientation: str,
    aspect_bucket: str,
    archetype: str,
    text_zone_box: list[float] | None,
    hero_box: list[float] | None = None,
    bg_box: list[float] | None = None,
) -> list[float]:
    aspect = max(float(width or 1) / max(float(height or 1), 1e-6), 1e-6)
    vec = [math.log(aspect)]
    vec.extend(_one_hot(orientation, ORIENTATIONS))
    vec.extend(_one_hot(aspect_bucket, ASPECT_BUCKETS))
    vec.extend(_one_hot(archetype, ARCHETYPES))
    vec.extend(_box_or_zeros(text_zone_box))
    vec.extend(_box_or_zeros(hero_box))
    vec.extend(_box_or_zeros(bg_box))
    vec.extend([1.0 if hero_box else 0.0, 1.0 if bg_box else 0.0, 1.0 if text_zone_box else 0.0])
    return vec


def make_clean_exemplar(banner: dict) -> dict | None:
    rb = get_bounds(banner)
    width, height = _num(rb.get("width")), _num(rb.get("height"))
    if width <= 0 or height <= 0:
        return None
    hero = find_role_node(banner, "hero_image")
    bg = find_role_node(banner, "background_shape")
    if not hero or not bg:
        return None
    hero_box = normalize_box(get_bounds(hero), width, height)
    bg_box = normalize_box(get_bounds(bg), width, height)
    text_zone = get_text_zone_box(banner)
    orientation = get_orientation(width, height)
    aspect_bucket = get_aspect_bucket(width, height)
    archetype = detect_visual_archetype(hero_box, bg_box, text_zone, width, height)
    brand = find_role_node(banner, "brand_group")
    headline = find_role_node(banner, "headline_group")
    legal = find_role_node(banner, "legal_text")
    return {
        "id": str(banner.get("id") or banner.get("path") or ""),
        "name": str(banner.get("name") or ""),
        "templateId": banner.get("templateId"),
        "width": width,
        "height": height,
        "aspect": width / height,
        "orientation": orientation,
        "aspect_bucket": aspect_bucket,
        "visual_archetype": archetype,
        "hero_image_box": hero_box,
        "background_shape_box": bg_box,
        "text_zone_box": text_zone,
        "brand_box": normalize_box(get_bounds(brand), width, height) if brand else None,
        "headline_box": normalize_box(get_bounds(headline), width, height) if headline else None,
        "legal_box": normalize_box(get_bounds(legal), width, height) if legal else None,
        "feature_vector": _feature_vector(width, height, orientation, aspect_bucket, archetype, text_zone, hero_box, bg_box),
    }


def make_runtime_query_features(
    raw_banner: dict,
    target_width: int,
    target_height: int,
    raw_candidate_info: dict | None,
) -> list[float]:
    orientation = get_orientation(target_width, target_height)
    aspect_bucket = get_aspect_bucket(target_width, target_height)
    text_zone = None
    archetype = "mixed"
    hero_box = None
    bg_box = None
    if raw_candidate_info:
        text_zone = raw_candidate_info.get("text_zone_box")
        archetype = raw_candidate_info.get("estimated_archetype") or "mixed"
        selected = raw_candidate_info.get("selected") or {}
        if selected.get("hero_image"):
            hero_box = selected["hero_image"].get("bbox_norm")
        if selected.get("background_shape"):
            bg_box = selected["background_shape"].get("bbox_norm")
    return _feature_vector(target_width, target_height, orientation, aspect_bucket, archetype, text_zone, hero_box, bg_box)

