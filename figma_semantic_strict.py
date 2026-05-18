"""
Strict canonical semantic naming for Figma banner mid JSON.

Pipeline: feature extract → deterministic prelabel → (optional Qwen on ambiguous only)
→ alias normalization → conflict resolution → validation → debug report.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from figma_semantic import (
    _AGE_BADGE_STRICT,
    _DELIVERY_MARKERS,
    _LEGAL_MARKERS,
    _bounds_area,
    build_semantic_figma_tree_from_mid,
    collect_allowed_ids_from_mid,
    lift_unassigned_wrappers_in_logo_subtrees,
    remove_non_mid_json_nodes,
    validate_final_json_ids,
)

# ---------------------------------------------------------------------------
# Canonical role schema
# ---------------------------------------------------------------------------

CANONICAL_ROLES: frozenset[str] = frozenset(
    {
        "banner_root",
        "hero_image",
        "background_shape",
        "background_gradient_1",
        "background_gradient_2",
        "brand_group",
        "logo",
        "logo_back",
        "logo_fore",
        "brand_name_first_part_1",
        "brand_name_first_part_2",
        "brand_name_second",
        "headline_group",
        "headline",
        "subheadline_delivery_time",
        "legal_text",
        "age_badge",
        "star_decoration_1",
        "star_decoration_2",
        "offer_group",
        "price_text",
        "old_price_text",
        "discount_badge",
        "product_label",
        "unassigned",
    }
)

ROLE_ALIASES: dict[str, str] = {
    "product_image": "hero_image",
    "main_image": "hero_image",
    "image_zone": "hero_image",
    "product_photo": "hero_image",
    "food_image": "hero_image",
    "person_image": "hero_image",
    "product_packshot": "hero_image",
    "hero_group": "hero_image",
    "product_group": "hero_image",
    "product_visual_group": "hero_image",
    "main_product": "hero_image",
    "drink_image": "hero_image",
    "medicine_image": "hero_image",
    "background_group": "background_shape",
    "base_background": "background_shape",
    "color_panel": "background_shape",
    "gradient_shape": "background_gradient_1",
    "background_gradient": "background_gradient_1",
    "background_gradient_3": "background_gradient_2",
    "background_gradient_4": "background_gradient_2",
    "background_gradient_5": "background_gradient_2",
    "age_badge_group": "age_badge",
    "age_badge_text": "age_badge",
    "legal_text_group": "legal_text",
    "legal_group": "legal_text",
    "headline_text": "headline",
    "headline_line": "headline",
    "subheadline": "subheadline_delivery_time",
    "subheadline_text": "subheadline_delivery_time",
    "delivery_time_text": "subheadline_delivery_time",
    "delivery_info_group": "headline_group",
    "brand_name": "brand_name_first_part_1",
    "brand_name_first": "brand_name_first_part_1",
    "brand_name_second_part_1": "brand_name_second",
    "brand_mark": "logo_fore",
    "brand_name_yandex": "brand_name_first_part_1",
    "brand_name_lavka": "brand_name_second",
    "price_value": "price_text",
    "current_price": "price_text",
    "discount_text": "discount_badge",
    "sparkle": "star_decoration_1",
    "sparkle_1": "star_decoration_1",
    "sparkle_2": "star_decoration_2",
    "star_decoration": "star_decoration_1",
    "star_decoration_3": "star_decoration_2",
    "glow_effect": "star_decoration_1",
    "glow_effect_1": "star_decoration_1",
    "shine_effect": "star_decoration_2",
    "ornament": "star_decoration_2",
    "decoration": "unassigned",
    "decoration_group": "unassigned",
    "snowflake": "star_decoration_1",
    "light_bulb": "star_decoration_2",
    "confetti": "star_decoration_2",
    "overlay_effect_group": "unassigned",
    "price_group": "offer_group",
    "old_price": "old_price_text",
    "old_price_group": "offer_group",
    "discount_badge_group": "discount_badge",
    "currency_symbol": "price_text",
}

FORBIDDEN_OUTPUT_NAMES: frozenset[str] = frozenset(
    {
        "product_image",
        "image_zone",
        "main_image",
        "product_photo",
        "food_image",
        "person_image",
        "hero_group",
        "decoration_group",
        "background_gradient",
        "gradient_shape",
        "age_badge_group",
        "legal_text_group",
    }
)

_RICH_METADATA_KEYS = frozenset(
    {
        "fills",
        "strokes",
        "effects",
        "fontSize",
        "fontName",
        "lineHeight",
        "letterSpacing",
        "characters",
        "imageHash",
        "gradientTransform",
        "opacity",
        "visible",
        "blendMode",
        "cornerRadius",
        "layoutMode",
        "textAlignHorizontal",
        "textAlignVertical",
        "textAutoResize",
    }
)

_PRICE_RE = re.compile(
    r"(₽|руб\.?|\$|€|%|\d[\d\s]*[.,]\d{2}|\d+\s*%|−\d+|старая\s+цена)",
    re.IGNORECASE | re.UNICODE,
)

STAR_NAME_HINTS = ("star", "sparkle", "glow", "shine", "snow", "bulb", "confetti", "ornament")

# Text roles that must not be demoted/promoted by headline conflict resolution or Qwen.
PROTECTED_TEXT_ROLES: frozenset[str] = frozenset(
    {
        "legal_text",
        "age_badge",
        "subheadline_delivery_time",
        "price_text",
        "old_price_text",
        "discount_badge",
        "product_label",
    }
)

STRICT_QWEN_SYSTEM_PROMPT = (
    'Output exactly one compact JSON object: {"names":{"<figma_id>":"<canonical_role>",...}}. '
    "Use ONLY these role values: "
    + ", ".join(sorted(CANONICAL_ROLES))
    + ". No markdown, no commentary."
)

STRICT_QWEN_USER_PROMPT = """You classify ambiguous banner elements from thumbnails + JSON hints.

Each ambiguous node has: id, type, bounds, optional text_preview, prelabel_guess, reason.

Rules:
- Large photo/person/product crop → hero_image (never decoration_group).
- Soft gradient/glow/mask without recognizable object → background_gradient_1 or background_gradient_2.
- Main slogan vs price: headline vs offer_group child (price_text / old_price_text / discount_badge).
- Brand row vectors/text → brand_name_first_part_* / brand_name_second / logo / logo_back / logo_fore.
- Tiny stars/sparkles → star_decoration_1 or star_decoration_2.
- If still unclear → unassigned.

Output names for every id in ambiguous_nodes only.
"""


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _node_fills(row: dict[str, Any]) -> list[dict[str, Any]]:
    fills = row.get("fills")
    return fills if isinstance(fills, list) else []


def _first_image_hash(row: dict[str, Any]) -> str | None:
    for fill in _node_fills(row):
        if isinstance(fill, dict) and fill.get("type") == "IMAGE":
            h = fill.get("imageHash")
            if h:
                return str(h)
    return None


def _has_solid_fill(row: dict[str, Any]) -> bool:
    return any(
        isinstance(f, dict) and str(f.get("type") or "") == "SOLID" and f.get("visible", True) is not False
        for f in _node_fills(row)
    )


def _mid_has_image_descendant(
    mid_by_id: dict[str, dict[str, Any]],
    features: dict[str, dict[str, Any]],
    sid: str,
) -> bool:
    stack = [str(c) for c in (features.get(sid, {}).get("mid_child_ids") or [])]
    seen: set[str] = set()
    while stack:
        cid = stack.pop()
        if cid in seen or cid not in mid_by_id:
            continue
        seen.add(cid)
        cf = features.get(cid, {})
        if cf.get("has_image_fill") or cf.get("imageHash"):
            return True
        for gc in cf.get("mid_child_ids") or []:
            stack.append(str(gc))
    return False


def _background_shape_priority(
    sid: str,
    row: dict[str, Any],
    feat: dict[str, Any],
    mid_by_id: dict[str, dict[str, Any]],
    features: dict[str, dict[str, Any]],
    frame_w: float,
) -> float:
    """Higher score = better main color-plate candidate (not hero wrappers or gradients)."""
    if _mid_has_image_descendant(mid_by_id, features, sid):
        return -1.0
    if feat.get("has_gradient_fill"):
        return -1.0
    typ = _norm_type(row)
    if typ == "instance":
        return -1.0
    area = float(feat.get("area") or 0)
    score = area
    if typ == "vector" and _has_solid_fill(row):
        score *= 2.25
    if typ == "rectangle" and _has_solid_fill(row):
        score *= 1.75
    try:
        bw = float((feat.get("bounds") or {}).get("width") or 0)
    except (TypeError, ValueError):
        bw = 0.0
    if frame_w > 0 and bw / frame_w >= 0.4:
        score *= 1.6
    return score


def _gradient_info(row: dict[str, Any]) -> tuple[bool, str | None]:
    for fill in _node_fills(row):
        if not isinstance(fill, dict):
            continue
        t = str(fill.get("type") or "")
        if "GRADIENT" in t.upper():
            return True, t
    return False, None


def _norm_type(row: dict[str, Any]) -> str:
    return str(row.get("type") or "").lower().replace("_", " ")


def _frame_metrics(mid_blocks: list[dict[str, Any]]) -> tuple[float, float, float]:
    root = None
    for b in mid_blocks:
        if isinstance(b, dict) and not (b.get("mid_parent_ids") or []):
            root = b
            break
    bounds = (root or {}).get("bounds") or {}
    w = float(bounds.get("width") or 1)
    h = float(bounds.get("height") or 1)
    return w, h, max(1.0, w * h)


def extract_node_features(mid_blocks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Rich geometric + fill + text features per mid row id."""
    frame_w, frame_h, frame_area = _frame_metrics(mid_blocks)
    out: dict[str, dict[str, Any]] = {}

    for row in mid_blocks:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        sid = str(row["id"])
        bounds = row.get("bounds") if isinstance(row.get("bounds"), dict) else {}
        try:
            bx = float(bounds.get("x") or 0)
            by = float(bounds.get("y") or 0)
            bw = float(bounds.get("width") or 0)
            bh = float(bounds.get("height") or 0)
        except (TypeError, ValueError):
            bx = by = bw = bh = 0.0
        area = max(0.0, bw * bh)
        area_ratio = area / frame_area if frame_area > 0 else 0.0
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        cx_ratio = cx / frame_w if frame_w > 0 else 0.5
        cy_ratio = cy / frame_h if frame_h > 0 else 0.5

        chars = row.get("characters")
        has_text = _norm_type(row) == "text" or isinstance(chars, str)
        char_str = str(chars) if isinstance(chars, str) else ""
        char_len = len(char_str.strip())

        has_grad, grad_type = _gradient_info(row)
        img_hash = _first_image_hash(row)
        has_image = img_hash is not None or any(
            isinstance(f, dict) and f.get("type") == "IMAGE" for f in _node_fills(row)
        )

        effects = row.get("effects")
        has_effect = isinstance(effects, list) and len(effects) > 0

        child_ids = [str(c) for c in (row.get("mid_child_ids") or [])]
        is_large = area_ratio >= 0.12 or bw >= frame_w * 0.35 or bh >= frame_h * 0.35
        is_tiny = area_ratio <= 0.004 and max(bw, bh) <= max(frame_w, frame_h) * 0.08

        out[sid] = {
            "id": sid,
            "path": row.get("path"),
            "name": row.get("name"),
            "type": row.get("type"),
            "bounds": bounds,
            "area": area,
            "area_ratio_to_canvas": area_ratio,
            "center_x_ratio": cx_ratio,
            "center_y_ratio": cy_ratio,
            "has_text": has_text,
            "characters": char_str,
            "fontSize": row.get("fontSize"),
            "fontName": row.get("fontName"),
            "text_length": char_len,
            "has_image_fill": has_image,
            "imageHash": img_hash,
            "has_gradient_fill": has_grad,
            "gradient_type": grad_type,
            "has_effect": has_effect,
            "opacity": row.get("opacity"),
            "visible": row.get("visible", True),
            "child_count": len(child_ids),
            "mid_child_ids": child_ids,
            "mid_parent_ids": [str(p) for p in (row.get("mid_parent_ids") or [])],
            "is_large": is_large,
            "is_tiny": is_tiny,
            "is_top": cy_ratio < 0.28,
            "is_bottom": cy_ratio > 0.72,
            "is_left": cx_ratio < 0.32,
            "is_right": cx_ratio > 0.68,
        }
    return out


# ---------------------------------------------------------------------------
# Deterministic prelabel
# ---------------------------------------------------------------------------


def _norm_chars(s: str) -> str:
    return " ".join((s or "").replace("\r", "\n").split()).strip()


def _is_age_badge_text(chars: str) -> bool:
    return bool(_AGE_BADGE_STRICT.match(_norm_chars(chars).replace(" ", "")))


def _is_legal_text(chars: str, feat: dict[str, Any]) -> bool:
    low = _norm_chars(chars).lower()
    if not low:
        return False
    if any(m.lower() in low for m in _LEGAL_MARKERS):
        return True
    fs = feat.get("fontSize")
    try:
        font_size = float(fs) if fs is not None else 0.0
    except (TypeError, ValueError):
        font_size = 0.0
    th = float((feat.get("bounds") or {}).get("height") or 0)
    char_len = len(low)
    return char_len >= 80 and font_size > 0 and font_size <= 28 and th > 0 and th <= 120


def _is_delivery_text(chars: str) -> bool:
    low = _norm_chars(chars).lower()
    return any(m in low for m in _DELIVERY_MARKERS)


def _is_price_text(chars: str) -> bool:
    return bool(_PRICE_RE.search(_norm_chars(chars)))


def _star_candidate(row: dict[str, Any], feat: dict[str, Any]) -> bool:
    typ = _norm_type(row)
    if typ == "star":
        return True
    nm = str(row.get("name") or "").lower()
    if any(h in nm for h in STAR_NAME_HINTS):
        return True
    if feat.get("is_tiny") and typ in ("vector", "star", "ellipse", "boolean operation"):
        return True
    return False


def _under_named_parent(
    sid: str,
    names: dict[str, str],
    features: dict[str, dict[str, Any]],
    parent_role: str,
) -> bool:
    for pid in features.get(sid, {}).get("mid_parent_ids") or []:
        if names.get(str(pid)) == parent_role:
            return True
    return False


def prelabel_roles(
    mid_blocks: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Deterministic first-pass role assignment."""
    mid_by_id = {str(b["id"]): b for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}
    frame_w, frame_h, frame_area = _frame_metrics(mid_blocks)
    roles: dict[str, str] = {}

    text_headline_scores: list[tuple[float, str]] = []

    for sid, row in mid_by_id.items():
        feat = features.get(sid, {})
        typ = _norm_type(row)

        if not (row.get("mid_parent_ids") or []):
            roles[sid] = "banner_root"
            continue

        if typ == "text":
            chars = feat.get("characters") or ""
            if _is_age_badge_text(chars):
                roles[sid] = "age_badge"
                continue
            if _is_legal_text(chars, feat):
                roles[sid] = "legal_text"
                continue
            if _is_delivery_text(chars):
                roles[sid] = "subheadline_delivery_time"
                continue
            if _is_price_text(chars):
                if "скидк" in chars.lower() or "%" in chars:
                    roles[sid] = "discount_badge"
                elif "стара" in chars.lower() or "было" in chars.lower():
                    roles[sid] = "old_price_text"
                else:
                    roles[sid] = "price_text"
                continue
            try:
                fs = float(feat.get("fontSize") or 0)
            except (TypeError, ValueError):
                fs = 0.0
            th = float((feat.get("bounds") or {}).get("height") or 0)
            score = fs * 2.0 + th * 0.05 + len(chars) * 0.2
            text_headline_scores.append((score, sid))
            roles[sid] = "unassigned"
            continue

        if feat.get("has_gradient_fill"):
            roles[sid] = "background_gradient_1"
            continue

        if feat.get("has_image_fill") and feat.get("is_large"):
            roles[sid] = "hero_image"
            continue

        if typ == "instance":
            child_ids = [str(c) for c in (feat.get("mid_child_ids") or []) if str(c) in mid_by_id]
            text_kids = [c for c in child_ids if _norm_type(mid_by_id[c]) == "text"]
            if len(text_kids) >= 1:
                roles[sid] = "headline_group"
                continue
            if any(features.get(c, {}).get("has_image_fill") for c in child_ids):
                roles[sid] = "hero_image"
                continue

        if typ in ("frame", "group"):
            child_ids = feat.get("mid_child_ids") or []
            child_rows = [mid_by_id[c] for c in child_ids if c in mid_by_id]
            if child_rows:
                has_img_child = any(features.get(str(c.get("id")), {}).get("has_image_fill") for c in child_rows)
                if has_img_child and len(child_rows) == 1:
                    roles[sid] = "hero_image"
                    continue
                text_kids = [c for c in child_rows if _norm_type(c) == "text"]
                if len(text_kids) >= 2:
                    roles[sid] = "headline_group"
                    continue
                brandish = sum(
                    1
                    for c in child_rows
                    if _norm_type(c) in ("vector", "boolean operation", "text")
                    or "brand" in str(c.get("name") or "").lower()
                )
                if brandish >= 2 and feat.get("is_top"):
                    roles[sid] = "brand_group"
                    continue
                if brandish >= 2:
                    roles[sid] = "brand_group"
                    continue

        if typ == "star":
            roles[sid] = "star_decoration_1"
            continue

        if typ in ("rectangle", "vector", "ellipse", "boolean operation"):
            if feat.get("has_image_fill"):
                roles[sid] = "hero_image"
                continue
            if typ != "star" and feat.get("is_large") and not feat.get("has_gradient_fill"):
                roles[sid] = "background_shape"
                continue

        roles.setdefault(sid, "unassigned")

    if text_headline_scores:
        text_headline_scores.sort(reverse=True)
        roles[text_headline_scores[0][1]] = "headline"
        for _score, sid in text_headline_scores[1:]:
            if roles.get(sid) == "unassigned":
                child_row = mid_by_id.get(sid)
                if child_row and _is_delivery_text(features.get(sid, {}).get("characters") or ""):
                    roles[sid] = "subheadline_delivery_time"
                elif roles.get(sid) == "unassigned":
                    roles[sid] = "subheadline_delivery_time"

    return roles


# ---------------------------------------------------------------------------
# Ambiguous detection + Qwen prompt
# ---------------------------------------------------------------------------


def identify_ambiguous_nodes(
    mid_blocks: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    prelabel: dict[str, str],
) -> list[str]:
    ambiguous: list[str] = []
    mid_by_id = {str(b["id"]): b for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}

    for sid, role in prelabel.items():
        if role == "banner_root":
            continue
        feat = features.get(sid, {})
        row = mid_by_id.get(sid, {})

        if role == "unassigned":
            ambiguous.append(sid)
            continue

        if feat.get("has_image_fill") and role not in ("hero_image", "background_gradient_1", "background_gradient_2"):
            ambiguous.append(sid)
            continue

        if role in ("hero_image", "background_shape", "background_gradient_1") and feat.get("is_tiny"):
            ambiguous.append(sid)
            continue

        if _norm_type(row) == "text" and role in ("unassigned", "headline", "price_text", "product_label"):
            ambiguous.append(sid)
            continue

        nm = str(row.get("name") or "").lower()
        if any(x in nm for x in ("group", "rectangle", "vector")) and role == "unassigned":
            ambiguous.append(sid)

    return sorted(set(ambiguous))


def build_qwen_ambiguous_payload(
    mid_blocks: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    prelabel: dict[str, str],
    ambiguous_ids: list[str],
) -> list[dict[str, Any]]:
    mid_by_id = {str(b["id"]): b for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}
    nodes: list[dict[str, Any]] = []
    for sid in ambiguous_ids:
        row = mid_by_id.get(sid)
        feat = features.get(sid, {})
        if not row:
            continue
        item: dict[str, Any] = {
            "id": sid,
            "type": row.get("type"),
            "bounds": row.get("bounds"),
            "prelabel_guess": prelabel.get(sid, "unassigned"),
            "has_image_fill": feat.get("has_image_fill"),
            "has_gradient_fill": feat.get("has_gradient_fill"),
            "area_ratio_to_canvas": round(float(feat.get("area_ratio_to_canvas") or 0), 4),
        }
        ch = feat.get("characters") or ""
        if ch:
            item["text_preview"] = ch[:200]
        nodes.append(item)
    return nodes


def build_qwen_ambiguous_user_text(ambiguous_nodes: list[dict[str, Any]]) -> str:
    payload = json.dumps({"ambiguous_nodes": ambiguous_nodes}, ensure_ascii=False, separators=(",", ":"))
    return STRICT_QWEN_USER_PROMPT + "\n\n" + payload


# ---------------------------------------------------------------------------
# Normalization + conflict resolution
# ---------------------------------------------------------------------------


def _canonicalize_role(name: str) -> str:
    raw = str(name or "").strip().lower()
    if not raw:
        return "unassigned"
    if raw in CANONICAL_ROLES:
        return raw
    if raw in ROLE_ALIASES:
        return ROLE_ALIASES[raw]
    # suffix variants
    for alias, target in ROLE_ALIASES.items():
        if raw.startswith(alias + "_"):
            return target
    return "unassigned"


def normalize_roles(
    mid_blocks: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    names: dict[str, str],
) -> dict[str, str]:
    allowed = collect_allowed_ids_from_mid(mid_blocks)
    out: dict[str, str] = {}
    for sid in allowed:
        role = _canonicalize_role(names.get(sid, "unassigned"))
        out[sid] = role

    mid_by_id = {str(b["id"]): b for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}

    for sid, row in mid_by_id.items():
        feat = features.get(sid, {})
        chars = feat.get("characters") or ""
        if _norm_type(row) == "text":
            if _is_age_badge_text(chars):
                out[sid] = "age_badge"
            elif _is_legal_text(chars, feat):
                out[sid] = "legal_text"
            elif _is_delivery_text(chars):
                out[sid] = "subheadline_delivery_time"

        if feat.get("has_gradient_fill"):
            if out.get(sid) == "background_shape":
                out[sid] = "background_gradient_1"

        if feat.get("has_image_fill") and out.get(sid) in FORBIDDEN_OUTPUT_NAMES | {"background_shape", "unassigned"}:
            if feat.get("is_large"):
                out[sid] = "hero_image"

    return out


def resolve_role_conflicts(
    mid_blocks: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    names: dict[str, str],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Pick single hero, dedupe gradients, fix decoration swallowing hero."""
    fixes: list[dict[str, str]] = []
    out = dict(names)
    mid_by_id = {str(b["id"]): b for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}
    _, _, frame_area = _frame_metrics(mid_blocks)

    def _fix(sid: str, new: str, reason: str) -> None:
        old = out.get(sid, "")
        if old != new:
            out[sid] = new
            fixes.append({"id": sid, "old": old, "new": new, "reason": reason})

    # Hero: largest image-like wins
    hero_candidates: list[tuple[float, str]] = []
    for sid, role in list(out.items()):
        feat = features.get(sid, {})
        if role == "hero_image" or (feat.get("has_image_fill") and feat.get("is_large")):
            area = float(feat.get("area") or 0)
            hero_candidates.append((area, sid))
    hero_candidates.sort(reverse=True)
    if hero_candidates:
        winner = hero_candidates[0][1]
        for area, sid in hero_candidates[1:]:
            if sid == winner:
                continue
            feat = features.get(sid, {})
            if feat.get("is_tiny"):
                _fix(sid, "star_decoration_1", "demote_tiny_hero_candidate")
            else:
                _fix(sid, "unassigned", "demote_extra_hero_candidate")
        _fix(winner, "hero_image", "promote_largest_hero")
    else:
        # No hero — promote largest imageHash node
        img_nodes = [
            (float(features[s].get("area") or 0), s)
            for s in features
            if features[s].get("imageHash")
        ]
        img_nodes.sort(reverse=True)
        if img_nodes:
            _fix(img_nodes[0][1], "hero_image", "promote_largest_image_hash")

    # Product photo wrapper (instance/frame/group) with one large image child → wrapper is hero_image
    for sid, row in mid_by_id.items():
        if sid not in mid_by_id:
            continue
        parent_role = out.get(sid, "")
        if parent_role in ("banner_root", "brand_group", "headline_group", "offer_group"):
            continue
        typ = _norm_type(row)
        if parent_role not in ("unassigned", "background_shape") and typ not in ("instance", "frame", "group"):
            continue
        child_ids = [str(c) for c in (features.get(sid, {}).get("mid_child_ids") or []) if str(c) in mid_by_id]
        if len(child_ids) != 1:
            continue
        cid = child_ids[0]
        cf = features.get(cid, {})
        if not (cf.get("has_image_fill") or cf.get("imageHash")):
            continue
        if not cf.get("is_large") and not cf.get("imageHash"):
            continue
        if out.get(cid) == "hero_image" or parent_role in ("unassigned", "background_shape"):
            _fix(sid, "hero_image", "hero_wrapper_is_hero")
            if out.get(cid) == "hero_image":
                _fix(cid, "unassigned", "hero_inner_image_leaf")

    # background_shape: best solid color plate (not hero instance wrappers or gradients)
    frame_w, _frame_h, _frame_area = _frame_metrics(mid_blocks)
    bg_candidates: list[tuple[float, str]] = []
    grad_ids: list[str] = []
    for sid, role in out.items():
        if role == "banner_root":
            continue
        row = mid_by_id.get(sid, {})
        feat = features.get(sid, {})
        if feat.get("has_gradient_fill"):
            grad_ids.append(sid)
            if role == "background_shape":
                _fix(sid, "background_gradient_1", "gradient_not_background_shape")
            continue
        priority = _background_shape_priority(sid, row, feat, mid_by_id, features, frame_w)
        if priority > 0 and (
            role == "background_shape"
            or (
                not feat.get("has_image_fill")
                and feat.get("is_large")
                and _norm_type(row) not in ("text", "star")
            )
        ):
            bg_candidates.append((priority, sid))
    bg_candidates.sort(reverse=True)
    if bg_candidates:
        keep = bg_candidates[0][1]
        _fix(keep, "background_shape", "promote_largest_background")
        for _prio, sid in bg_candidates[1:]:
            if out.get(sid) == "background_shape":
                _fix(sid, "unassigned", "demote_extra_background_shape")

    for sid, row in mid_by_id.items():
        if out.get(sid) != "background_shape":
            continue
        if _mid_has_image_descendant(mid_by_id, features, sid) or _norm_type(row) == "instance":
            _fix(sid, "unassigned", "hero_wrapper_not_background_shape")

    grad_ids = [s for s in features if features[s].get("has_gradient_fill")]
    grad_ids.sort(key=lambda s: (features[s].get("center_y_ratio", 0), features[s].get("center_x_ratio", 0)))
    for i, sid in enumerate(grad_ids[:2]):
        want = f"background_gradient_{i + 1}"
        if out.get(sid) != want:
            _fix(sid, want, "number_gradients")

    # Headline: only among headline candidates — never steal legal/age/delivery/price roles.
    headlines = [
        (
            float(features[s].get("fontSize") or 0) * max(1.0, float(features[s].get("text_length") or 1)),
            s,
        )
        for s, r in out.items()
        if r == "headline"
        or (
            r not in PROTECTED_TEXT_ROLES
            and _norm_type(mid_by_id.get(s, {})) == "text"
            and float(features[s].get("fontSize") or 0) >= 40
        )
    ]
    headlines.sort(reverse=True)
    if headlines:
        keep_h = headlines[0][1]
        _fix(keep_h, "headline", "promote_main_headline")
        for _sc, sid in headlines[1:]:
            if out.get(sid) in PROTECTED_TEXT_ROLES:
                continue
            if out.get(sid) == "headline":
                if _is_delivery_text(features[sid].get("characters") or ""):
                    _fix(sid, "subheadline_delivery_time", "demote_extra_headline_to_delivery")
                elif _is_price_text(features[sid].get("characters") or ""):
                    _fix(sid, "price_text", "demote_extra_headline_to_price")
                else:
                    _fix(sid, "unassigned", "demote_extra_headline")

    # Offer vs headline: price text cannot be headline
    for sid, role in list(out.items()):
        if role != "headline":
            continue
        if _is_price_text(features.get(sid, {}).get("characters") or ""):
            _fix(sid, "price_text", "price_not_headline")

    # Group containers
    for sid, row in mid_by_id.items():
        child_ids = features.get(sid, {}).get("mid_child_ids") or []
        if not child_ids:
            continue
        child_roles = [out.get(c, "") for c in child_ids]
        if "headline" in child_roles or "subheadline_delivery_time" in child_roles:
            if out.get(sid) in ("unassigned", "headline", "delivery_info_group"):
                _fix(sid, "headline_group", "wrap_headline_children")
        brand_parts = sum(1 for r in child_roles if r.startswith("brand_name") or r in ("logo", "logo_back", "logo_fore"))
        if brand_parts >= 2:
            _fix(sid, "brand_group", "wrap_brand_children")
        price_parts = sum(1 for r in child_roles if r in ("price_text", "old_price_text", "discount_badge", "product_label"))
        if price_parts >= 2:
            _fix(sid, "offer_group", "wrap_offer_children")

    # Tiny nodes cannot be hero or background_shape (skip brand-row wordmarks / logo marks)
    for sid, feat in features.items():
        if not feat.get("is_tiny"):
            continue
        if any(out.get(str(p)) == "brand_group" for p in (feat.get("mid_parent_ids") or [])):
            continue
        if out.get(sid) == "hero_image":
            _fix(sid, "star_decoration_1", "tiny_not_hero")
        if out.get(sid) == "background_shape":
            _fix(sid, "star_decoration_2", "tiny_not_background")

    out, fixes = assign_hero_wrapper_roles(mid_blocks, features, out, fixes)

    return out, fixes


def assign_hero_wrapper_roles(
    mid_blocks: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    names: dict[str, str],
    fixes: list[dict[str, str]],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Ensure hero photo wrappers are ``hero_image``, not ``unassigned`` parents of ``hero_image``."""
    out = dict(names)
    mid_by_id = {str(b["id"]): b for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}

    def _fix(sid: str, new: str, reason: str) -> None:
        old = out.get(sid, "")
        if old != new:
            out[sid] = new
            fixes.append({"id": sid, "old": old, "new": new, "reason": reason})

    heroes = [sid for sid, role in out.items() if role == "hero_image"]
    for hid in heroes:
        parents = features.get(hid, {}).get("mid_parent_ids") or []
        if not parents:
            continue
        pid = str(parents[-1])
        if pid not in mid_by_id or out.get(pid) == "banner_root":
            continue
        if out.get(pid) == "unassigned":
            _fix(pid, "hero_image", "hero_parent_unassigned")
            _fix(hid, "unassigned", "hero_inner_leaf")

    for sid, row in mid_by_id.items():
        if out.get(sid) == "banner_root":
            continue
        child_ids = [str(c) for c in (row.get("mid_child_ids") or []) if str(c) in mid_by_id]
        if len(child_ids) != 1:
            continue
        cid = child_ids[0]
        if out.get(cid) != "hero_image":
            continue
        if out.get(sid) in ("unassigned", "background_shape"):
            _fix(sid, "hero_image", "hero_wrapper_is_hero")
            _fix(cid, "unassigned", "hero_inner_leaf")

    return out, fixes


def _bounds_x(row: dict[str, Any]) -> float:
    b = row.get("bounds") or {}
    try:
        return float(b.get("x") or 0)
    except (TypeError, ValueError):
        return 0.0


def _is_logo_like_cluster(row: dict[str, Any], mid_by_id: dict[str, dict[str, Any]]) -> bool:
    typ = _norm_type(row)
    subs = [str(x) for x in (row.get("mid_child_ids") or []) if str(x) in mid_by_id]
    if typ in ("frame", "group", "instance"):
        vecs = [x for x in subs if _norm_type(mid_by_id[x]) == "vector"]
        return 1 <= len(vecs) <= 2 and len(vecs) == len(subs)
    if typ == "boolean operation":
        if not subs:
            return False
        return all(_norm_type(mid_by_id[x]) in ("vector", "boolean operation") for x in subs)
    return False


def _mid_descendant_ids(start: str, mid_by_id: dict[str, dict[str, Any]]) -> set[str]:
    out_ids: set[str] = set()
    stack = [start]
    seen: set[str] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        out_ids.add(cur)
        row = mid_by_id.get(cur)
        if not row:
            continue
        for xid in row.get("mid_child_ids") or []:
            xs = str(xid)
            if xs in mid_by_id:
                stack.append(xs)
    return out_ids


def _leaf_vector_ids_under(root_id: str, mid_by_id: dict[str, dict[str, Any]]) -> list[str]:
    acc: list[str] = []
    stack = [root_id]
    seen: set[str] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur != root_id:
            sub = mid_by_id.get(cur)
            if sub and _norm_type(sub) == "vector" and not (sub.get("mid_child_ids") or []):
                acc.append(cur)
                continue
        subw = mid_by_id.get(cur)
        if not subw:
            continue
        for xid in subw.get("mid_child_ids") or []:
            xs = str(xid)
            if xs in mid_by_id:
                stack.append(xs)
    return acc


def _pick_brand_row_logo_id(child_ids: list[str], mid_by_id: dict[str, dict[str, Any]]) -> str | None:
    candidates = [c for c in child_ids if _is_logo_like_cluster(mid_by_id[c], mid_by_id)]
    if not candidates:
        return None
    bools = [c for c in candidates if _norm_type(mid_by_id[c]) == "boolean operation"]
    pool = bools or candidates
    return sorted(pool, key=lambda c: -_bounds_area(mid_by_id[c].get("bounds")))[0]


def _assign_logo_interior(
    logo_id: str,
    mid_by_id: dict[str, dict[str, Any]],
    out: dict[str, str],
    _fix,
) -> None:
    row = mid_by_id[logo_id]
    ch_ids = [str(c) for c in (row.get("mid_child_ids") or []) if str(c) in mid_by_id]
    vecs = [c for c in ch_ids if _norm_type(mid_by_id[c]) == "vector"]

    if 1 <= len(vecs) <= 2 and len(vecs) == len(ch_ids):
        ordered = sorted(vecs, key=lambda c: -_bounds_area(mid_by_id[c].get("bounds")))
        _fix(ordered[0], "logo_back", "brand_logo_back")
        if len(ordered) > 1:
            _fix(ordered[1], "logo_fore", "brand_logo_fore")
        return

    if len(ch_ids) == 2 and len(vecs) == 1:
        vec_id = vecs[0]
        other = next(c for c in ch_ids if c != vec_id)
        _fix(vec_id, "logo_back", "brand_logo_back")
        if _norm_type(mid_by_id[other]) == "boolean operation":
            _fix(other, "logo_fore", "brand_logo_fore_wrapper")
        else:
            _fix(other, "logo_fore", "brand_logo_fore")
        for leaf_id in _leaf_vector_ids_under(logo_id, mid_by_id):
            if leaf_id == vec_id:
                _fix(leaf_id, "logo_back", "brand_logo_back_leaf")
            else:
                _fix(leaf_id, "logo_fore", "brand_logo_fore_leaf")
        return

    leaf_vecs = _leaf_vector_ids_under(logo_id, mid_by_id)
    if not leaf_vecs:
        return
    leaf_vecs.sort(key=lambda c: -_bounds_area(mid_by_id[c].get("bounds")))
    _fix(leaf_vecs[0], "logo_back", "brand_logo_back_leaf")
    if len(leaf_vecs) >= 2:
        _fix(leaf_vecs[1], "logo_fore", "brand_logo_fore_leaf")
    for cid in ch_ids:
        if cid in leaf_vecs:
            continue
        if _norm_type(mid_by_id[cid]) == "boolean operation":
            _fix(cid, "logo_fore", "brand_logo_fore_group")


def _is_brand_row_frame(
    child_ids: list[str],
    mid_by_id: dict[str, dict[str, Any]],
) -> bool:
    if len(child_ids) < 3:
        return False
    vectors = sum(1 for c in child_ids if _norm_type(mid_by_id[c]) == "vector")
    has_logo = any(_is_logo_like_cluster(mid_by_id[c], mid_by_id) for c in child_ids)
    return has_logo and vectors >= 2


def assign_brand_row_child_roles(
    mid_blocks: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    names: dict[str, str],
    fixes: list[dict[str, str]],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Name brand row: one ``logo`` cluster + ``logo_back``/``logo_fore`` inside; wordmarks by x vs logo."""
    out = dict(names)
    mid_by_id = {str(b["id"]): b for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}

    def _fix(sid: str, new: str, reason: str) -> None:
        old = out.get(sid, "")
        if old != new:
            out[sid] = new
            fixes.append({"id": sid, "old": old, "new": new, "reason": reason})

    for sid, row in mid_by_id.items():
        child_ids = [str(c) for c in (row.get("mid_child_ids") or []) if str(c) in mid_by_id]
        if len(child_ids) < 2:
            continue
        if out.get(sid) != "brand_group":
            if not _is_brand_row_frame(child_ids, mid_by_id):
                continue
            _fix(sid, "brand_group", "brand_row_structure_detected")

        logo_id = _pick_brand_row_logo_id(child_ids, mid_by_id)
        if logo_id:
            _fix(logo_id, "logo", "brand_row_logo_cluster")
            _assign_logo_interior(logo_id, mid_by_id, out, _fix)

        logo_interior: set[str] = set()
        if logo_id:
            logo_interior = _mid_descendant_ids(logo_id, mid_by_id)
            logo_interior.discard(logo_id)

        word_vectors = [
            cid
            for cid in sorted(child_ids, key=lambda c: _bounds_x(mid_by_id[c]))
            if cid != logo_id
            and cid not in logo_interior
            and _norm_type(mid_by_id[cid]) == "vector"
        ]

        if logo_id:
            logo_x = _bounds_x(mid_by_id[logo_id])
            left = [c for c in word_vectors if _bounds_x(mid_by_id[c]) < logo_x]
            right = [c for c in word_vectors if c not in left]
        else:
            left = word_vectors[:-1] if len(word_vectors) > 1 else list(word_vectors)
            right = word_vectors[-1:] if len(word_vectors) > 1 else []

        for i, cid in enumerate(left):
            _fix(cid, f"brand_name_first_part_{i + 1}", "brand_row_left_of_logo")
        if len(right) == 1:
            _fix(right[0], "brand_name_second", "brand_row_right_of_logo")
        elif len(right) > 1:
            for i, cid in enumerate(right[:-1]):
                _fix(cid, f"brand_name_first_part_{len(left) + i + 1}", "brand_row_right_of_logo_extra")
            _fix(right[-1], "brand_name_second", "brand_row_right_of_logo_last")

    return out, fixes


def number_star_decorations(
    mid_blocks: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    names: dict[str, str],
    fixes: list[dict[str, str]],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Assign ``star_decoration_1`` / ``star_decoration_2`` to Figma STAR nodes only (LTR, then TTB)."""
    out = dict(names)
    mid_by_id = {str(b["id"]): b for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}

    def _fix(sid: str, new: str, reason: str) -> None:
        old = out.get(sid, "")
        if old != new:
            out[sid] = new
            fixes.append({"id": sid, "old": old, "new": new, "reason": reason})

    for sid, row in mid_by_id.items():
        role = out.get(sid, "")
        if role.startswith("star_decoration") and _norm_type(row) != "star":
            _fix(sid, "unassigned", "star_role_on_non_star_node")

    star_ids = [sid for sid, row in mid_by_id.items() if _norm_type(row) == "star"]
    star_ids.sort(
        key=lambda s: (
            float(features.get(s, {}).get("center_x_ratio") or 0),
            float(features.get(s, {}).get("center_y_ratio") or 0),
        )
    )
    for i, sid in enumerate(star_ids):
        want = f"star_decoration_{min(i + 1, 2)}"
        _fix(sid, want, "number_star_decorations")
    return out, fixes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class SemanticValidationResult:
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    auto_fixes: list[dict[str, str]] = field(default_factory=list)


def validate_and_autofix_roles(
    mid_blocks: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    names: dict[str, str],
) -> tuple[dict[str, str], SemanticValidationResult]:
    result = SemanticValidationResult()
    out = dict(names)
    mid_by_id = {str(b["id"]): b for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}

    def _autofix(sid: str, new: str, reason: str) -> None:
        old = out.get(sid, "")
        if old != new:
            out[sid] = new
            result.auto_fixes.append({"id": sid, "old": old, "new": new, "reason": reason})

    heroes = [s for s, r in out.items() if r == "hero_image"]
    image_ids = [s for s, f in features.items() if f.get("imageHash")]
    if not heroes and image_ids:
        largest = max(image_ids, key=lambda s: float(features[s].get("area") or 0))
        _autofix(largest, "hero_image", "validator_missing_hero")
        result.warnings.append("validator:hero_image_missing_autofixed")

    for sid in heroes:
        feat = features.get(sid, {})
        if feat.get("is_tiny"):
            _autofix(sid, "star_decoration_1", "validator_hero_too_small")
            result.errors.append(f"hero_image_too_small:{sid}")

    for sid, role in out.items():
        feat = features.get(sid, {})
        if role == "background_shape" and feat.get("has_gradient_fill"):
            _autofix(sid, "background_gradient_1", "validator_bg_shape_has_gradient")
            result.warnings.append(f"validator:background_shape_had_gradient:{sid}")
        if role.startswith("background_gradient") and not feat.get("has_gradient_fill"):
            if feat.get("has_image_fill"):
                _autofix(sid, "hero_image", "validator_gradient_was_image")
            else:
                result.warnings.append(f"validator:gradient_role_without_gradient_fill:{sid}")
        if role == "age_badge" and not _is_age_badge_text(feat.get("characters") or ""):
            result.errors.append(f"age_badge_invalid_text:{sid}")
        if role == "legal_text":
            fs = float(feat.get("fontSize") or 0)
            if fs >= 72:
                result.warnings.append(f"validator:legal_text_large_font:{sid}")

    for forbidden in FORBIDDEN_OUTPUT_NAMES:
        for sid, role in list(out.items()):
            if role == forbidden or _canonicalize_role(role) != role:
                fixed = _canonicalize_role(role)
                if fixed != role:
                    _autofix(sid, fixed, f"validator_forbidden_{forbidden}")

    headlines = [s for s, r in out.items() if r == "headline"]
    large_text = [
        s
        for s, f in features.items()
        if _norm_type(mid_by_id.get(s, {})) == "text"
        and float(f.get("fontSize") or 0) >= 48
        and not _is_legal_text(f.get("characters") or "", f)
        and not _is_age_badge_text(f.get("characters") or "")
    ]
    if not headlines and large_text:
        best = max(large_text, key=lambda s: float(features[s].get("fontSize") or 0))
        _autofix(best, "headline", "validator_missing_headline")
        result.warnings.append("validator:headline_missing_autofixed")

    # decoration_group must not be final name
    for sid, role in out.items():
        if role in FORBIDDEN_OUTPUT_NAMES:
            result.errors.append(f"forbidden_role_in_output:{sid}:{role}")

    # Rich metadata: mid rows must still carry keys (checked at tree build)
    for sid, row in mid_by_id.items():
        if _norm_type(row) == "text":
            for key in ("fontSize", "fontName", "characters"):
                if key not in row:
                    result.warnings.append(f"validator:mid_missing_{key}:{sid}")

    for sid, row in mid_by_id.items():
        if not (row.get("mid_parent_ids") or []):
            if out.get(sid) != "banner_root":
                _autofix(sid, "banner_root", "validator_force_banner_root")
                result.errors.append(f"root_not_banner_root:{sid}:{out.get(sid)}")

    for sid, role in list(out.items()):
        feat = features.get(sid, {})
        chars = feat.get("characters") or ""
        if role == "headline" and _is_legal_text(chars, feat):
            _autofix(sid, "legal_text", "validator_legal_not_headline")
            result.errors.append(f"legal_text_mislabeled_headline:{sid}")
        if role.startswith("background_gradient") and not feat.get("has_gradient_fill"):
            if _norm_type(mid_by_id.get(sid, {})) == "text":
                continue
            _autofix(sid, "unassigned", "validator_gradient_without_fill")
            result.warnings.append(f"validator:solid_not_gradient:{sid}")

    for sid, role in list(out.items()):
        if role != "hero_image":
            continue
        parents = features.get(sid, {}).get("mid_parent_ids") or []
        if not parents:
            continue
        pid = str(parents[-1])
        if out.get(pid) == "unassigned":
            _autofix(pid, "hero_image", "validator_hero_wrapper_unassigned")
            _autofix(sid, "unassigned", "validator_hero_inner_leaf")
            result.warnings.append(f"validator:hero_inside_unassigned_autofixed:{pid}:{sid}")

    return out, result


# ---------------------------------------------------------------------------
# Hierarchy naming (groups only; tree structure from mid_child_ids)
# ---------------------------------------------------------------------------


def apply_canonical_group_names(names: dict[str, str], mid_blocks: list[dict[str, Any]]) -> dict[str, str]:
    """Ensure group containers use canonical group role names (never rename canvas root)."""
    out = dict(names)
    mid_by_id = {str(b["id"]): b for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}
    for sid, row in mid_by_id.items():
        if not (row.get("mid_parent_ids") or []):
            out[sid] = "banner_root"
            continue
        if _norm_type(row) not in ("frame", "group"):
            continue
        child_ids = [str(c) for c in (row.get("mid_child_ids") or [])]
        child_roles = [out.get(c, "") for c in child_ids]
        if sum(1 for r in child_roles if r.startswith("brand_name") or r in ("logo", "logo_back", "logo_fore")) >= 2:
            out[sid] = "brand_group"
        elif any(r in ("headline", "subheadline_delivery_time") for r in child_roles):
            # Only wrap when this frame is not the banner root (headline+legal can be direct children).
            direct_text = [c for c in child_ids if _norm_type(mid_by_id.get(c, {})) == "text"]
            if len(direct_text) <= 2 and out.get(sid) not in ("brand_group",):
                pass
            elif out.get(sid) in ("unassigned", "headline_group"):
                out[sid] = "headline_group"
        elif sum(1 for r in child_roles if r in ("price_text", "old_price_text", "discount_badge")) >= 2:
            out[sid] = "offer_group"
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@dataclass
class StrictSemanticResult:
    names: dict[str, str]
    ambiguous_ids: list[str]
    semantic_debug: dict[str, Any]
    validation: SemanticValidationResult


def run_strict_semantic_naming(
    mid_blocks: list[dict[str, Any]],
    qwen_names: dict[str, str] | None = None,
) -> StrictSemanticResult:
    """
    Deterministic strict naming pipeline. Optional ``qwen_names`` merges only ambiguous ids.
    """
    features = extract_node_features(mid_blocks)
    prelabel = prelabel_roles(mid_blocks, features)
    ambiguous = identify_ambiguous_nodes(mid_blocks, features, prelabel)

    merged = dict(prelabel)
    ambiguous_set = set(ambiguous)
    qwen_applied: dict[str, str] = {}
    if qwen_names:
        for sid, role in qwen_names.items():
            sk = str(sid)
            if sk not in ambiguous_set:
                continue
            if prelabel.get(sk) in PROTECTED_TEXT_ROLES:
                continue
            merged[sk] = _canonicalize_role(role)
            qwen_applied[sk] = merged[sk]

    normalized = normalize_roles(mid_blocks, features, merged)
    resolved, conflict_fixes = resolve_role_conflicts(mid_blocks, features, normalized)
    resolved = apply_canonical_group_names(resolved, mid_blocks)
    brand_star_fixes: list[dict[str, str]] = []
    resolved, brand_star_fixes = assign_brand_row_child_roles(
        mid_blocks, features, resolved, brand_star_fixes
    )
    resolved, brand_star_fixes = number_star_decorations(
        mid_blocks, features, resolved, brand_star_fixes
    )
    conflict_fixes.extend(brand_star_fixes)
    final_names, validation = validate_and_autofix_roles(mid_blocks, features, resolved)

    debug = {
        "brand_row_pass": "v2",
        "prelabel_roles": {k: prelabel[k] for k in sorted(prelabel)},
        "qwen_roles": qwen_applied,
        "normalized_roles": {k: normalized[k] for k in sorted(normalized)},
        "conflicts_fixed": conflict_fixes,
        "validator_warnings": validation.warnings,
        "validator_errors": validation.errors,
        "validator_autofixes": validation.auto_fixes,
        "ambiguous_ids": ambiguous,
        "ambiguous_count": len(ambiguous),
    }

    return StrictSemanticResult(
        names=final_names,
        ambiguous_ids=ambiguous,
        semantic_debug=debug,
        validation=validation,
    )


def build_semantic_json_from_strict_names(
    mid_blocks: list[dict[str, Any]],
    names: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    """Build nested rich semantic JSON tree; preserves all mid metadata."""
    tree = build_semantic_figma_tree_from_mid(mid_blocks, names, warnings)
    tree = lift_unassigned_wrappers_in_logo_subtrees(tree, warnings)
    allowed = collect_allowed_ids_from_mid(mid_blocks)
    tree = remove_non_mid_json_nodes(tree, allowed, warnings)
    validate_final_json_ids(tree, mid_blocks, warnings)
    return tree


def assert_rich_metadata_preserved(mid_blocks: list[dict[str, Any]], tree: dict[str, Any]) -> list[str]:
    """Verify output tree nodes still carry rich fields from mid rows."""
    warnings: list[str] = []
    mid_by_id = {str(b["id"]): b for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}

    def walk(node: dict[str, Any]) -> None:
        nid = node.get("id")
        if nid is None:
            return
        sid = str(nid)
        src = mid_by_id.get(sid)
        if not src:
            return
        for key in _RICH_METADATA_KEYS:
            if key in src and key not in node:
                warnings.append(f"rich_metadata_lost:{sid}:{key}")
        for ch in node.get("children") or []:
            if isinstance(ch, dict):
                walk(ch)

    walk(tree)
    return warnings
