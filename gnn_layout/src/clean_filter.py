"""Strict filters for extracting clean semantic banner JSONs from mixed exports."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

MAIN_ROLES = {
    "hero_image",
    "brand_group",
    "headline_group",
    "legal_text",
    "age_badge",
}

LEAKED_ROOT_ROLES = {
    "logo",
    "brand_name",
    "brand_name_first",
    "brand_name_second",
    "headline",
    "subheadline",
    "subheadline_delivery_time",
}

IGNORED_PREFIXES = (
    "background_shape",
    "background_gradient",
    "star_decoration",
    "decoration",
    "gradient",
)


def normalize_role_name(name: str) -> str:
    """Map a layer/text name into the semantic role vocabulary used by cleaning."""
    raw = str(name or "").strip().lower()
    raw = re.sub(r"\s+", "_", raw).replace("-", "_")
    if not raw:
        return ""
    if "unassigned" in raw:
        return "unassigned"
    if raw == "0+" or "0+" in raw:
        return "age_badge"
    if raw == "image_zone" or raw.startswith("image_zone_"):
        return "hero_image"
    if raw == "hero_image" or raw.startswith("hero_image_"):
        return "hero_image"
    if raw == "brand_group" or raw.startswith("brand_group_"):
        return "brand_group"
    if raw == "headline_group" or raw.startswith("headline_group_"):
        return "headline_group"
    if raw == "legal_text" or raw.startswith("legal_text_") or raw == "legal":
        return "legal_text"
    if raw == "age_badge" or raw.startswith("age_badge_"):
        return "age_badge"
    if raw in {"brand_name", "brand_name_first", "brand_name_second"}:
        return raw
    if raw.startswith("brand_name_first_"):
        return "brand_name_first"
    if raw.startswith("brand_name_second_"):
        return "brand_name_second"
    if raw.startswith("brand_name_"):
        return "brand_name"
    if raw == "logo" or raw.startswith("logo_"):
        return "logo"
    if raw == "headline" or raw.startswith("headline_"):
        return "headline"
    if raw == "subheadline_delivery_time" or raw.startswith("subheadline_delivery_time_"):
        return "subheadline_delivery_time"
    if raw == "subheadline" or raw.startswith("subheadline_"):
        return "subheadline"
    if raw == "background" or any(raw.startswith(prefix) for prefix in IGNORED_PREFIXES):
        return ""
    return ""


def flatten_nodes(node: dict, depth: int = 0, parent_role: str | None = None) -> list[dict[str, Any]]:
    """Flatten a Figma tree and annotate depth, path, normalized role, and nearest parent role."""
    out: list[dict[str, Any]] = []

    def walk(item: Any, item_depth: int, inherited_role: str | None, path: str) -> None:
        if not isinstance(item, dict):
            return
        role = normalize_role_name(str(item.get("name") or item.get("characters") or ""))
        out.append(
            {
                "node": item,
                "depth": item_depth,
                "role": role,
                "parent_role": inherited_role,
                "path": path,
            }
        )
        next_parent_role = role or inherited_role
        for idx, child in enumerate(item.get("children") or []):
            child_path = f"{path}/{idx}" if path else str(idx)
            walk(child, item_depth + 1, next_parent_role, child_path)

    walk(node, depth, parent_role, "")
    return out


def count_roles(flat_nodes: list[dict[str, Any]]) -> dict[str, int]:
    """Count distinct semantic role containers, ignoring child text that repeats parent role."""
    counts: Counter[str] = Counter()
    for item in flat_nodes:
        role = item.get("role") or ""
        if not role:
            continue
        if role in MAIN_ROLES and item.get("parent_role") == role:
            continue
        counts[role] += 1
    return dict(counts)


def has_valid_bounds(node: dict) -> bool:
    bounds = node.get("bounds") if isinstance(node, dict) else None
    if not isinstance(bounds, dict):
        return False
    width = _finite_float(bounds.get("width"))
    height = _finite_float(bounds.get("height"))
    return width > 0 and height > 0


def has_text_descendant(node: dict) -> bool:
    return bool(get_all_text(node).strip())


def get_all_text(node: dict) -> str:
    pieces: list[str] = []

    def walk(item: Any) -> None:
        if not isinstance(item, dict):
            return
        for key in ("characters", "text", "content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value.strip())
                break
        for child in item.get("children") or []:
            walk(child)

    walk(node)
    return " ".join(pieces)


def is_clean_banner(banner: dict, strict: bool = True) -> tuple[bool, list[str]]:
    """Return whether a banner is clean enough for supervised GNN layout training."""
    reasons: list[str] = []
    if not isinstance(banner, dict):
        return False, ["banner_not_object"]

    if str(banner.get("type") or "").lower() != "frame":
        reasons.append("root_not_frame")
    if not has_valid_bounds(banner):
        reasons.append("root_invalid_bounds")

    flat = flatten_nodes(banner)
    counts = count_roles(flat)
    root_name = str(banner.get("name") or "").lower()
    has_semantic_child = any(
        item["depth"] == 1 and item["role"] in (MAIN_ROLES | LEAKED_ROOT_ROLES)
        for item in flat
    )
    if "banner_root" not in root_name and not has_semantic_child:
        reasons.append("root_missing_banner_root_or_semantic_children")

    for role in MAIN_ROLES:
        count = counts.get(role, 0)
        if count == 0:
            reasons.append(f"missing_role:{role}")
        elif count > 1:
            reasons.append(f"duplicate_role:{role}:{count}")

    if counts.get("unassigned", 0) > 0:
        reasons.append("contains_unassigned")

    for item in flat:
        role = item["role"]
        if item["depth"] == 1 and role in LEAKED_ROOT_ROLES:
            reasons.append(f"root_level_leaked_role:{role}:{item['path']}")

    role_nodes = _representative_role_nodes(flat)
    for role in MAIN_ROLES:
        node = role_nodes.get(role)
        if node is not None and not has_valid_bounds(node):
            reasons.append(f"invalid_role_bounds:{role}")

    legal_node = role_nodes.get("legal_text")
    legal_text = get_all_text(legal_node) if legal_node else ""
    if len(legal_text.strip()) <= 10:
        reasons.append("legal_text_too_short")

    age_node = role_nodes.get("age_badge")
    age_text = get_all_text(age_node) if age_node else ""
    if age_node is not None and not _looks_like_age_badge(age_text, str(age_node.get("name") or "")):
        reasons.append("age_badge_text_invalid")

    headline_node = role_nodes.get("headline_group")
    if headline_node is not None and not has_text_descendant(headline_node):
        reasons.append("headline_group_missing_text")

    if not strict:
        reasons = [
            reason
            for reason in reasons
            if reason
            not in {
                "age_badge_text_invalid",
                "headline_group_missing_text",
                "legal_text_too_short",
            }
        ]
    return len(reasons) == 0, reasons


def _representative_role_nodes(flat_nodes: list[dict[str, Any]]) -> dict[str, dict]:
    reps: dict[str, dict] = {}
    for item in flat_nodes:
        role = item.get("role") or ""
        if role not in MAIN_ROLES:
            continue
        if item.get("parent_role") == role:
            continue
        reps.setdefault(role, item["node"])
    return reps


def _looks_like_age_badge(text: str, name: str) -> bool:
    value = f"{text} {name}".strip().lower()
    if "0+" in value:
        return True
    compact = re.sub(r"\s+", "", value)
    return bool(re.fullmatch(r"\d{1,2}\+?", compact))


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else 0.0
    except (TypeError, ValueError):
        return 0.0
