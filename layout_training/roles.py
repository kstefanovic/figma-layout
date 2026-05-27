"""Canonical top-level role mapping for V1 layout training.

Raw semantic JSONs keep their original root.children names. Training records map
those names to a smaller, stable role vocabulary so one transformer can learn
families with optional or inconsistent top-level groups.
"""

from __future__ import annotations

import re
from typing import Any


TRAIN_ROLE_MAP: dict[str, str] = {
    "headline_group": "text_main_group",
    "offer_group": "text_main_group",
    "text_group": "text_main_group",
    "price_group": "text_main_group",
    "discount_badge_group": "badge_group",
    "badge_group": "badge_group",
    "age_badge_group": "badge_group",
    "hero_group": "hero_group",
    "product_group": "hero_group",
    "brand_group": "brand_group",
    "legal_group": "legal_group",
    "decoration_group": "decoration_group",
    "star_decoration": "decoration_group",
    "star_decoration_1": "decoration_group",
    "star_decoration_2": "decoration_group",
    "star_decoration_3": "decoration_group",
    "star_decoration_4": "decoration_group",
    "star_decoration_5": "decoration_group",
    "background_gradient": "background_gradient",
    "background_gradient_1": "background_gradient_1",
    "background_gradient_2": "background_gradient_2",
    "background_gradient_3": "background_gradient_3",
    "background_gradient_4": "background_gradient_4",
    "background_gradient_5": "background_gradient_5",
    "background_shape": "background_shape_cluster",
    "background_shape_1": "background_shape_cluster",
    "background_shape_2": "background_shape_cluster",
    "background_shape_3": "background_shape_cluster",
    "background_group": "background_shape_cluster",
    "foreground_group": "foreground_group",
    "unknown_group": "unknown_group",
}

COMMON_RAW_ROLE_GROUPS = {
    "hero_group": {"hero_group", "product_group"},
    "brand_group": {"brand_group"},
    "text_main_group": {"headline_group", "offer_group", "text_group", "price_group"},
    "legal_group": {"legal_group"},
    "background_shape_cluster": {"background_shape", "background_shape_1", "background_shape_2", "background_shape_3", "background_group"},
}

MERGED_TRAIN_ROLES = {"background_shape_cluster"}
ROLE_OCCURRENCE_BY_AREA = {"text_main_group"}
ROLE_OCCURRENCE_BY_POSITION = {"badge_group"}


def normalize_role(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown_group"


def train_role_for(raw_role: Any) -> str:
    role = normalize_role(raw_role)
    if re.fullmatch(r"background_shape_\d+", role):
        return "background_shape_cluster"
    if re.fullmatch(r"background_gradient_\d+", role):
        return role
    return TRAIN_ROLE_MAP.get(role, "unknown_group")


def is_known_raw_role(raw_role: Any) -> bool:
    role = normalize_role(raw_role)
    return role in TRAIN_ROLE_MAP or bool(re.fullmatch(r"background_shape_\d+|background_gradient_\d+", role))


def token_id_for(train_role: str, occurrence_index: int) -> str:
    return f"{train_role}#{occurrence_index}"

