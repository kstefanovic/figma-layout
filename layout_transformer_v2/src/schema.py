"""Shared role and class vocabularies for the multi-model rich layout system."""

from __future__ import annotations

PARENT_ROLES = [
    "hero_image",
    "background_shape",
    "brand_group",
    "headline_group",
    "legal_text",
    "legal_text_group",
]

CHILD_ROLES = [
    "headline",
    "subheadline_delivery_time",
    "brand_name_first_part_1",
    "brand_name_first_part_2",
    "brand_name_second",
    "logo",
    "logo_back",
    "logo_fore",
]

FLOATING_ROLES = [
    "age_badge",
    "star_decoration_1",
    "star_decoration_2",
    "background_gradient_1",
    "background_gradient_2",
]

ALL_ROLES = PARENT_ROLES + CHILD_ROLES + FLOATING_ROLES

CHILD_PARENT = {
    "headline": "headline_group",
    "subheadline_delivery_time": "headline_group",
    "brand_name_first_part_1": "brand_group",
    "brand_name_first_part_2": "brand_group",
    "brand_name_second": "brand_group",
    "logo": "brand_group",
    "logo_back": "logo",
    "logo_fore": "logo",
}

ALIGN_H = ["LEFT", "CENTER", "RIGHT", "JUSTIFIED", "UNKNOWN"]
ALIGN_V = ["TOP", "CENTER", "BOTTOM", "UNKNOWN"]
ALIGN_H_TO_ID = {value: idx for idx, value in enumerate(ALIGN_H)}
ALIGN_V_TO_ID = {value: idx for idx, value in enumerate(ALIGN_V)}
ID_TO_ALIGN_H = {idx: value for value, idx in ALIGN_H_TO_ID.items()}
ID_TO_ALIGN_V = {idx: value for value, idx in ALIGN_V_TO_ID.items()}

ORIENTATIONS = ["portrait", "square", "landscape"]
ORIENTATION_TO_ID = {value: idx for idx, value in enumerate(ORIENTATIONS)}

UNKNOWN_NODE_TYPE = "<unknown>"
DEFAULT_NODE_TYPES = [
    UNKNOWN_NODE_TYPE,
    "boolean operation",
    "ellipse",
    "frame",
    "group",
    "line",
    "rectangle",
    "star",
    "text",
    "vector",
]

GROUP_TYPES = {"frame", "group"}
TEXT_TYPES = {"text"}

ROLE_WEIGHTS = {
    "hero_image": 1.5,
    "background_shape": 1.25,
    "brand_group": 2.0,
    "headline_group": 2.0,
    "legal_text": 1.25,
    "legal_text_group": 1.25,
    "headline": 2.0,
    "subheadline_delivery_time": 1.75,
    "brand_name_first_part_1": 1.5,
    "brand_name_first_part_2": 1.5,
    "brand_name_second": 1.5,
    "logo": 1.75,
    "logo_back": 1.25,
    "logo_fore": 1.25,
    "age_badge": 1.25,
    "star_decoration_1": 0.8,
    "star_decoration_2": 0.8,
    "background_gradient_1": 0.8,
    "background_gradient_2": 0.8,
}


def orientation_id(width: float, height: float) -> int:
    """Return a coarse canvas orientation id."""
    if height > width * 1.05:
        return ORIENTATION_TO_ID["portrait"]
    if width > height * 1.05:
        return ORIENTATION_TO_ID["landscape"]
    return ORIENTATION_TO_ID["square"]

