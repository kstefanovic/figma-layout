"""Semantic roles used by the layout transformer pipeline."""

TRAIN_ROLES = [
    "hero_image",
    "background_shape",
    "brand_group",
    "headline_group",
    "legal_text",
]

FLOATING_ROLES = [
    "age_badge",
    "star_decoration_1",
    "star_decoration_2",
    "background_gradient_1",
    "background_gradient_2",
]

CHILD_ROLES = [
    "headline",
    "subheadline_delivery_time",
    "logo",
    "logo_back",
    "logo_fore",
    "brand_name_first_part_1",
    "brand_name_first_part_2",
    "brand_name_second",
]

ROLES = TRAIN_ROLES + FLOATING_ROLES + CHILD_ROLES

ROLE_TO_ID = {role: idx for idx, role in enumerate(TRAIN_ROLES)}
ID_TO_ROLE = {idx: role for role, idx in ROLE_TO_ID.items()}
NUM_ROLES = len(TRAIN_ROLES)

FLOATING_ROLE_SET = set(FLOATING_ROLES)
CHILD_ROLE_SET = set(CHILD_ROLES)
TRAIN_ROLE_SET = set(TRAIN_ROLES)
