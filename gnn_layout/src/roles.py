"""Semantic roles predicted by the GNN."""

ROLES = [
    "hero_image",
    "brand_group",
    "headline_group",
    "legal_text",
    "age_badge",
]

ROLE_TO_IDX = {role: idx for idx, role in enumerate(ROLES)}
IDX_TO_ROLE = {idx: role for role, idx in ROLE_TO_IDX.items()}
NUM_ROLES = len(ROLES)

OTHER_ROLE = "other"
NODE_ROLE_FEATURES = ROLES + [OTHER_ROLE]
