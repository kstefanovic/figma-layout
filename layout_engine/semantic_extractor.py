MAIN_ROLES = {
    "banner_root",
    "background",
    "background_shape",
    "image_zone",
    "hero_image",
    "brand_group",
    "logo",
    "brand_name",
    "brand_name_first",
    "brand_name_second",
    "headline_group",
    "headline",
    "subheadline",
    "subheadline_delivery_time",
    "legal_text",
    "age_badge",
    "decoration_group",
    "decoration",
}


def normalize_name(name: str) -> str:
    name = (name or "").strip().lower()

    # remove common Figma random suffix style
    # example: "headline_group 123456" or "headline_group_Group"
    for role in MAIN_ROLES:
        if name == role:
            return role
        if name.startswith(role + "_"):
            return role
        if name.startswith(role + " "):
            return role
        if name.startswith(role + "-"):
            return role

    if "image_zone" in name:
        return "image_zone"
    if "hero" in name and "image" in name:
        return "hero_image"
    if "hero_image" in name:
        return "hero_image"

    if "headline_group" in name:
        return "headline_group"
    if "headline" in name and "group" in name:
        return "headline_group"
    if "subheadline_delivery" in name:
        return "subheadline_delivery_time"
    if "subheadline" in name:
        return "subheadline"
    if "headline" in name:
        return "headline"

    if "legal" in name:
        return "legal_text"

    if "age_badge" in name:
        return "age_badge"
    if name == "0+" or "0+" in name:
        return "age_badge"

    if "brand_group" in name:
        return "brand_group"
    if "brand_name_first" in name:
        return "brand_name_first"
    if "brand_name_second" in name:
        return "brand_name_second"
    if "brand_name" in name:
        return "brand_name"
    if "brand" in name:
        return "brand_group"

    if "logo" in name:
        return "logo"

    if "background_shape" in name:
        return "background_shape"
    if "background" in name:
        return "background"

    if "decoration_group" in name:
        return "decoration_group"
    if "decoration" in name:
        return "decoration"

    return ""


def collect_semantic_nodes(node: dict, result=None):
    if result is None:
        result = {}

    role = normalize_name(node.get("name", ""))

    if role:
        result.setdefault(role, []).append(node)

    for child in node.get("children", []) or []:
        collect_semantic_nodes(child, result)

    return result


def get_primary_node(nodes_by_role: dict, *roles):
    """
    Return first available node from role priority list.
    """
    for role in roles:
        arr = nodes_by_role.get(role, [])
        if arr:
            return arr[0]
    return None


def get_area(node: dict) -> float:
    b = node.get("bounds", {}) or {}
    return float(b.get("width", 0)) * float(b.get("height", 0))


def get_largest_node(nodes_by_role: dict, *roles):
    candidates = []
    for role in roles:
        candidates.extend(nodes_by_role.get(role, []))

    if not candidates:
        return None

    return max(candidates, key=get_area)
