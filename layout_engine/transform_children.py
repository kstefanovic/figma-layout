import copy


def get_bounds(node: dict) -> dict:
    return node.get("bounds", {}) or {}


def set_bounds(node: dict, x: float, y: float, w: float, h: float):
    node.setdefault("bounds", {})
    node["bounds"]["x"] = round(float(x), 2)
    node["bounds"]["y"] = round(float(y), 2)
    node["bounds"]["width"] = round(float(w), 2)
    node["bounds"]["height"] = round(float(h), 2)


def scale_text_style(node: dict, scale: float):
    if scale <= 0:
        return
    if "fontSize" in node:
        try:
            node["fontSize"] = round(max(1.0, float(node["fontSize"]) * scale), 2)
        except (TypeError, ValueError):
            pass
    style = node.get("style")
    if isinstance(style, dict) and "fontSize" in style:
        try:
            style["fontSize"] = round(max(1.0, float(style["fontSize"]) * scale), 2)
        except (TypeError, ValueError):
            pass


def transform_node_tree(node: dict, old_box: dict, new_box: dict):
    old_x = float(old_box.get("x", 0))
    old_y = float(old_box.get("y", 0))
    old_w = max(float(old_box.get("width", 1)), 1)
    old_h = max(float(old_box.get("height", 1)), 1)

    new_x = float(new_box.get("x", 0))
    new_y = float(new_box.get("y", 0))
    new_w = max(float(new_box.get("width", 1)), 1)
    new_h = max(float(new_box.get("height", 1)), 1)
    font_scale = min(new_w / old_w, new_h / old_h)

    b = get_bounds(node)

    rel_x = (float(b.get("x", 0)) - old_x) / old_w
    rel_y = (float(b.get("y", 0)) - old_y) / old_h
    rel_w = float(b.get("width", 0)) / old_w
    rel_h = float(b.get("height", 0)) / old_h

    set_bounds(
        node,
        new_x + rel_x * new_w,
        new_y + rel_y * new_h,
        rel_w * new_w,
        rel_h * new_h,
    )
    if (node.get("type") or "").lower() == "text" or "fontSize" in node:
        scale_text_style(node, font_scale)

    for child in node.get("children", []) or []:
        transform_node_tree(child, old_box, new_box)


def move_and_scale_group(node: dict, new_box: dict):
    old_box = copy.deepcopy(get_bounds(node))
    transform_node_tree(node, old_box, new_box)


def set_bounds_and_scale_text(node: dict, x: float, y: float, w: float, h: float):
    old_box = copy.deepcopy(get_bounds(node))
    old_w = max(float(old_box.get("width", 1) or 1), 1)
    old_h = max(float(old_box.get("height", 1) or 1), 1)
    scale = min(max(float(w), 1) / old_w, max(float(h), 1) / old_h)
    set_bounds(node, x, y, w, h)
    if (node.get("type") or "").lower() == "text" or "fontSize" in node:
        scale_text_style(node, scale)
