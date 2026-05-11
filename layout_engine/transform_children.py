import copy


def get_bounds(node: dict) -> dict:
    return node.get("bounds", {}) or {}


def set_bounds(node: dict, x: float, y: float, w: float, h: float):
    node.setdefault("bounds", {})
    node["bounds"]["x"] = round(float(x), 2)
    node["bounds"]["y"] = round(float(y), 2)
    node["bounds"]["width"] = round(float(w), 2)
    node["bounds"]["height"] = round(float(h), 2)


def transform_node_tree(node: dict, old_box: dict, new_box: dict):
    old_x = float(old_box.get("x", 0))
    old_y = float(old_box.get("y", 0))
    old_w = max(float(old_box.get("width", 1)), 1)
    old_h = max(float(old_box.get("height", 1)), 1)

    new_x = float(new_box.get("x", 0))
    new_y = float(new_box.get("y", 0))
    new_w = max(float(new_box.get("width", 1)), 1)
    new_h = max(float(new_box.get("height", 1)), 1)

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

    for child in node.get("children", []) or []:
        transform_node_tree(child, old_box, new_box)


def move_and_scale_group(node: dict, new_box: dict):
    old_box = copy.deepcopy(get_bounds(node))
    transform_node_tree(node, old_box, new_box)
