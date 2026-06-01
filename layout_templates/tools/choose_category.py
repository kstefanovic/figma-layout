#!/usr/bin/env python3
"""Choose a template category from a semantic-rich JSON payload."""

from __future__ import annotations

import json
import sys
from typing import Any


def _normalize_root_node(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                return item
    return {}


def _extract_top_child_names(payload: Any) -> set[str]:
    root = _normalize_root_node(payload)
    children = root.get("children", [])
    names: set[str] = set()

    if not isinstance(children, list):
        return names

    for child in children:
        if not isinstance(child, dict):
            continue
        name = child.get("name") or child.get("semantic_name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())

    return names


def choose_category(payload: Any) -> str | None:
    """Return category based on top-level child semantic groups."""
    top_children = _extract_top_child_names(payload)

    if "word_vector_group" in top_children:
        return "mnogo_tmp"
    if "hero_group" not in top_children:
        return "no_img_temp"
    if "headline_group" in top_children and "offer_group" not in top_children:
        return "headline_group_temp"
    if "offer_group" in top_children and "headline_group" not in top_children:
        return "offer_group_temp"
    return None


def main() -> int:
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.load(sys.stdin)

    result = choose_category(payload)
    print(result if result is not None else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
