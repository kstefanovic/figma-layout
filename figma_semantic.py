"""Flatten Figma export raw JSON to leaf-only mid JSON; helpers for Qwen semantic naming."""

from __future__ import annotations

import json
import re
from typing import Any


def _is_leaf(node: dict[str, Any]) -> bool:
    ch = node.get("children")
    return not ch


def _collect_leaves(node: dict[str, Any], out: list[dict[str, Any]]) -> None:
    if _is_leaf(node):
        leaf = {k: v for k, v in node.items() if k != "children"}
        out.append(leaf)
        return
    for child in node.get("children") or []:
        if isinstance(child, dict):
            _collect_leaves(child, out)


def flatten_raw_to_mid(raw: Any, frame_index: int = 0) -> list[dict[str, Any]]:
    """
    Collect deepest (leaf) nodes from Figma-like trees. Omits wrapper nodes that only exist
    to hold `children`. Each leaf keeps all of its own properties except `children`.
    """
    if isinstance(raw, list):
        if not raw:
            return []
        if frame_index < 0 or frame_index >= len(raw):
            raise ValueError(f"frame_index {frame_index} is out of range (0..{len(raw) - 1})")
        root = raw[frame_index]
        if not isinstance(root, dict):
            raise ValueError("Selected frame is not a JSON object")
        roots: list[dict[str, Any]] = [root]
    elif isinstance(raw, dict):
        roots = [raw]
    else:
        raise ValueError("Raw JSON must be a list (frames) or a single root object")

    out: list[dict[str, Any]] = []
    for r in roots:
        _collect_leaves(r, out)
    return out


def mid_node_prompt_slice(node: dict[str, Any]) -> dict[str, Any]:
    """Minimal fields sent to the vision model (bbox guides crop mentally)."""
    item: dict[str, Any] = {
        "id": node.get("id"),
        "type": node.get("type"),
        "bounds": node.get("bounds"),
        "old_name": node.get("name"),
    }
    ch = node.get("characters")
    if isinstance(ch, str) and ch.strip():
        item["text_preview"] = ch.strip()[:240]
    return item


def chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_naming_user_prompt(
    nodes_minimal: list[dict[str, Any]],
    frame_hint: str,
    *,
    has_reference_grid: bool = False,
) -> str:
    nodes_json = json.dumps(nodes_minimal, ensure_ascii=False, separators=(",", ":"))
    vision_intro = (
        "You are given TWO images before this text, then a JSON array `nodes` of leaf layout elements.\n"
        "(1) First image: the full Figma banner export (same coordinate world as `bounds` in the JSON).\n"
        "(2) Second image: a reference **grid** image. Each cell shows one element (thumbnail/crop or render) "
        "together with its **id** label in that cell. Use the grid to tie each id to what that element actually "
        "looks like, then assign a precise semantic name. Prefer the grid when it disagrees with a quick guess "
        "from the full banner alone.\n\n"
        if has_reference_grid
        else (
            "You are given a Figma-export banner image and a JSON array `nodes` of leaf layout elements.\n"
            "Use the image together with bounds (and text_preview when present) to infer what each leaf represents.\n\n"
        )
    )
    node_fields = (
        "Each node has:\n"
        "- id: stable identifier (never change it in your mapping)\n"
        "- type: Figma node type (text, rectangle, vector, group, …)\n"
        "- bounds: {x, y, width, height} in Figma export coordinates (origin top-left)\n"
        "- old_name: placeholder / anonymous name from export\n"
        "- text_preview: optional snippet for TEXT nodes\n\n"
        f"Frame context: {frame_hint}\n\n"
    )
    naming_rules = (
        "Assign one concise semantic identifier per id: English, lowercase_snake_case, 2–6 words as one "
        "identifier (e.g. headline_product_title, legal_footer_disclaimer, age_rating_badge_zero_plus).\n\n"
        "Output ONLY valid JSON (no markdown, no code fences, no commentary) with exactly this shape:\n"
        '{"names":{"<id>":"<semantic_name>",...}}\n\n'
        "Rules:\n"
        "- Include every id from the input exactly once as a key in names.\n"
        "- Do not invent ids.\n"
        "- Values must be non-empty strings.\n\n"
        f"nodes={nodes_json}"
    )
    return vision_intro + node_fields + naming_rules


def extract_first_json_object(text: str) -> Any:
    """Parse first top-level JSON object from model text (strips optional ``` fences)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("Unbalanced braces in model JSON output")


def parse_names_object(text: str) -> dict[str, str]:
    obj = extract_first_json_object(text)
    if not isinstance(obj, dict):
        raise ValueError("Model output root must be a JSON object")
    names = obj.get("names")
    if not isinstance(names, dict):
        raise ValueError('Model output must contain object key "names" mapping id -> string')
    out: dict[str, str] = {}
    for k, v in names.items():
        if isinstance(v, str) and v.strip():
            out[str(k)] = v.strip()
    return out


def apply_semantic_names(mid: list[dict[str, Any]], names: dict[str, str]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for node in mid:
        row = dict(node)
        nid = row.get("id")
        if nid is not None and str(nid) in names:
            row["name"] = names[str(nid)]
        merged.append(row)
    return merged


def missing_name_ids(mid: list[dict[str, Any]], names: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for node in mid:
        nid = node.get("id")
        if nid is None:
            continue
        sid = str(nid)
        if sid not in names:
            missing.append(sid)
    return missing
