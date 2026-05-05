"""Flatten Figma export raw JSON to leaf-only mid JSON; helpers for Qwen semantic naming."""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any

# Full raw JSON → semantic Figma JSON (single Qwen call: banner + grid + raw text).
FIGMA_CONVERT_SYSTEM_PROMPT = textwrap.dedent(
    """
    You output exactly one JSON object: the same Figma tree as the user’s raw JSON, but every node must get a
    meaningful English snake_case semantic `name` from the vocabulary the user lists (or a short phrase built only
    from those vocabulary roots).

    Hard requirements:
    - Copy `id`, `type`, `bounds`, `characters`, `visible`, `opacity`, `path`, and geometry unchanged unless you
      intentionally collapse a meaningless wrapper (then keep children and geometry consistent with the rules).
    - Never use lazy Figma defaults as final semantic names: bare digits ("5", "10"), "Frame", "Rectangle", "Vector",
      "Ellipse", "Line", "Text", names like "Group 3912003", pixel-size roots like "960x512", or the raw `id` string
      as `name`.
    - Never emit ``//`` or ``/* */`` comments inside JSON; output must be strictly parseable (no "omitted for brevity"
      lines).
    - Emit **compact** JSON (no indentation / no extra newlines between keys)—deep trees must still be one fully
      closed object ending with ``}``; never stop mid-key or mid-string.
    - In ``characters`` strings use only normal spaces and ``\\n`` escapes—never Unicode line/paragraph separators
      (U+2028 / U+2029) copied from Figma.
    - The root frame must be named `banner_root` unless the file is clearly not a full banner (still prefer
      `banner_root` for single exported frames).
    - If a node is only decorative clutter, prefer `decoration_group` / `sparkle` / `ornament` etc. from the vocabulary
      rather than generic "Group…".
    - When you look at the **full banner** image, infer roles from **typography and layering** (size, weight, color,
      position), not only from Figma `type` strings: e.g. the largest boldest text block is rarely legal copy; the
      smallest thinnest text block is rarely the main headline.
    """
).strip()

FIGMA_CONVERT_PROMPT = textwrap.dedent(
    """
    You are a strict Figma semantic JSON converter.

    Inputs:
    1. Full banner image.
    2. Grid image: each cell has a **grey header bar** with text `id:<figma_id>` and, below it, a **thumbnail** of that
       element. The `<figma_id>` matches the `"id"` field in the raw JSON for the same element—use it to align grid
       crops with JSON nodes.
    3. Raw Figma JSON with anonymous names.

    Goal:
    Convert the raw Figma JSON into clean semantic JSON.

    How to read the banner (image 1) — typical retail / promo layout:
    - **Hero / product visual:** Often a large rectangle (photo or flat fill) for the product or scene; map to
      `hero_group`, `product_visual_group`, `main_product`, or `product_packshot` by what is shown.
    - **Background:** Often a colored rectangle or **vector** shapes behind content (e.g. blue panel); use
      `background_group` / `base_background` when it is clearly a plate or wash behind other layers—not the hero
      product itself.
    - **Brand group:** Cluster of **brand name** (text/vector; sometimes **two separate parts**—e.g. two wordmarks)
      plus **logo / mark**. The logo may sit on stacked layers: a **rear** backing shape and a **foreground** mark;
      keep them under `brand_group` with `brand_mark`, `brand_name_yandex`, or `brand_name_lavka` as appropriate, not
      as unrelated generic vectors.
    - **Headline group:** Usually **two text roles** near each other:
      - **Main headline:** The **largest and/or boldest** promotional line in the whole banner—what the product is,
        the hero offer, or the **current price** emphasis. Prefer `headline_text` / `headline_line` / `offer_group` /
        `current_price` by content.
      - **Sub-headline:** **Smaller** type, often **medium weight** (not hairline), placed **closest** to the main
        headline—delivery window, weight, pack size, secondary promise. Map to `subheadline_text`,
        `delivery_info_group`, `delivery_time_text` when it matches that role (not the legal footer).
    - **Legal / compliance copy:** The **smallest and thinnest** text in the layout—addresses, SKU, disclaimers,
      “conditions apply”, etc. Almost always `legal_group` / `legal_text` even if it sits above the bottom edge.
    - **Age badge:** Small badge with “0+”, “6+”, “12+”, etc. → `age_badge_group` / `age_badge_text`.
    - **Decorations:** Non-message flourishes—stars, glows, bulbs, sparkles, confetti, shine—→ `decoration_group`,
      `sparkle`, `star_decoration`, `glow_effect`, `shine_effect`, `ornament`, `overlay_effect_group` as fits.

    Use this mental model together with **bounds** and **grid thumbnails** so each raw `id` gets the right semantic
    role even when Figma node types are generic (`RECTANGLE`, `VECTOR`, `TEXT`).

    Important rules:
    - Output ONLY valid JSON—one complete root object that ends with ``}`` (not cut off by length).
    - Use **compact** JSON: minimal whitespace between tokens (no pretty-printed trees); this is required so the
      whole file fits in the model output limit.
    - Never put ``//`` or ``/* */`` comments inside the JSON; every line must be valid JSON (models often break
      parsers with ``// ... omitted ...`` — do not do that).
    - In ``characters`` fields use only printable text, normal spaces, and ``\\n`` for newlines—do not paste invisible
      Unicode line separators (U+2028 / U+2029) from Figma.
    - Do not explain.
    - Do not create mid_json.
    - Do not invent new elements.
    - Do not change id, type, bounds, characters, visible, opacity, or geometry.
    - Use the full banner image for global context.
    - Use the grid image to understand each element by its raw JSON id.
    - Use raw JSON ids exactly as provided.
    - **Mandatory:** replace **every** `"name"` in the output with a snake_case semantic identifier from the
      vocabulary below (or a 2–4 word snake_case phrase composed only from those vocabulary roots, e.g.
      `product_visual_group_left`). Every node including leaves and wrappers must comply—no numeric-only names, no
      `Group …` leftovers, no dimension strings as names.
    - Remove/collapse unnecessary wrapper groups/frames only when they have no independent visual meaning.
    - When removing a wrapper, preserve all meaningful children and promote them to the correct semantic parent.
    - Keep useful semantic containers such as brand_group, headline_group, hero_group, product_visual_group, legal_group, age_badge_group, decoration_group, background_group.
    - Build a correct nested hierarchy by semantic meaning, not by the original raw hierarchy.
    - Exact bbox already exists in raw JSON. Do not estimate bbox.

    Semantic naming vocabulary:
    banner_root
    background_group
    base_background
    brand_group
    brand_name_yandex
    brand_mark
    brand_name_lavka
    headline_group
    headline_text
    headline_line
    delivery_info_group
    delivery_time_text
    subheadline_text
    legal_group
    legal_text
    age_badge_group
    age_badge_text
    hero_group
    product_visual_group
    main_product
    product_packshot
    product_label
    offer_group
    price_group
    current_price
    old_price
    discount_badge_group
    discount_badge_text
    decoration_group
    sparkle
    star_decoration
    ornament
    overlay_effect_group
    glow_effect
    shine_effect

    Naming rules (align with banner reading above):
    - Main brand/logo area: brand_group; split name/logo parts with `brand_name_*` / `brand_mark`; stacked logo
      layers → still brand-related, not random vectors.
    - Yandex text/vector: brand_name_yandex.
    - Lavka text/vector: brand_name_lavka.
    - Heart/logo mark: brand_mark.
    - Main headline vs sub-headline: use **relative size/weight and proximity**—main is the dominant line; sub is
      smaller and paired with it (delivery, weight, etc.).
    - Delivery promise like "от 15 минут": delivery_info_group / delivery_time_text.
    - Legal copy: smallest/thinnest text block(s) → legal_group / legal_text.
    - "0+", "6+", "12+", "16+", "18+": age_badge_group / age_badge_text.
    - Product/photo/person area: hero_group or product_visual_group.
    - Stars/snowflakes/lights/confetti/glow/bulb motifs: decoration_group and specific decoration vocabulary.
    - Large color/photo background plate: background_group / base_background.

    Return the modified semantic JSON in the same general Figma JSON structure, with semantic "name" values on
    **all** nodes and unnecessary wrappers collapsed only where allowed above.
    """
).strip()


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


def _unwrap_model_json_text(text: str) -> str:
    """
    Strip optional markdown fences. If the model opened ```json but never closed ``` (truncated), drop the
    opening fence line only so parsing starts at ``{``.
    """
    t = (text or "").strip()
    fence = re.search(r"```(?:json|text)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1 :].lstrip()
    rt = t.rstrip()
    if rt.endswith("```"):
        t = t[: t.rfind("```")].rstrip()
    return t


def _strip_trailing_commas_outside_strings(s: str) -> str:
    """Remove JSON trailing commas (`,}` / `,]`) outside double-quoted strings."""
    out: list[str] = []
    i = 0
    n = len(s)
    in_string = False
    escape_next = False
    while i < n:
        ch = s[i]
        if in_string:
            out.append(ch)
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == '"':
                slashes = 0
                j = len(out) - 2
                while j >= 0 and out[j] == "\\":
                    slashes += 1
                    j -= 1
                if slashes % 2 == 0:
                    in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and s[j] in " \t\n\r":
                j += 1
            if j < n and s[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _strip_full_line_double_slash_comments(s: str) -> str:
    """Remove lines that are only ``//`` comments (models often inject invalid ``// ...`` in JSON)."""
    out: list[str] = []
    for line in s.splitlines():
        if line.lstrip().startswith("//"):
            continue
        out.append(line)
    return "\n".join(out)


def _normalize_json_string_literals_for_parse(s: str) -> str:
    """
    Fix characters inside JSON double-quoted string literals that ``json.loads`` rejects:
    raw ASCII controls (U+0000–U+001F), Unicode line/paragraph separators (U+2028, U+2029, U+0085), BOM.
    Models often copy Figma text containing U+2028 instead of a normal space or ``\\n``.
    """
    out: list[str] = []
    in_string = False
    escape_next = False
    i = 0
    while i < len(s):
        ch = s[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                slashes = 0
                j = i - 1
                while j >= 0 and s[j] == "\\":
                    slashes += 1
                    j -= 1
                if slashes % 2 == 0:
                    in_string = True
            i += 1
            continue
        if escape_next:
            out.append(ch)
            escape_next = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            escape_next = True
            i += 1
            continue
        if ch == '"':
            slashes = 0
            j = i - 1
            while j >= 0 and s[j] == "\\":
                slashes += 1
                j -= 1
            if slashes % 2 == 0:
                in_string = False
            out.append(ch)
            i += 1
            continue
        o = ord(ch)
        if o < 0x20:
            if ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(f"\\u{o:04x}")
        elif o in (0x2028, 0x2029, 0x0085):
            out.append(" ")
        elif o == 0xFEFF:
            pass
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def extract_first_json_object(text: str) -> Any:
    """Parse first top-level JSON object from model text (strips optional ``` fences)."""
    t = _unwrap_model_json_text(text)
    t = _strip_full_line_double_slash_comments(t)
    t = _normalize_json_string_literals_for_parse(t)
    t = _strip_trailing_commas_outside_strings(t)
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


def extract_first_json_value(text: str) -> Any:
    """Parse first top-level JSON object or array from model text (handles ```json fences)."""
    t = _unwrap_model_json_text(text)
    t = _strip_full_line_double_slash_comments(t)
    t = _normalize_json_string_literals_for_parse(t)
    t = _strip_trailing_commas_outside_strings(t)
    brace = t.find("{")
    bracket = t.find("[")
    if brace < 0 and bracket < 0:
        raise ValueError("No JSON object or array found in model output")
    if brace < 0:
        start = bracket
    elif bracket < 0:
        start = brace
    else:
        start = min(brace, bracket)
    decoder = json.JSONDecoder()
    slice_ = t[start:]
    try:
        value, _end = decoder.raw_decode(slice_)
    except json.JSONDecodeError as exc:
        tail = slice_.rstrip()
        looks_truncated = bool(tail) and tail[0] == "{" and not tail.endswith("}")
        if tail.startswith("["):
            looks_truncated = not tail.endswith("]")
        hint = ""
        if looks_truncated:
            hint = (
                " Model output looks truncated (incomplete JSON at end). "
                "Raise max_new_tokens (e.g. 4096) and ask the model for compact JSON (no indentation)."
            )
        raise ValueError(f"Invalid JSON in model output: {exc}.{hint}") from exc
    return value


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
