"""Flatten Figma export raw JSON to leaf-only mid JSON; helpers for Qwen semantic naming."""

from __future__ import annotations

import copy
import json
import re
import textwrap
from typing import Any

# Banner + grid + mid layout JSON → model outputs ``{"names":{...}}``; server merges into the full export tree.
FIGMA_CONVERT_SYSTEM_PROMPT = textwrap.dedent(
    """
    You output exactly one small JSON object of the form ``{"names":{"<figma_node_id>":"<semantic_snake_name>",...}}``.
    The server builds the final nested JSON **only** from ``mid`` rows (plus your ``names``); wrapper ids removed at
    mid flattening must **not** appear in ``names``. Do **not** echo the full tree, do **not** pretty-print, do **not**
    wrap in markdown fences.

    Hard requirements:
    - Every id in ``required_node_ids`` (same set as the ``mid`` rows) must appear as a key in ``names`` (omit none).
    - Do **not** output keys for any id that is not a row in ``mid``.
    - Values must be meaningful English ``snake_case`` from the vocabulary (or a 2–4 word phrase built only from those
      roots). Never use lazy defaults as final names: bare digits, "Frame", "Rectangle", "Vector", "Text", "Group …",
      pixel-size strings like "960x512", or the raw ``id`` string as the semantic name.
    - Never emit ``//`` or ``/* */`` comments; output must be strictly parseable JSON.
    - Use **compact** JSON: a single line is ideal—no indentation, no extra newlines outside string values.
    - In ``names`` string values you only output identifiers—no Unicode line separators (U+2028 / U+2029).
    - The root node’s semantic name must be ``banner_root`` unless the design is clearly not a single banner frame.
    - Decorative clutter → ``decoration_group`` / ``sparkle`` / ``ornament`` / ``glow_effect`` etc., not generic shapes.
    - Infer roles from the **banner image** using **typography and layering** (size, weight, color, position), and
      from the **grid** thumbnails keyed by ``id``—not from Figma ``type`` alone.
    """
).strip()

# FIGMA_CONVERT_PROMPT = textwrap.dedent(
#     """
#     You are a strict Figma semantic **renaming** assistant (vision + layout JSON).

#     Inputs:
#     1. Full banner image.
#     2. Grid image: each cell has a **grey header bar** with text `id:<figma_id>` and, below it, a **thumbnail** of that
#        element. The `<figma_id>` matches the `"id"` field in the raw JSON for the same element—use it to align grid
#        crops with JSON nodes.
#     3. Raw Figma JSON with anonymous names (your job is only better ``name`` strings, keyed by ``id``).

#     Output shape (mandatory — nothing else):
#     - Return **only** one JSON object: ``{"names":{"<id>":"<snake_case_semantic>", ...}}``.
#     - Include **every** ``"id"`` value from the raw JSON tree (all depths). No markdown, no code fences, no commentary.
#     - **Compact** JSON (ideally one line) so the response is short.

#     How to read the banner (image 1) — typical retail / promo layout:
#     - **Hero / product visual:** Often a large rectangle (photo or flat fill) for the product or scene; map to
#       `hero_group`, `product_visual_group`, `main_product`, or `product_packshot` by what is shown.
#     - **Background:** Often a colored rectangle or **vector** shapes behind content (e.g. blue panel); use
#       `background_group` / `base_background` when it is clearly a plate or wash behind other layers—not the hero
#       product itself.
#     - **Brand group:** Cluster of **brand name** (text/vector; sometimes **two separate parts**—e.g. two wordmarks)
#       plus **logo / mark**. The logo may sit on stacked layers: a **rear** backing shape and a **foreground** mark;
#       keep them under `brand_group` with `brand_mark`, `brand_name_yandex`, or `brand_name_lavka` as appropriate, not
#       as unrelated generic vectors.
#     - **Headline group:** Usually **two text roles** near each other:
#       - **Main headline:** The **largest and/or boldest** promotional line in the whole banner—what the product is,
#         the hero offer, or the **current price** emphasis. Prefer `headline_text` / `headline_line` / `offer_group` /
#         `current_price` by content.
#       - **Sub-headline:** **Smaller** type, often **medium weight** (not hairline), placed **closest** to the main
#         headline—delivery window, weight, pack size, secondary promise. Map to `subheadline_text`,
#         `delivery_info_group`, `delivery_time_text` when it matches that role (not the legal footer).
#     - **Legal / compliance copy:** The **smallest and thinnest** text in the layout—addresses, SKU, disclaimers,
#       “conditions apply”, etc. Almost always `legal_group` / `legal_text` even if it sits above the bottom edge.
#     - **Age badge:** Small badge with “0+”, “6+”, “12+”, etc. → `age_badge_group` / `age_badge_text`.
#     - **Decorations:** Non-message flourishes—stars, glows, bulbs, sparkles, confetti, shine—→ `decoration_group`,
#       `sparkle`, `star_decoration`, `glow_effect`, `shine_effect`, `ornament`, `overlay_effect_group` as fits.

#     Use this mental model together with **bounds** and **grid thumbnails** so each raw `id` gets the right semantic
#     role even when Figma node types are generic (`RECTANGLE`, `VECTOR`, `TEXT`).

#     Rules:
#     - Do not explain. Do not output the full Figma tree—only the ``names`` map.
#     - Do not invent ids; keys must match raw JSON ``"id"`` strings exactly.
#     - Values are semantic **names** only (snake_case); the server keeps geometry, ``characters``, etc. unchanged.

#     Semantic naming vocabulary:
#     banner_root
#     background_group
#     base_background
#     brand_group
#     brand_name_yandex
#     brand_mark
#     brand_name_lavka
#     headline_group
#     headline_text
#     headline_line
#     delivery_info_group
#     delivery_time_text
#     subheadline_text
#     legal_group
#     legal_text
#     age_badge_group
#     age_badge_text
#     hero_group
#     product_visual_group
#     main_product
#     product_packshot
#     product_label
#     offer_group
#     price_group
#     current_price
#     old_price
#     discount_badge_group
#     discount_badge_text
#     decoration_group
#     sparkle
#     star_decoration
#     ornament
#     overlay_effect_group
#     glow_effect
#     shine_effect

#     Naming rules (align with banner reading above):
#     - Main brand/logo area: brand_group; split name/logo parts with `brand_name_*` / `brand_mark`; stacked logo
#       layers → still brand-related, not random vectors.
#     - Yandex text/vector: brand_name_yandex.
#     - Lavka text/vector: brand_name_lavka.
#     - Heart/logo mark: brand_mark.
#     - Main headline vs sub-headline: use **relative size/weight and proximity**—main is the dominant line; sub is
#       smaller and paired with it (delivery, weight, etc.).
#     - Delivery promise like "от 15 минут": delivery_info_group / delivery_time_text.
#     - Legal copy: smallest/thinnest text block(s) → legal_group / legal_text.
#     - "0+", "6+", "12+", "16+", "18+": age_badge_group / age_badge_text.
#     - Product/photo/person area: hero_group or product_visual_group.
#     - Stars/snowflakes/lights/confetti/glow/bulb motifs: decoration_group and specific decoration vocabulary.
#     - Large color/photo background plate: background_group / base_background.

#     Example (format only — ids are fake): ``{"names":{"1:2":"banner_root","1:3":"hero_group"}}``
#     """
# ).strip()

FIGMA_CONVERT_PROMPT = textwrap.dedent(
    """
    You are a strict semantic naming assistant for flattened Figma banner nodes.

INPUTS:
1. Full banner image.
2. Atlas/grid image. Each cell has `id:<figma_id>` and a thumbnail of that mid block.
3. Layout JSON with ``mid`` (flat blocks) and ``required_node_ids`` (same ids as ``mid`` rows—no removed wrappers).

TASK:
Assign one semantic snake_case name to every id in ``required_node_ids`` / ``mid``.

OUTPUT ONLY compact JSON:
{"names":{"<id>":"<semantic_name>"}}

RULES:
- Include every ``required_node_ids`` entry exactly once (same as every ``mid`` row ``id``).
- Keys must exactly match those ids only—never include a raw wrapper id that was omitted from ``mid``.
- Do not invent ids.
- Do not output markdown, explanation, bbox, or full tree.
- Do not reuse anonymous raw names like `1`, `2`, `Group 123`, `Rectangle 5`, `unknown`, `object`.
- If unsure, use `unassigned`.

ALLOWED NAMES:
banner_root, unassigned,

brand_group, brand_name, brand_name_first, brand_name_second,
brand_name_first_part_1, brand_name_first_part_2, brand_name_first_part_3,
brand_name_second_part_1, brand_name_second_part_2, brand_name_second_part_3,
logo, logo_back, logo_fore, brand_mark,

headline_group, headline, subheadline, subheadline_delivery_time,
subheadline_weight, subheadline_product_name, subheadline_discount, product_name,

legal_text_group, legal_text,
age_badge_group, age_badge,

price_group, current_price, old_price, currency_symbol,
discount_badge_group, discount_badge, discount_text,

hero_group, image_zone, hero_image, person_image,
product_group, product_image, product_packshot, product_label,
product_controller, food_image, drink_image, medicine_image,

background_group, base_background, background_shape, color_panel, gradient_shape,

decoration_group, decoration,
star_decoration, star_decoration_1, star_decoration_2, star_decoration_3,
sparkle, sparkle_1, sparkle_2, sparkle_3,
glow_effect, glow_effect_1, glow_effect_2,
shine_effect, shine_effect_1, shine_effect_2,
confetti, ornament, snowflake, light_bulb

ROLE RULES:
1. Root/frame:
- Main frame if present: `banner_root`.

2. Brand:
- Whole brand cluster: `brand_group`.
- First brand word: `brand_name_first`.
- Second brand word: `brand_name_second`.
- One-piece brand: `brand_name`.
- If a brand word is split into vector pieces, name pieces left-to-right:
  `brand_name_first_part_1`, `brand_name_first_part_2`, etc.
- Logo/icon wrapper: `logo`.
- Logo outer/back/circle/heart/blob layer: `logo_back`.
- Logo inner/front/mark layer: `logo_fore`.
- Brand logo is NOT decoration.

Yandex Lavka example:
- Яндекс = `brand_name_first` or `brand_name_first_part_*`
- heart/circle logo = `logo_back` + `logo_fore`
- Лавка = `brand_name_second` or `brand_name_second_part_*`

3. Headline:
- Largest/boldest main marketing message: `headline`.
- Smaller support line near headline: `subheadline`.
- Delivery/time phrases become `subheadline_delivery_time`.
Examples:
`с доставкой от 15 минут`, `от 15 минут`, `привезёт курьер от 15 минут`.

4. Legal:
- Small dense disclaimer/compliance text: `legal_text`.
Legal indicators:
`ООО`, `ОГРН`, `ИНН`, `Реклама`, `Рекламодатель`, `Продавец`,
`Количество товаров ограничено`, `Доставку осуществляют`, address-like text.
Do NOT classify headline or delivery promise as legal text.

5. Age badge:
- Text exactly `0+`, `3+`, `6+`, `12+`, `16+`, `18+` = `age_badge`.
Age badge is NOT price, discount, or legal text.

6. Price / discount:
- Dominant price: `current_price`.
- Crossed/secondary price: `old_price`.
- ₽/$ if separate: `currency_symbol`.
- Discount badge shape: `discount_badge`.
- Discount text like `-52%`: `discount_text`.

7. Images/products:
- Main photo/person/scene area: `hero_image` or `hero_group`.
- Product object/photo: `product_image`, `product_packshot`, `food_image`, `drink_image`, `medicine_image`.
- Image crop/zone: `image_zone`.
- Product label text/shape on product: `product_label`.
Do NOT call product/photo image `background`.

8. Background:
- Large color plate/full banner fill: `base_background`.
- Large abstract color shape: `background_shape`.
- Panel behind text: `color_panel`.
- Gradient/fade shape: `gradient_shape`.

9. Decorations:
- Stars, sparkles, glows, shine, ornaments, snowflakes, bulbs, confetti:
  `star_decoration_*`, `sparkle_*`, `glow_effect_*`, `shine_effect_*`, `ornament`, `snowflake`, `light_bulb`.
- Use numeric suffixes left-to-right, then top-to-bottom.

ID RULE:
You are given mid_json, not raw_json.
Only output names for ids that exist in mid_json.
Do not output ids from raw_json wrappers.
Do not output parent wrapper ids that were removed during flattening.
If an id is not present in mid_json, never include it.

The final names map must have exactly the same id set as mid_json non-null ids.

IMPORTANT MID_JSON RULE:
Only name existing mid_json ids. Do not infer raw wrapper ids.

For flattened mid_json:
- a text node with legal content is legal_text, not legal_text_group
- a text node with 0+ is age_badge, not age_badge_group
- a wrapper containing headline + delivery text is headline_group
- a compact vector row with brand word parts + logo parts is brand_group
- compact brand vectors are never hero_group
- delivery phrases like "с доставкой от 15 минут" are subheadline_delivery_time

PRIORITY:
1. Age regex wins.
2. Legal indicators + tiny dense text = legal_text.
3. Large bold promo text = headline.
4. Delivery/time phrase = subheadline_delivery_time.
5. Brand row/logo pieces = brand roles.
6. Product/photo = hero/product/image roles.
7. Stars/glows = decoration roles.
8. If uncertain = unassigned.

USE:
- Full banner for context.
- Atlas thumbnails for exact element identity.
- mid_json characters, type, bounds, font size, and parent/child hints.
- Shared parent/child hints help identify split brand/logo pieces.

FINAL CHECK:
Every non-null id included once.
Output exactly:
{"names":{"id":"semantic_name"}}
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


def _normalize_figma_type(node: dict[str, Any]) -> str:
    return str(node.get("type") or "").lower().replace("_", " ")


def _is_unnecessary_wrapper(node: dict[str, Any], *, is_document_root: bool) -> bool:
    """
    Passthrough / empty layout containers to omit from the parallel mid list.

    - Document root frame is never treated as a wrapper (even if single-child).
    - INSTANCE and TEXT are never wrappers.
    - GROUP / FRAME with 0 or 1 child are wrappers (hoist children or drop empties).
    """
    if is_document_root:
        return False
    typ = _normalize_figma_type(node)
    if typ == "instance":
        return False
    if typ == "text":
        return False
    if typ not in ("group", "frame"):
        return False
    ch = node.get("children")
    if not isinstance(ch, list):
        ch = []
    return len(ch) <= 1


def _index_figma_subtree(
    node: dict[str, Any],
    parent_id: str | None,
    nodes: dict[str, dict[str, Any]],
    parent_map: dict[str, str | None],
) -> None:
    nid = node.get("id")
    if nid is None:
        return
    sid = str(nid)
    nodes[sid] = node
    parent_map[sid] = parent_id
    for c in node.get("children") or []:
        if isinstance(c, dict) and c.get("id") is not None:
            _index_figma_subtree(c, sid, nodes, parent_map)


def _semantic_parent_ids_for_mid(
    node_id: str,
    nodes: dict[str, dict[str, Any]],
    parent_map: dict[str, str | None],
    root_ids: set[str],
) -> list[str]:
    """Meaningful ancestors root→parent (raw wrapper group/frame ids skipped)."""
    out: list[str] = []
    cur: str | None = parent_map.get(node_id)
    while cur:
        pn = nodes.get(cur)
        if pn is None:
            break
        is_root = cur in root_ids
        if not _is_unnecessary_wrapper(pn, is_document_root=is_root):
            out.append(cur)
        cur = parent_map.get(cur)
    out.reverse()
    return out


def _semantic_child_ids_for_mid(
    node: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    root_ids: set[str],
) -> list[str]:
    """Meaningful child ids under ``node``; single-child wrappers are flattened through."""
    result: list[str] = []
    for c in node.get("children") or []:
        if not isinstance(c, dict) or c.get("id") is None:
            continue
        cid = str(c["id"])
        cn = nodes.get(cid)
        if cn is None:
            continue
        is_root = cid in root_ids
        if _is_unnecessary_wrapper(cn, is_document_root=is_root):
            result.extend(_semantic_child_ids_for_mid(cn, nodes, root_ids))
        else:
            result.append(cid)
    return result


def raw_fig_tree_to_mid_blocks(raw: Any, frame_index: int = 0) -> list[dict[str, Any]]:
    """
    Flat **mid** representation of Figma-like raw JSON for the convert-semantic pipeline.

    - Drops unnecessary wrapper rows: ``group`` / ``frame`` with 0 or 1 child (document root never dropped).
    - All remaining blocks are a **parallel** list (no nested ``children``); each row keeps the node’s own fields
      except ``children``, plus:
    - ``mid_parent_ids``: meaningful ancestor ids from root → parent (wrappers omitted from the chain).
    - ``mid_child_ids``: meaningful child ids (wrappers hoisted through).
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

    nodes: dict[str, dict[str, Any]] = {}
    parent_map: dict[str, str | None] = {}
    for r in roots:
        if isinstance(r, dict) and r.get("id") is not None:
            _index_figma_subtree(r, None, nodes, parent_map)
    root_ids = {str(r["id"]) for r in roots if isinstance(r, dict) and r.get("id") is not None}

    out: list[dict[str, Any]] = []

    def emit_walk(n: dict[str, Any]) -> None:
        nid = n.get("id")
        if nid is None:
            return
        sid = str(nid)
        is_root = sid in root_ids
        if _is_unnecessary_wrapper(n, is_document_root=is_root):
            for c in n.get("children") or []:
                if isinstance(c, dict):
                    emit_walk(c)
            return
        row = {k: v for k, v in n.items() if k != "children"}
        row["mid_parent_ids"] = _semantic_parent_ids_for_mid(sid, nodes, parent_map, root_ids)
        row["mid_child_ids"] = _semantic_child_ids_for_mid(n, nodes, root_ids)
        out.append(row)
        for c in n.get("children") or []:
            if isinstance(c, dict):
                emit_walk(c)

    for r in roots:
        emit_walk(r)
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
                "Raise max_new_tokens if needed; prefer compact one-line JSON or the {\"names\":{}} id map format."
            )
        raise ValueError(f"Invalid JSON in model output: {exc}.{hint}") from exc
    return value


def collect_node_ids_from_figma_raw(root: Any) -> set[str]:
    """All string ``id`` values reachable from a Figma-like raw JSON root (dict or list of frames)."""
    out: set[str] = set()

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            nid = n.get("id")
            if nid is not None:
                out.add(str(nid))
            for c in n.get("children") or []:
                walk(c)
        elif isinstance(n, list):
            for item in n:
                walk(item)

    walk(root)
    return out


def merge_semantic_names_into_raw_tree(raw: Any, names: dict[str, str]) -> Any:
    """Deep copy ``raw`` and overwrite each node's ``name`` when ``str(id)`` is in ``names``."""
    tree: Any = copy.deepcopy(raw)

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            nid = n.get("id")
            if nid is not None and str(nid) in names:
                n["name"] = names[str(nid)]
            for c in n.get("children") or []:
                walk(c)
        elif isinstance(n, list):
            for item in n:
                walk(item)

    walk(tree)
    return tree


def collect_allowed_ids_from_mid(mid_blocks: list[dict[str, Any]]) -> set[str]:
    """Concrete Figma ids allowed in convert-semantic ``final_json`` (one row per id in ``mid``)."""
    return {str(b["id"]) for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}


def _bounds_area(bounds: Any) -> float:
    if not isinstance(bounds, dict):
        return 0.0
    try:
        w = float(bounds.get("width") or 0)
        h = float(bounds.get("height") or 0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, w * h)


_AGE_BADGE_TEXT = re.compile(r"^\s*\d{1,2}\s*\+\s*$", re.UNICODE)
_AGE_BADGE_STRICT = re.compile(r"^\s*(0|3|6|12|16|18)\+\s*$", re.UNICODE)

_LEGAL_MARKERS = (
    "ООО",
    "ОГРН",
    "ИНН",
    "Реклама",
    "Рекламодатель",
    "Продавец",
    "Количество товаров ограничено",
    "Доставку осуществляют",
)

_DELIVERY_MARKERS = (
    "с доставкой",
    "от 15 минут",
    "15 минут",
    "доставка",
    "доставкой",
    "привезёт",
    "привезет",
    "курьер",
)


def validate_semantic_names(
    mid_json: list[dict[str, Any]],
    names_map: dict[str, str],
) -> list[str]:
    """
    Validate semantic ``id -> name`` map against ``mid_json`` geometry and text rules.
    Returns a list of human-readable warning strings (empty if all checks pass).
    """
    warnings: list[str] = []
    allowed = collect_allowed_ids_from_mid(mid_json)
    keys = {str(k) for k in names_map.keys()}
    if keys != allowed:
        warnings.append(
            f"v1_ids_mismatch extra={sorted(keys - allowed)[:24]} missing={sorted(allowed - keys)[:24]}"
        )

    mid_by_id: dict[str, dict[str, Any]] = {
        str(b["id"]): b for b in mid_json if isinstance(b, dict) and b.get("id") is not None
    }

    def _nm(sid: str) -> str:
        return str(names_map.get(sid, "") or "").lower()

    def _norm_chars(s: str) -> str:
        return " ".join((s or "").replace("\r", "\n").split()).strip()

    root_row = None
    for b in mid_json:
        if isinstance(b, dict) and not (b.get("mid_parent_ids") or []):
            root_row = b
            break
    frame_bounds = (root_row or {}).get("bounds") or {}
    try:
        frame_w = float(frame_bounds.get("width") or 0)
        frame_h = float(frame_bounds.get("height") or 0)
    except (TypeError, ValueError):
        frame_w, frame_h = 0.0, 0.0
    frame_area = _bounds_area(frame_bounds)

    def _has_text_descendant(start: str) -> bool:
        stack = [start]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            row = mid_by_id.get(cur)
            if not row:
                continue
            if str(row.get("type") or "").lower() == "text":
                return True
            for cid in row.get("mid_child_ids") or []:
                cs = str(cid)
                if cs in mid_by_id:
                    stack.append(cs)
        return False

    def _has_brand_or_headline_ancestor(sid: str) -> bool:
        row = mid_by_id.get(sid)
        if not row:
            return False
        for pid in row.get("mid_parent_ids") or []:
            p = str(pid)
            g = _nm(p)
            if g in ("brand_group", "headline_group"):
                return True
        return False

    # v2–v5 text
    for sid, row in mid_by_id.items():
        if str(row.get("type") or "").lower() != "text":
            continue
        chars = _norm_chars(str(row.get("characters") or ""))
        cmp_age = chars.replace(" ", "")
        low = chars.lower()
        n = _nm(sid)
        if _AGE_BADGE_STRICT.match(cmp_age) and n != "age_badge":
            warnings.append(f"v2_age_badge_text:{sid}:{names_map.get(sid)}")
        if any(m.lower() in low for m in _LEGAL_MARKERS) and n != "legal_text":
            warnings.append(f"v3_legal_text:{sid}:{names_map.get(sid)}")
        if any(m in low for m in _DELIVERY_MARKERS) and n not in ("subheadline_delivery_time", "subheadline"):
            warnings.append(f"v4_delivery_subheadline:{sid}:{names_map.get(sid)}")
        fs = row.get("fontSize")
        try:
            font_size = float(fs) if fs is not None else 0.0
        except (TypeError, ValueError):
            font_size = 0.0
        try:
            th = float((row.get("bounds") or {}).get("height") or 0)
        except (TypeError, ValueError):
            th = 0.0
        age_hit = bool(_AGE_BADGE_STRICT.match(cmp_age))
        legal_hit = any(m.lower() in low for m in _LEGAL_MARKERS)
        del_hit = any(m in low for m in _DELIVERY_MARKERS)
        headline_hit = (not age_hit) and (not legal_hit) and (not del_hit) and (
            font_size >= 180.0 or th >= 500.0 or (font_size >= 120.0 and len(chars) >= 12)
        )
        if headline_hit and n != "headline":
            warnings.append(f"v5_headline_prominent_text:{sid}:{names_map.get(sid)}")

    def _direct_text_ids(sid: str) -> list[str]:
        row = mid_by_id.get(sid)
        if not row:
            return []
        out_ids: list[str] = []
        for cid in row.get("mid_child_ids") or []:
            cs = str(cid)
            if cs in mid_by_id and str(mid_by_id[cs].get("type") or "").lower() == "text":
                out_ids.append(cs)
        return out_ids

    def _headline_block_text_children(sid: str) -> bool:
        tch = _direct_text_ids(sid)
        if len(tch) < 2:
            return False
        names = [_nm(cid) for cid in tch]
        has_h = any(
            n == "headline" or (n.startswith("headline") and n != "headline_group") for n in names if n
        )
        has_s = any(
            n == "subheadline_delivery_time"
            or n == "subheadline"
            or (n.startswith("subheadline") and "group" not in n)
            for n in names
            if n
        )
        return bool(has_h and has_s)

    # v6–v7 headline_group
    for sid, row in mid_by_id.items():
        if _nm(sid) != "headline_group":
            continue
        if not _has_text_descendant(sid):
            warnings.append(f"v7_headline_group_no_text_descendants:{sid}")
        elif not _headline_block_text_children(sid):
            warnings.append(f"v6_headline_group_missing_headline_sub_pair:{sid}")

    # v8 compact brand mis-tagged
    def _is_visible_node(r: dict[str, Any]) -> bool:
        return r.get("visible") is not False

    def _nested_logo_child_id(sid: str) -> str | None:
        row = mid_by_id.get(sid)
        if not row:
            return None
        for cid in row.get("mid_child_ids") or []:
            cs = str(cid)
            if cs not in mid_by_id:
                continue
            sub = mid_by_id[cs]
            st = str(sub.get("type") or "").lower()
            if st not in ("frame", "group", "instance"):
                continue
            subs = [str(x) for x in (sub.get("mid_child_ids") or []) if str(x) in mid_by_id]
            vecs = [x for x in subs if str(mid_by_id[x].get("type") or "").lower() == "vector"]
            if 1 <= len(vecs) <= 2 and len(vecs) == len(subs):
                return cs
        return None

    def _is_brand_geometry(sid: str) -> bool:
        row = mid_by_id.get(sid)
        if not row:
            return False
        typ = str(row.get("type") or "").lower()
        if typ not in ("frame", "group", "instance"):
            return False
        ch_ids = [str(c) for c in (row.get("mid_child_ids") or []) if str(c) in mid_by_id]
        vis = [c for c in ch_ids if _is_visible_node(mid_by_id[c])]
        if len(vis) < 3:
            return False
        for c in vis:
            if str(mid_by_id[c].get("type") or "").lower() != "text":
                continue
            tn = _nm(c)
            if tn == "headline" or (tn.startswith("headline") and tn != "headline_group"):
                return False
        vf = sum(
            1
            for c in vis
            if str(mid_by_id[c].get("type") or "").lower() in ("vector", "frame", "group", "instance")
        )
        if vf < max(3, int(0.6 * len(vis))):
            return False
        if _nested_logo_child_id(sid) is None:
            return False
        try:
            rw = float((row.get("bounds") or {}).get("width") or 0)
            rh = float((row.get("bounds") or {}).get("height") or 0)
            y0 = float((row.get("bounds") or {}).get("y") or 0)
        except (TypeError, ValueError):
            return False
        if rh <= 0 or rw / rh < 2.0:
            return False
        if rh > min(700.0, 0.35 * frame_h) if frame_h > 0 else 700.0:
            return False
        if frame_h > 0 and y0 > 0.55 * frame_h:
            return False
        xs: list[float] = []
        for c in vis:
            b = mid_by_id[c].get("bounds") or {}
            x = float(b.get("x") or 0)
            w = float(b.get("width") or 0)
            xs.extend([x, x + w])
        if xs and (max(xs) - min(xs)) < 0.45 * rw:
            return False
        return True

    for sid, row in mid_by_id.items():
        if not _is_brand_geometry(sid):
            continue
        g = _nm(sid)
        if g in ("hero_group", "headline_group", "product_group"):
            warnings.append(f"v8_brand_row_wrong_group:{sid}:{names_map.get(sid)}")

    # v9–v10 logo under brand_group
    for sid, row in mid_by_id.items():
        if _nm(sid) != "brand_group":
            continue
        lid = _nested_logo_child_id(sid)
        if lid is None:
            continue
        if _nm(lid) not in ("logo",):
            warnings.append(f"v9_brand_nested_vectors_not_logo:{lid}:{names_map.get(lid)}")
        sub = mid_by_id.get(lid)
        if not sub:
            continue
        subs = [str(x) for x in (sub.get("mid_child_ids") or []) if str(x) in mid_by_id]
        vecs = [x for x in subs if str(mid_by_id[x].get("type") or "").lower() == "vector"]
        if len(vecs) == 2:
            a0 = _bounds_area(mid_by_id[vecs[0]].get("bounds"))
            a1 = _bounds_area(mid_by_id[vecs[1]].get("bounds"))
            larger, smaller = (vecs[0], vecs[1]) if a0 >= a1 else (vecs[1], vecs[0])
            if _nm(larger) != "logo_back":
                warnings.append(f"v10_logo_back:{larger}:{names_map.get(larger)}")
            if _nm(smaller) != "logo_fore":
                warnings.append(f"v10_logo_fore:{smaller}:{names_map.get(smaller)}")

    # v11–v12 huge vector
    for sid, row in mid_by_id.items():
        if str(row.get("type") or "").lower() != "vector":
            continue
        if _has_brand_or_headline_ancestor(sid):
            continue
        b = row.get("bounds") or {}
        try:
            nw = float(b.get("width") or 0)
            nh = float(b.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if frame_w <= 0 or frame_h <= 0 or frame_area <= 0:
            continue
        area_ratio = (nw * nh) / frame_area
        wr, hr = nw / frame_w, nh / frame_h
        if not (wr > 0.35 or hr > 0.35 or area_ratio > 0.20):
            continue
        n = _nm(sid)
        if n in ("logo", "logo_back", "logo_fore"):
            warnings.append(f"v11_huge_vector_logo_name:{sid}:{names_map.get(sid)}")
        if n == "hero_image":
            warnings.append(f"v12_huge_vector_hero_image:{sid}:{names_map.get(sid)}")
        elif area_ratio > 0.20 and n not in ("background_shape", "base_background", "color_panel", "unassigned"):
            warnings.append(f"v12_huge_abstract_vector_name:{sid}:{names_map.get(sid)}")

    return warnings


def postprocess_semantic_names(
    mid_json: list[dict[str, Any]],
    names_map: dict[str, Any],
    debug: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Deterministic corrections after VLM ``names`` output. Returns a new ``id -> semantic_name`` map.

    ``debug`` may be a dict mutated with ``postprocess_used``, ``forced_renames``, ``semantic_validation_warnings``.
    """
    dbg = debug if debug is not None else {}
    dbg["postprocess_used"] = True
    dbg.setdefault("forced_renames", [])
    # Compact top brand rows (vectors + one logo cluster) must never keep these container roles.
    _FORBIDDEN_BRAND_ROW_PARENT = frozenset(
        {
            "hero_group",
            "headline_group",
            "product_group",
            "hero_image",
            "legal_text_group",
            "background_group",
            "unassigned",
            "",
        }
    )

    def _track(sid: str, old: str, new: str, reason: str) -> None:
        if old == new:
            return
        dbg["forced_renames"].append({"id": sid, "old": old, "new": new, "reason": reason})

    def _force(out: dict[str, str], sid: str, new: str, reason: str) -> None:
        old = out.get(sid, "")
        out[sid] = new
        _track(sid, old, new, reason)

    allowed = collect_allowed_ids_from_mid(mid_json)
    mid_by_id: dict[str, dict[str, Any]] = {
        str(b["id"]): b for b in mid_json if isinstance(b, dict) and b.get("id") is not None
    }

    out: dict[str, str] = {}
    for k, v in names_map.items():
        sk = str(k)
        if sk not in allowed:
            continue
        if isinstance(v, str) and v.strip():
            out[sk] = v.strip()
    for sid in allowed:
        if sid not in out:
            _force(out, sid, "unassigned", "rule1_missing_id")

    # --- Rule 2: TEXT roles (priority age > legal > delivery > headline) ---
    def _norm_chars(s: str) -> str:
        return " ".join((s or "").replace("\r", "\n").split()).strip()

    for sid, row in mid_by_id.items():
        typ = str(row.get("type") or "").lower()
        if typ != "text":
            continue
        chars_raw = row.get("characters")
        chars = _norm_chars(str(chars_raw) if chars_raw is not None else "")
        chars_cmp = chars.replace(" ", "")
        age_hit = bool(_AGE_BADGE_STRICT.match(chars_cmp))
        low = chars.lower()
        legal_hit = any(m.lower() in low for m in _LEGAL_MARKERS)
        del_hit = any(m in low for m in _DELIVERY_MARKERS)

        fs = row.get("fontSize")
        try:
            font_size = float(fs) if fs is not None else 0.0
        except (TypeError, ValueError):
            font_size = 0.0
        try:
            th = float((row.get("bounds") or {}).get("height") or 0)
        except (TypeError, ValueError):
            th = 0.0
        headline_hit = (not age_hit) and (not legal_hit) and (not del_hit) and (
            font_size >= 180.0 or th >= 500.0 or (font_size >= 120.0 and len(chars) >= 12)
        )

        if age_hit:
            _force(out, sid, "age_badge", "rule2a_age_badge_text")
        elif legal_hit:
            _force(out, sid, "legal_text", "rule2b_legal_indicators")
        elif del_hit:
            _force(out, sid, "subheadline_delivery_time", "rule2c_delivery_phrase")
        elif headline_hit:
            _force(out, sid, "headline", "rule2d_headline_prominent_text")

    # TEXT nodes must never keep *_group suffixes for legal/age (VLM mistakes)
    for sid, row in mid_by_id.items():
        if str(row.get("type") or "").lower() != "text":
            continue
        nm = out.get(sid, "").lower()
        chars_cmp = _norm_chars(str(row.get("characters") or "")).replace(" ", "")
        if nm == "legal_text_group":
            _force(out, sid, "legal_text", "rule2_fix_text_misnamed_legal_text_group")
        elif nm == "age_badge_group" and _AGE_BADGE_STRICT.match(chars_cmp):
            _force(out, sid, "age_badge", "rule2_fix_text_misnamed_age_badge_group")

    # --- Root metrics for ratio rules ---
    root_row = None
    for b in mid_json:
        if isinstance(b, dict) and not (b.get("mid_parent_ids") or []):
            root_row = b
            break
    frame_bounds = (root_row or {}).get("bounds") or {}
    try:
        frame_w = float(frame_bounds.get("width") or 0)
        frame_h = float(frame_bounds.get("height") or 0)
    except (TypeError, ValueError):
        frame_w, frame_h = 0.0, 0.0
    frame_area = _bounds_area(frame_bounds)

    # --- Rule 3+: helpers ---
    def _child_names(sid: str) -> list[str]:
        row = mid_by_id.get(sid)
        if not row:
            return []
        return [out.get(str(cid), "") for cid in (row.get("mid_child_ids") or []) if str(cid) in mid_by_id]

    def _direct_text_child_ids(sid: str) -> list[str]:
        row = mid_by_id.get(sid)
        if not row:
            return []
        out_ids: list[str] = []
        for cid in (row.get("mid_child_ids") or []):
            cs = str(cid)
            if cs not in mid_by_id:
                continue
            if str(mid_by_id[cs].get("type") or "").lower() == "text":
                out_ids.append(cs)
        return out_ids

    def _has_text_descendant(sid: str) -> bool:
        stack = [sid]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            r = mid_by_id.get(cur)
            if not r:
                continue
            if str(r.get("type") or "").lower() == "text":
                return True
            for cid in r.get("mid_child_ids") or []:
                cs = str(cid)
                if cs in mid_by_id:
                    stack.append(cs)
        return False

    def _text_child_headline_and_delivery_sub(sid: str) -> bool:
        tch = _direct_text_child_ids(sid)
        if len(tch) < 2:
            return False
        names = [out.get(cid, "").lower() for cid in tch]
        has_headline = any(
            n == "headline" or (n.startswith("headline") and n != "headline_group") for n in names if n
        )
        has_sub = any(
            n == "subheadline_delivery_time"
            or n == "subheadline"
            or (n.startswith("subheadline") and "group" not in n)
            for n in names
            if n
        )
        return bool(has_headline and has_sub)

    def _is_visible_node(r: dict[str, Any]) -> bool:
        return r.get("visible") is not False

    def _is_logo_like_cluster_child_id(cs: str) -> bool:
        """Single child under a brand row: small frame/group with 1–2 vectors only, or a boolean logo cluster."""
        sub = mid_by_id.get(cs)
        if not sub:
            return False
        st = _normalize_figma_type(sub)
        if st in ("frame", "group", "instance"):
            subs = [str(x) for x in (sub.get("mid_child_ids") or []) if str(x) in mid_by_id]
            vecs = [x for x in subs if _normalize_figma_type(mid_by_id[x]) == "vector"]
            if 1 <= len(vecs) <= 2 and len(vecs) == len(subs):
                return True
            return False
        if st == "boolean operation":
            subs = [str(x) for x in (sub.get("mid_child_ids") or []) if str(x) in mid_by_id]
            if not subs:
                return False
            return all(
                _normalize_figma_type(mid_by_id[x]) in ("vector", "boolean operation") for x in subs
            )
        return False

    def _nested_logo_frame_child_id(sid: str) -> str | None:
        row = mid_by_id.get(sid)
        if not row:
            return None
        for cid in (row.get("mid_child_ids") or []):
            cs = str(cid)
            if cs not in mid_by_id:
                continue
            if _is_logo_like_cluster_child_id(cs):
                return cs
        return None

    def _is_brand_row_compact_one_logo(sid: str) -> bool:
        """Compact horizontal row, no text under it, exactly one logo-like boolean/group child → brand_group."""
        row = mid_by_id.get(sid)
        if not row:
            return False
        typ = str(row.get("type") or "").lower()
        if typ not in ("frame", "group", "instance"):
            return False
        if _has_text_descendant(sid):
            return False
        ch_ids = [str(c) for c in (row.get("mid_child_ids") or []) if str(c) in mid_by_id]
        vis = [c for c in ch_ids if _is_visible_node(mid_by_id[c])]
        if len(vis) < 2:
            return False
        logo_like = [c for c in vis if _is_logo_like_cluster_child_id(c)]
        if len(logo_like) != 1:
            return False
        logo_id = logo_like[0]
        others = [c for c in vis if c != logo_id]
        if not others:
            return False
        # Wordmark pieces are direct vectors; avoid rows with extra frames/booleans as siblings.
        if not all(_normalize_figma_type(mid_by_id[c]) == "vector" for c in others):
            return False
        try:
            rw = float((row.get("bounds") or {}).get("width") or 0)
            rh = float((row.get("bounds") or {}).get("height") or 0)
            y0 = float((row.get("bounds") or {}).get("y") or 0)
        except (TypeError, ValueError):
            return False
        if rh <= 0 or rw < 160:
            return False
        if rw / rh < 2.0:
            return False
        if frame_h > 0 and rh > min(520.0, 0.30 * frame_h):
            return False
        if frame_h > 0 and y0 > 0.62 * frame_h:
            return False
        return True

    def _is_brand_row_rule5(sid: str) -> bool:
        row = mid_by_id.get(sid)
        if not row:
            return False
        typ = str(row.get("type") or "").lower()
        if typ not in ("frame", "group", "instance"):
            return False
        ch_ids = [str(c) for c in (row.get("mid_child_ids") or []) if str(c) in mid_by_id]
        vis = [c for c in ch_ids if _is_visible_node(mid_by_id[c])]
        if len(vis) < 3:
            return False
        for c in vis:
            if str(mid_by_id[c].get("type") or "").lower() != "text":
                continue
            tn = out.get(c, "").lower()
            if tn == "headline" or (tn.startswith("headline") and tn != "headline_group"):
                return False
        vf = sum(
            1
            for c in vis
            if _normalize_figma_type(mid_by_id[c])
            in ("vector", "frame", "group", "instance", "boolean operation")
        )
        if vf < max(3, int(0.6 * len(vis))):
            return False
        if _nested_logo_frame_child_id(sid) is None:
            return False
        try:
            rw = float((row.get("bounds") or {}).get("width") or 0)
            rh = float((row.get("bounds") or {}).get("height") or 0)
            y0 = float((row.get("bounds") or {}).get("y") or 0)
        except (TypeError, ValueError):
            return False
        if rh <= 0 or rw / rh < 2.0:
            return False
        if rh > min(700.0, 0.35 * frame_h) if frame_h > 0 else 700.0:
            return False
        if frame_h > 0 and y0 > 0.55 * frame_h:
            return False
        xs: list[float] = []
        for c in vis:
            b = mid_by_id[c].get("bounds") or {}
            x = float(b.get("x") or 0)
            w = float(b.get("width") or 0)
            xs.extend([x, x + w])
        if xs and (max(xs) - min(xs)) < 0.45 * rw:
            return False
        return True

    def _is_compact_vector_brand_row(sid: str) -> bool:
        return _is_brand_row_rule5(sid) or _is_brand_row_compact_one_logo(sid)

    def _has_real_legal_text_descendant(root_sid: str) -> bool:
        """True if some descendant TEXT node is classified ``legal_text`` (post rule2)."""
        stack = [
            str(c)
            for c in (mid_by_id.get(root_sid, {}).get("mid_child_ids") or [])
            if str(c) in mid_by_id
        ]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen or cur == root_sid:
                continue
            seen.add(cur)
            r = mid_by_id.get(cur)
            if not r:
                continue
            if str(r.get("type") or "").lower() == "text" and out.get(cur, "").lower() == "legal_text":
                return True
            for cid in r.get("mid_child_ids") or []:
                cs = str(cid)
                if cs in mid_by_id:
                    stack.append(cs)
        return False

    # --- Rule 5: compact vector brand row ---
    for sid, row in mid_by_id.items():
        if not _is_brand_row_rule5(sid):
            continue
        cur = out.get(sid, "").lower()
        if cur in _FORBIDDEN_BRAND_ROW_PARENT:
            _force(out, sid, "brand_group", "rule5_compact_vector_brand_row")

    # --- User rule 2: compact horizontal row, no text, one logo-like cluster → brand_group ---
    for sid, row in mid_by_id.items():
        if not _is_brand_row_compact_one_logo(sid):
            continue
        cur = out.get(sid, "").lower()
        if cur in _FORBIDDEN_BRAND_ROW_PARENT:
            _force(out, sid, "brand_group", "user_rule2_compact_row_one_logo_cluster")

    # --- Rule 3: headline_group (valid headline + delivery text children only) ---
    for sid, row in mid_by_id.items():
        if not _text_child_headline_and_delivery_sub(sid):
            continue
        _force(out, sid, "headline_group", "rule3_headline_block_direct_text_children")

    for sid, row in mid_by_id.items():
        if out.get(sid, "").lower() != "headline_group":
            continue
        if not _has_text_descendant(sid) or not _text_child_headline_and_delivery_sub(sid):
            if _is_compact_vector_brand_row(sid):
                _force(out, sid, "brand_group", "rule3_invalid_headline_group_to_brand")
            else:
                _force(out, sid, "unassigned", "rule3_headline_group_cleared_no_valid_text_block")

    for sid, row in mid_by_id.items():
        if not _text_child_headline_and_delivery_sub(sid):
            continue
        if out.get(sid, "").lower() == "product_group":
            _force(out, sid, "headline_group", "rule3_product_group_with_headline_block")

    # --- Rule 3b/3c: legal / age wrapper groups (not root) ---
    for sid, row in mid_by_id.items():
        if not (row.get("mid_parent_ids") or []):
            continue
        typ = str(row.get("type") or "").lower()
        if typ == "text":
            continue
        if not _has_real_legal_text_descendant(sid):
            continue
        _force(out, sid, "legal_text_group", "rule3b_container_with_legal_text_descendant")
    for sid, row in mid_by_id.items():
        if out.get(sid, "").lower() != "legal_text_group":
            continue
        if str(row.get("type") or "").lower() == "text":
            continue
        if _has_real_legal_text_descendant(sid):
            continue
        if _is_compact_vector_brand_row(sid):
            _force(out, sid, "brand_group", "rule3b_clear_legal_group_vector_only_brand_row")
        else:
            _force(out, sid, "unassigned", "rule3b_clear_legal_group_without_real_legal_text")
    for sid, row in mid_by_id.items():
        if not (row.get("mid_parent_ids") or []):
            continue
        ch_names = _child_names(sid)
        if any(n == "age_badge" for n in ch_names):
            typ = str(row.get("type") or "").lower()
            if typ != "text":
                _force(out, sid, "age_badge_group", "rule3c_container_with_age_badge_child")

    # --- Rule 3e: large hero instance ---
    for sid, row in mid_by_id.items():
        typ = str(row.get("type") or "").lower()
        if typ != "instance":
            continue
        ch_ids = [str(c) for c in (row.get("mid_child_ids") or []) if str(c) in mid_by_id]
        if len(ch_ids) != 1:
            continue
        ch = mid_by_id[ch_ids[0]]
        ctyp = str(ch.get("type") or "").lower()
        carea = _bounds_area(ch.get("bounds"))
        if ctyp in ("rectangle", "vector", "frame") and carea >= max(4_000_000.0, 0.25 * frame_area):
            _force(out, sid, "hero_group", "rule3e_large_instance_visual_block")
            if ctyp == "rectangle":
                _force(out, ch_ids[0], "hero_image", "rule3e_child_large_rectangle")

    # --- Rule 6b: boolean / mixed cluster as ``logo`` under brand_group ---
    for sid, row in mid_by_id.items():
        if _normalize_figma_type(row) != "boolean operation":
            continue
        pids = row.get("mid_parent_ids") or []
        if not pids:
            continue
        pid = str(pids[-1])
        if out.get(pid, "").lower() != "brand_group":
            continue
        if _is_logo_like_cluster_child_id(sid):
            _force(out, sid, "logo", "rule6b_boolean_cluster_as_logo")

    # --- Rule 6: logo nested frame under brand_group ---
    for sid, row in mid_by_id.items():
        ch_ids = [str(c) for c in (row.get("mid_child_ids") or []) if str(c) in mid_by_id]
        vecs = [c for c in ch_ids if str(mid_by_id[c].get("type") or "").lower() == "vector"]
        if not (1 <= len(vecs) <= 2) or len(vecs) != len(ch_ids):
            continue
        pid = str((row.get("mid_parent_ids") or [])[-1]) if (row.get("mid_parent_ids") or []) else ""
        if not pid or out.get(pid, "").lower() != "brand_group":
            continue
        _force(out, sid, "logo", "rule6_logo_nested_frame_vectors")
        if len(vecs) == 2:
            a0 = _bounds_area(mid_by_id[vecs[0]].get("bounds"))
            a1 = _bounds_area(mid_by_id[vecs[1]].get("bounds"))
            if a0 >= a1:
                _force(out, vecs[0], "logo_back", "rule6_area_larger_logo_back")
                _force(out, vecs[1], "logo_fore", "rule6_area_smaller_logo_fore")
            else:
                _force(out, vecs[1], "logo_back", "rule6_area_larger_logo_back")
                _force(out, vecs[0], "logo_fore", "rule6_area_smaller_logo_fore")
        else:
            _force(out, vecs[0], "logo_back", "rule6_single_vector_logo_mark")

    def _leaf_vector_ids_under_logo(logo_id: str) -> list[str]:
        acc: list[str] = []
        stack = [logo_id]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur != logo_id:
                sub = mid_by_id.get(cur)
                if sub and str(sub.get("type") or "").lower() == "vector" and not (sub.get("mid_child_ids") or []):
                    acc.append(cur)
                    continue
            subw = mid_by_id.get(cur)
            if not subw:
                continue
            for xid in subw.get("mid_child_ids") or []:
                xs = str(xid)
                if xs in mid_by_id:
                    stack.append(xs)
        return acc

    def _mid_descendant_id_set(start: str) -> set[str]:
        out_d: set[str] = set()
        stk = [start]
        vis: set[str] = set()
        while stk:
            cur = stk.pop()
            if cur in vis:
                continue
            vis.add(cur)
            out_d.add(cur)
            rw = mid_by_id.get(cur)
            if not rw:
                continue
            for xid in rw.get("mid_child_ids") or []:
                xs = str(xid)
                if xs in mid_by_id:
                    stk.append(xs)
        return out_d

    # --- Rule 6c: largest / second leaf vectors under any ``logo`` cluster ---
    for sid, row in mid_by_id.items():
        if out.get(sid, "").lower() != "logo":
            continue
        leaf_vecs = _leaf_vector_ids_under_logo(sid)
        if len(leaf_vecs) < 1:
            continue
        leaf_vecs.sort(
            key=lambda vid: _bounds_area(mid_by_id[vid].get("bounds")),
            reverse=True,
        )
        _force(out, leaf_vecs[0], "logo_back", "rule6c_largest_leaf_vector_under_logo")
        if len(leaf_vecs) >= 2:
            _force(out, leaf_vecs[1], "logo_fore", "rule6c_second_leaf_vector_under_logo")

    # --- Rule 7: brand wordmark vectors by x relative to logo (skip vectors inside logo subtree) ---
    for bid, brow in mid_by_id.items():
        if out.get(bid, "").lower() != "brand_group":
            continue
        ch_ids = [str(c) for c in (brow.get("mid_child_ids") or []) if str(c) in mid_by_id]
        logo_children = [c for c in ch_ids if out.get(c, "").lower() == "logo"]
        if len(logo_children) != 1:
            continue
        lid = logo_children[0]
        logo_interior = _mid_descendant_id_set(lid)
        logo_interior.discard(lid)
        ordered: list[tuple[float, str, str]] = []
        for c in ch_ids:
            if c in logo_interior:
                continue
            r = mid_by_id[c]
            t = str(r.get("type") or "").lower()
            b = r.get("bounds") or {}
            bx = float(b.get("x") or 0)
            if t == "vector":
                ordered.append((bx, c, "v"))
            elif c == lid:
                ordered.append((bx, c, "logo"))
        ordered.sort(key=lambda z: z[0])
        try:
            i_logo = next(i for i, z in enumerate(ordered) if z[2] == "logo")
        except StopIteration:
            continue
        left = [z[1] for z in ordered[:i_logo] if z[2] == "v"]
        right = [z[1] for z in ordered[i_logo + 1 :] if z[2] == "v"]
        for i, vid in enumerate(left):
            _force(out, vid, f"brand_name_first_part_{i + 1}", "rule7_vectors_left_of_logo_by_x")
        if len(right) == 1:
            _force(out, right[0], "brand_name_second", "rule7_vector_right_of_logo")
        elif len(right) > 1:
            for i, vid in enumerate(right[:-1]):
                _force(out, vid, f"brand_name_second_part_{i + 1}", "rule7_vectors_right_of_logo_piece")
            _force(out, right[-1], "brand_name_second", "rule7_vectors_right_of_logo_last")

    def _has_brand_or_headline_ancestor(sid: str) -> bool:
        row = mid_by_id.get(sid)
        if not row:
            return False
        for pid in row.get("mid_parent_ids") or []:
            g = out.get(str(pid), "").lower()
            if g in ("brand_group", "headline_group"):
                return True
        return False

    # --- Rule 4: huge abstract vector (not under brand/headline) ---
    for sid, row in mid_by_id.items():
        if str(row.get("type") or "").lower() != "vector":
            continue
        if _has_brand_or_headline_ancestor(sid):
            continue
        b = row.get("bounds") or {}
        try:
            nw = float(b.get("width") or 0)
            nh = float(b.get("height") or 0)
            y0 = float(b.get("y") or 0)
        except (TypeError, ValueError):
            continue
        if frame_w <= 0 or frame_h <= 0 or frame_area <= 0:
            continue
        area_ratio = (nw * nh) / frame_area
        wr, hr = nw / frame_w, nh / frame_h
        if not (wr > 0.35 or hr > 0.35 or area_ratio > 0.20):
            continue
        nm = out.get(sid, "").lower()
        if wr >= 0.92 and hr >= 0.92:
            newn = "base_background"
        elif frame_h > 0 and y0 > 0.62 * frame_h and area_ratio < 0.14:
            newn = "color_panel"
        else:
            newn = "background_shape"
        if nm in ("logo", "logo_back", "logo_fore", "hero_image") or nm not in (
            "background_shape",
            "base_background",
            "color_panel",
        ):
            if nm != newn:
                _force(out, sid, newn, "rule4_huge_vector_background_family")

    # --- User rule 1: large leaf rectangle cannot be brand_group / brand_name_* ---
    for sid, row in mid_by_id.items():
        if str(row.get("type") or "").lower() != "rectangle":
            continue
        if row.get("mid_child_ids"):
            continue
        nm = out.get(sid, "").lower()
        if nm != "brand_group" and not nm.startswith("brand_name"):
            continue
        area = _bounds_area(row.get("bounds"))
        if frame_area <= 0 or area < 0.08 * frame_area:
            continue
        b = row.get("bounds") or {}
        try:
            nw = float(b.get("width") or 0)
            nh = float(b.get("height") or 0)
        except (TypeError, ValueError):
            nw, nh = 0.0, 0.0
        ar = area / frame_area if frame_area > 0 else 0.0
        wr, hr = (nw / frame_w if frame_w else 0.0), (nh / frame_h if frame_h else 0.0)
        long_ratio = max(nw, nh) / max(min(nw, nh), 0.01)
        if wr >= 0.82 and hr >= 0.82 and ar >= 0.42:
            newn = "hero_image"
        elif long_ratio >= 3.0 and max(wr, hr) >= 0.5:
            newn = "background_shape"
        elif ar >= 0.16 or area >= 1_200_000.0:
            newn = "image_zone"
        else:
            newn = "background_shape"
        _force(out, sid, newn, "user_rule1_large_leaf_rectangle_not_brand")

    # --- Rule 8: wrong container roles on compact brand row (VLm / ordering fallout) ---
    for sid, row in mid_by_id.items():
        if not _is_compact_vector_brand_row(sid):
            continue
        cur = out.get(sid, "").lower()
        if cur in _FORBIDDEN_BRAND_ROW_PARENT:
            _force(out, sid, "brand_group", "rule8_compact_brand_strip_wrong_container_role")

    # --- Rule 6a: star decorations ---
    for sid, row in mid_by_id.items():
        if str(row.get("type") or "").lower() != "star":
            continue
        pids = row.get("mid_parent_ids") or []
        pid = str(pids[-1]) if pids else ""
        par = mid_by_id.get(pid)
        if not par:
            continue
        siblings = [str(c) for c in (par.get("mid_child_ids") or []) if str(c) in mid_by_id]
        star_siblings = [c for c in siblings if str(mid_by_id[c].get("type") or "").lower() == "star"]
        if sid not in star_siblings:
            continue
        idx = star_siblings.index(sid) + 1
        _force(out, sid, f"star_decoration_{idx}", "rule6a_star_order_under_parent")

    def _descendant_names(start: str) -> list[str]:
        out_names: list[str] = []
        stack = [start]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            row = mid_by_id.get(cur)
            if not row:
                continue
            for cid in (row.get("mid_child_ids") or []):
                cs = str(cid)
                if cs not in mid_by_id:
                    continue
                out_names.append(out.get(cs, "").lower())
                stack.append(cs)
        return out_names

    soft_semantic: list[str] = []
    for sid, row in mid_by_id.items():
        typ = str(row.get("type") or "").lower()
        nm = out.get(sid, "").lower()
        if typ == "text" and nm == "legal_text_group":
            soft_semantic.append(f"semantic:legal_text_node_is_group_suffix:{sid}")
        if typ == "text" and _AGE_BADGE_STRICT.match(
            _norm_chars(str(row.get("characters") or "")).replace(" ", "")
        ) and nm == "age_badge_group":
            soft_semantic.append(f"semantic:age_text_is_age_badge_group:{sid}")
        if out.get(sid, "").lower() == "hero_group":
            dnames = _descendant_names(sid)
            if any(n == "headline" or (n.startswith("headline") and "group" not in n) for n in dnames if n):
                soft_semantic.append(f"semantic:hero_group_contains_headline_like:{sid}")
        if out.get(sid, "").lower() == "product_group":
            dnames = _descendant_names(sid)
            if any(n == "headline" or (n.startswith("headline") and "group" not in n) for n in dnames if n):
                soft_semantic.append(f"semantic:product_group_contains_headline_like:{sid}")

    # Root banner
    for sid, row in mid_by_id.items():
        if not (row.get("mid_parent_ids") or []):
            _force(out, sid, "banner_root", "rule_root_banner")

    dbg["semantic_validation_warnings"] = soft_semantic + validate_semantic_names(mid_json, out)

    return out


def build_semantic_figma_tree_from_mid(
    mid_blocks: list[dict[str, Any]],
    names: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    """
    Build a nested Figma-like JSON tree using only ``mid_blocks`` rows and ``mid_child_ids`` order.
    VLM ``names`` are applied only for ids in this mid list; other keys are ignored.
    """
    allowed_ids = collect_allowed_ids_from_mid(mid_blocks)
    if not allowed_ids:
        raise ValueError("Mid JSON has no nodes with ids")

    filtered_names = {str(k): str(v).strip() for k, v in names.items() if str(k) in allowed_ids}
    dropped_vlm = set(names.keys()) - allowed_ids
    for rid in sorted(dropped_vlm):
        warnings.append(f"removed_non_mid_json_id:{rid}")

    mid_by_id: dict[str, dict[str, Any]] = {}
    for b in mid_blocks:
        if not isinstance(b, dict) or b.get("id") is None:
            continue
        sid = str(b["id"])
        node = {k: copy.deepcopy(v) for k, v in b.items() if k not in ("mid_parent_ids", "mid_child_ids")}
        if sid in filtered_names:
            node["name"] = filtered_names[sid]
        node["children"] = []
        mid_by_id[sid] = node

    missing = allowed_ids - set(filtered_names.keys())
    if missing:
        warnings.append(
            f"Model omitted {len(missing)} mid id(s); those nodes keep names from mid_json rows."
        )

    for b in mid_blocks:
        if not isinstance(b, dict) or b.get("id") is None:
            continue
        sid = str(b["id"])
        if sid not in mid_by_id:
            continue
        ch_ids = [str(c) for c in (b.get("mid_child_ids") or []) if str(c) in mid_by_id]
        mid_by_id[sid]["children"] = [mid_by_id[c] for c in ch_ids]

    roots = [str(b["id"]) for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None and not (b.get("mid_parent_ids") or [])]
    if not roots:
        raise ValueError("Mid JSON has no root row (expected empty mid_parent_ids on the banner frame)")
    if len(roots) > 1:
        warnings.append(f"Multiple mid roots {roots}; using first as output root.")
    root_id = roots[0]
    if root_id not in mid_by_id:
        raise ValueError("Mid root id not found in indexed nodes")
    return mid_by_id[root_id]


def apply_semantic_corrections(
    root: dict[str, Any],
    _mid_by_id: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """
    Post-process semantic ``name`` values using mid geometry/type (age badge text, huge hero blocks, etc.).
    Mutates ``root`` in place.
    """
    frame_area = _bounds_area(root.get("bounds"))

    def walk(node: dict[str, Any]) -> None:
        if not isinstance(node, dict):
            return
        nid = node.get("id")
        sid = str(nid) if nid is not None else ""
        name = str(node.get("name") or "")
        nm = name.lower()
        typ = str(node.get("type") or "").lower().replace("_", " ")
        chars = node.get("characters") if isinstance(node.get("characters"), str) else ""
        area = _bounds_area(node.get("bounds"))

        if "age_badge" in nm:
            if typ != "text" or not _AGE_BADGE_TEXT.match(chars.strip() or ""):
                node["name"] = "star_decoration"
                warnings.append(f"semantic_heuristic:age_badge_invalid_type_or_text:{sid}")

        brandish = ("brand_name", "brand_mark", "brand_group")
        if any(b in nm for b in brandish) and typ in ("rectangle", "vector", "instance", "frame"):
            huge = area >= 4_000_000.0 or (frame_area > 0 and area >= 0.35 * frame_area)
            if huge:
                node["name"] = "hero_image"
                warnings.append(f"semantic_heuristic:brand_like_on_huge_visual_renamed_hero_image:{sid}")

        for ch in node.get("children") or []:
            if isinstance(ch, dict):
                walk(ch)

    walk(root)
    return root


def collect_all_figma_ids_from_tree(node: Any, *, include_semantic_prefix: bool = False) -> set[str]:
    out: set[str] = set()

    def walk(n: Any) -> None:
        if not isinstance(n, dict):
            return
        nid = n.get("id")
        if nid is not None:
            s = str(nid)
            if include_semantic_prefix or not s.startswith("semantic_"):
                out.add(s)
        for c in n.get("children") or []:
            walk(c)

    walk(node)
    return out


def validate_final_json_ids(
    final_json: Any,
    mid_blocks: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    allowed = collect_allowed_ids_from_mid(mid_blocks)
    found = collect_all_figma_ids_from_tree(final_json, include_semantic_prefix=False)
    bad = found - allowed
    if bad:
        warnings.append(f"validate_final_json_ids: non_mid ids still present after cleanup: {sorted(bad)}")


def remove_non_mid_json_nodes(
    root: dict[str, Any],
    allowed_ids: set[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Remove subtrees whose concrete Figma id is not in ``allowed_ids``; lift their children to the parent."""

    def process(node: dict[str, Any]) -> dict[str, Any] | None:
        nid = node.get("id")
        sid = str(nid) if nid is not None else ""
        new_children: list[dict[str, Any]] = []
        for ch in node.get("children") or []:
            if not isinstance(ch, dict):
                continue
            cid = ch.get("id")
            if cid is None:
                continue
            cs = str(cid)
            if cs.startswith("semantic_"):
                sub = process(ch)
                if sub is not None:
                    new_children.append(sub)
                continue
            if cs not in allowed_ids:
                warnings.append(f"removed_non_mid_json_id:{cs}")
                for gc in ch.get("children") or []:
                    if isinstance(gc, dict):
                        g2 = process(gc)
                        if g2 is not None:
                            new_children.append(g2)
                continue
            sub = process(ch)
            if sub is not None:
                new_children.append(sub)
        out = {k: v for k, v in node.items() if k != "children"}
        out["children"] = new_children
        if sid and not sid.startswith("semantic_") and sid not in allowed_ids:
            return None
        return out

    out = process(root)
    return out if out is not None else root


def lift_unassigned_wrappers_in_logo_subtrees(root: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """
    Under nodes named ``logo``, replace a single-child ``unassigned`` frame/group wrapper with its child
    (user rule: lift trivial wrappers inside the logo cluster).
    """
    def _nm(n: dict[str, Any]) -> str:
        return str(n.get("name") or "").lower()

    def _lift_chain_at_logo(node: dict[str, Any]) -> None:
        if _nm(node) != "logo":
            return
        changed = True
        while changed:
            changed = False
            ch = node.get("children")
            if not isinstance(ch, list):
                break
            new_ch: list[Any] = []
            for el in ch:
                if not isinstance(el, dict):
                    new_ch.append(el)
                    continue
                t = _normalize_figma_type(el)
                if (
                    _nm(el) == "unassigned"
                    and t in ("group", "frame")
                    and isinstance(el.get("children"), list)
                    and len(el["children"]) == 1
                    and isinstance(el["children"][0], dict)
                ):
                    warnings.append(f"lift_unassigned_single_wrapper_in_logo:{el.get('id')}")
                    new_ch.append(el["children"][0])
                    changed = True
                else:
                    new_ch.append(el)
            if changed:
                node["children"] = new_ch

    def walk(n: dict[str, Any]) -> None:
        if not isinstance(n, dict):
            return
        _lift_chain_at_logo(n)
        for c in n.get("children") or []:
            if isinstance(c, dict):
                walk(c)

    walk(root)
    return root


def normalize_convert_semantic_output(
    parsed: Any,
    mid_blocks: list[dict[str, Any]],
    warnings: list[str],
) -> Any:
    """
    If the model returned ``{"names":{...}}``, build nested semantic JSON **only** from ``mid_blocks``
    (no raw wrapper ids). Otherwise return ``parsed`` unchanged (legacy full tree from model).
    """
    if not isinstance(parsed, dict):
        return parsed
    if set(parsed.keys()) != {"names"}:
        return parsed
    names_obj = parsed.get("names")
    if not isinstance(names_obj, dict):
        return parsed
    names = {
        str(k): str(v).strip()
        for k, v in names_obj.items()
        if isinstance(v, str) and str(v).strip()
    }
    if not names:
        return parsed

    allowed_ids = collect_allowed_ids_from_mid(mid_blocks)
    mid_by_id = {str(b["id"]): b for b in mid_blocks if isinstance(b, dict) and b.get("id") is not None}

    pp_dbg: dict[str, Any] = {}
    names = postprocess_semantic_names(mid_blocks, names, pp_dbg)
    warnings.append("postprocess_debug:" + json.dumps(pp_dbg, ensure_ascii=False, separators=(",", ":")))

    tree = build_semantic_figma_tree_from_mid(mid_blocks, names, warnings)
    tree = lift_unassigned_wrappers_in_logo_subtrees(tree, warnings)
    tree = remove_non_mid_json_nodes(tree, allowed_ids, warnings)
    validate_final_json_ids(tree, mid_blocks, warnings)
    return tree


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
