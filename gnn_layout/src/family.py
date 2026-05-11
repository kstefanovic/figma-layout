"""Creative-family grouping for leakage-safe source/target pairs."""

from __future__ import annotations

import re

from .semantic_utils import find_role_node, get_all_text


def get_family_key(banner: dict) -> str:
    """
    Build a stable semantic key so directed pairs stay inside one creative/template family.

    Vector logos often contain no text, so the key is intentionally weighted toward templateId,
    headline, subheadline-like text, and the legal copy prefix.
    """
    template_id = _strip_size_tokens(_norm_text(str(banner.get("templateId") or banner.get("template_id") or "")))
    headline = _role_text(banner, "headline_group")
    legal = _role_text(banner, "legal_text")[:80]
    subheadline = _subheadline_text(banner)
    root_name = _strip_size_tokens(_norm_text(str(banner.get("name") or "")))
    fallback = f"{template_id} {root_name}".strip()
    parts = [template_id, headline, subheadline, legal]
    if not headline and not subheadline:
        parts.append(fallback)
    key = "|".join(part for part in parts if part)
    return key or _strip_size_tokens(_norm_text(str(banner.get("name") or banner.get("id") or "unknown_family")))


def _role_text(banner: dict, role: str) -> str:
    node = find_role_node(banner, role)
    return _norm_text(get_all_text(node)) if node else ""


def _subheadline_text(banner: dict) -> str:
    texts: list[str] = []
    for node in _walk(banner):
        name = _norm_text(str(node.get("name") or ""))
        if "subheadline" in name:
            text = _norm_text(get_all_text(node))
            if text:
                texts.append(text)
    return " ".join(texts)[:160]


def _walk(node: dict):
    if isinstance(node, dict):
        yield node
        for child in node.get("children") or []:
            yield from _walk(child)


def _norm_text(value: str) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _strip_size_tokens(value: str) -> str:
    text = re.sub(r"\b\d{2,5}\s*[x×]\s*\d{2,5}\b", "", value)
    text = re.sub(r"\b(dooh|banner|frame)\b", "", text)
    text = re.sub(r"[_·|:-]+", " ", text)
    return _norm_text(text)
