from __future__ import annotations

from typing import Any

from .feature_extractor import (
    detect_visual_archetype,
    flatten_nodes,
    get_bounds,
    normalize_box,
)


LEGAL_MARKERS = ("Доставку", "Продавец", "Реклама", "ООО", "ОГРН", "ИНН")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _area_norm(box: list[float]) -> float:
    return max(0.0, box[2]) * max(0.0, box[3])


def _has_text(node: dict) -> bool:
    if str(node.get("type") or "").lower() == "text":
        return True
    stack = list(node.get("children") or [])
    while stack:
        cur = stack.pop()
        if not isinstance(cur, dict):
            continue
        if str(cur.get("type") or "").lower() == "text" or cur.get("characters"):
            return True
        stack.extend(cur.get("children") or [])
    return False


def _has_star(node: dict) -> bool:
    stack = [node]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, dict):
            continue
        name = str(cur.get("name") or "").lower()
        typ = str(cur.get("type") or "").lower()
        if typ == "star" or "star" in name:
            return True
        stack.extend(cur.get("children") or [])
    return False


def _raw_name_is_numeric(node: dict) -> bool:
    return str(node.get("name") or "").strip().isdigit()


def _union(boxes: list[list[float]]) -> list[float] | None:
    boxes = [b for b in boxes if b and b[2] > 0 and b[3] > 0]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def _candidate_payload(idx: int, item: dict, score: float, reasons: list[str], root_w: float, root_h: float) -> dict:
    node = item["node"]
    return {
        "candidate_id": f"c{idx:03d}",
        "figma_id": str(node.get("id") or ""),
        "path": str(node.get("path") or item.get("path") or ""),
        "score": round(float(score), 4),
        "bbox_norm": normalize_box(get_bounds(node), root_w, root_h),
        "reasons": reasons,
    }


def select_raw_visual_candidates(raw_banner: dict) -> dict:
    rb = get_bounds(raw_banner)
    root_w = max(_num(rb.get("width")), 1.0)
    root_h = max(_num(rb.get("height")), 1.0)
    flat = flatten_nodes(raw_banner)

    text_boxes: list[list[float]] = []
    for item in flat:
        if int(item.get("depth") or 0) == 0:
            continue
        node = item["node"]
        text = str(item.get("text") or node.get("characters") or "")
        typ = str(node.get("type") or "").lower()
        if typ != "text" and not text:
            continue
        low = text.lower()
        b = normalize_box(get_bounds(node), root_w, root_h)
        if any(m.lower() in low for m in LEGAL_MARKERS) or _area_norm(b) > 0.01:
            text_boxes.append(b)
    text_zone_box = _union(text_boxes)

    hero_scored: list[tuple[float, dict, list[str]]] = []
    bg_scored: list[tuple[float, dict, list[str]]] = []
    brand_scored: list[tuple[float, dict, list[str]]] = []
    headline_scored: list[tuple[float, dict, list[str]]] = []
    legal_scored: list[tuple[float, dict, list[str]]] = []

    for item in flat:
        node = item["node"]
        typ = str(node.get("type") or "").lower()
        box = normalize_box(get_bounds(node), root_w, root_h)
        area = _area_norm(box)
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        text = str(item.get("text") or node.get("characters") or "")
        low_text = text.lower()

        if typ == "text" or text:
            legal_score = 0.0
            legal_reasons: list[str] = []
            if any(m.lower() in low_text for m in LEGAL_MARKERS):
                legal_score += 0.65
                legal_reasons.append("legal_keyword")
            if y > 0.55:
                legal_score += 0.15
                legal_reasons.append("lower_text_position")
            if area <= 0.04:
                legal_score += 0.10
                legal_reasons.append("small_dense_text")
            if legal_score >= 0.45:
                legal_scored.append((legal_score, item, legal_reasons))

            if legal_score < 0.45:
                headline_score = 0.0
                headline_reasons: list[str] = []
                if area > 0.01:
                    headline_score += min(area * 6.0, 0.35)
                    headline_reasons.append("large_text_area")
                if 0.10 <= y <= 0.80:
                    headline_score += 0.15
                    headline_reasons.append("main_message_zone")
                if len(text.strip()) >= 6:
                    headline_score += 0.15
                    headline_reasons.append("meaningful_text")
                if typ in {"frame", "group"}:
                    headline_score += 0.10
                    headline_reasons.append("text_container")
                if headline_score >= 0.25:
                    headline_scored.append((headline_score, item, headline_reasons))

        if typ in {"frame", "group", "vector"} and not _has_star(node):
            brand_score = 0.0
            brand_reasons: list[str] = []
            aspect = w / max(h, 1e-6)
            # Brand rows are compact logo/wordmark clusters, never large image wrappers or off-canvas hero frames.
            if (
                0.002 <= area <= 0.12
                and 0.6 <= aspect <= 12.0
                and -0.05 <= y <= 0.55
                and 0.02 <= h <= 0.32
            ):
                brand_score += 0.25
                brand_reasons.append("compact_horizontal_cluster")
                if y < 0.35:
                    brand_score += 0.2
                    brand_reasons.append("upper_brand_zone")
                if typ in {"group", "frame"}:
                    brand_score += 0.15
                    brand_reasons.append("cluster_type")
                if not text:
                    brand_score += 0.10
                    brand_reasons.append("vector_wordmark_like")
            if brand_score >= 0.35:
                brand_scored.append((brand_score, item, brand_reasons))

        if typ not in {"frame", "group", "rectangle", "vector"}:
            continue
        if area <= 0.015:
            continue
        if _has_text(node) or _has_star(node):
            continue
        bleed = x < -0.02 or y < -0.02 or x + w > 1.02 or y + h > 1.02
        name_numeric = _raw_name_is_numeric(node)

        hero_score = 0.0
        hero_reasons: list[str] = []
        if typ in {"rectangle", "frame", "group"}:
            hero_score += 0.25
            hero_reasons.append("image_like_type")
        if area > 0.12:
            hero_score += min(area, 0.6)
            hero_reasons.append("large_area")
        if bleed:
            hero_score += 0.15
            hero_reasons.append("partial_bleed")
        if cx > 0.45 or cy < 0.55:
            hero_score += 0.08
            hero_reasons.append("visual_zone_position")
        if name_numeric:
            hero_score -= 0.03
        if hero_score >= 0.35:
            hero_scored.append((hero_score, item, hero_reasons))

        bg_score = 0.0
        bg_reasons: list[str] = []
        if typ in {"rectangle", "vector", "frame", "group"}:
            bg_score += 0.2
            bg_reasons.append("shape_like_type")
        if area > 0.08:
            bg_score += min(area * 0.9, 0.45)
            bg_reasons.append("large_support_shape")
        if text_zone_box:
            tx, ty, tw, th = text_zone_box
            overlap_x = max(0.0, min(x + w, tx + tw) - max(x, tx))
            overlap_y = max(0.0, min(y + h, ty + th) - max(y, ty))
            if overlap_x * overlap_y > 0:
                bg_score += 0.25
                bg_reasons.append("supports_text_zone")
        if bleed and (w > 0.45 or h > 0.45):
            bg_score += 0.08
            bg_reasons.append("canvas_bleed_shape")
        if bg_score >= 0.30:
            bg_scored.append((bg_score, item, bg_reasons))

    hero_scored.sort(key=lambda x: x[0], reverse=True)
    bg_scored.sort(key=lambda x: x[0], reverse=True)
    brand_scored.sort(key=lambda x: x[0], reverse=True)
    headline_scored.sort(key=lambda x: x[0], reverse=True)
    legal_scored.sort(key=lambda x: x[0], reverse=True)
    hero_candidates = [
        _candidate_payload(i + 1, item, score, reasons, root_w, root_h)
        for i, (score, item, reasons) in enumerate(hero_scored[:5])
    ]
    bg_candidates = [
        _candidate_payload(i + 1, item, score, reasons, root_w, root_h)
        for i, (score, item, reasons) in enumerate(bg_scored[:5])
    ]
    brand_candidates = [
        _candidate_payload(i + 1, item, score, reasons, root_w, root_h)
        for i, (score, item, reasons) in enumerate(brand_scored[:5])
    ]
    headline_candidates = [
        _candidate_payload(i + 1, item, score, reasons, root_w, root_h)
        for i, (score, item, reasons) in enumerate(headline_scored[:5])
    ]
    legal_candidates = [
        _candidate_payload(i + 1, item, score, reasons, root_w, root_h)
        for i, (score, item, reasons) in enumerate(legal_scored[:5])
    ]

    selected = {
        "hero_image": hero_candidates[0] if hero_candidates and hero_candidates[0]["score"] >= 0.35 else None,
        "background_shape": None,
        "brand_group": brand_candidates[0] if brand_candidates and brand_candidates[0]["score"] >= 0.35 else None,
        "headline_group": headline_candidates[0] if headline_candidates and headline_candidates[0]["score"] >= 0.25 else None,
        "legal_text": legal_candidates[0] if legal_candidates and legal_candidates[0]["score"] >= 0.45 else None,
    }
    hero_path = str((selected["hero_image"] or {}).get("path") or "")
    filtered_bg_candidates = []
    for cand in bg_candidates:
        bg_path = str(cand.get("path") or "")
        overlaps_hero = (
            hero_path
            and bg_path
            and (
                bg_path == hero_path
                or hero_path.startswith(bg_path + "/")
                or bg_path.startswith(hero_path + "/")
            )
        )
        if not overlaps_hero:
            filtered_bg_candidates.append(cand)
    bg_candidates = filtered_bg_candidates
    selected["background_shape"] = bg_candidates[0] if bg_candidates and bg_candidates[0]["score"] >= 0.30 else None
    estimated_archetype = detect_visual_archetype(
        selected["hero_image"]["bbox_norm"] if selected["hero_image"] else None,
        selected["background_shape"]["bbox_norm"] if selected["background_shape"] else None,
        text_zone_box,
        root_w,
        root_h,
    )
    return {
        "hero_image_candidates": hero_candidates,
        "background_shape_candidates": bg_candidates,
        "brand_group_candidates": brand_candidates,
        "headline_group_candidates": headline_candidates,
        "legal_text_candidates": legal_candidates,
        "selected": selected,
        "text_zone_box": text_zone_box,
        "estimated_archetype": estimated_archetype,
    }

