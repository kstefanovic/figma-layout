"""Local Figma JSON embedding indexes grouped by class number.

The raw files are expected at:
  raw_jsons/1.json
  raw_jsons/2.json
  raw_jsons/3.json
  raw_jsons/4.json

Each raw file can contain many top-level Figma frame JSON objects. This module
embeds each top-level frame and stores one index per class.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


RAW_JSON_DIR = Path("raw_jsons")
EMBEDDING_DIR = Path("json_embeddings")
EMBED_DIM = 256
VALID_CLASSES = {1, 2, 3, 4}


def parse_aspect_ratio(value: str | float | int) -> float:
    """Parse aspect ratio from '16:9', '1080x1920', '1.777', or a number."""
    if isinstance(value, int | float):
        ratio = float(value)
    else:
        s = value.strip().lower()
        if not s:
            raise ValueError("aspect_ratio is empty")
        if ":" in s:
            left, right = s.split(":", 1)
            ratio = float(left.strip()) / float(right.strip())
        elif "x" in s:
            left, right = s.split("x", 1)
            ratio = float(left.strip()) / float(right.strip())
        elif "/" in s:
            left, right = s.split("/", 1)
            ratio = float(left.strip()) / float(right.strip())
        else:
            ratio = float(s)

    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"aspect_ratio must be a positive finite number, got {value!r}")
    return ratio


def parse_resolution(value: str) -> tuple[float, float]:
    """Parse target resolution like '2280x360' into (width, height)."""
    s = value.strip().lower()
    if "x" not in s:
        raise ValueError("target resolution must use WIDTHxHEIGHT, for example 2280x360")
    left, right = s.split("x", 1)
    width = float(left.strip())
    height = float(right.strip())
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError(f"target resolution must be positive finite WIDTHxHEIGHT, got {value!r}")
    return width, height


def _bounds(node: dict[str, Any]) -> dict[str, float]:
    b = node.get("bounds") or {}
    if not isinstance(b, dict):
        return {}
    out: dict[str, float] = {}
    for key in ("x", "y", "width", "height"):
        val = b.get(key)
        if isinstance(val, int | float):
            out[key] = float(val)
    return out


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for child in node.get("children") or []:
            yield from _walk(child)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _leaf_count(node: Any) -> int:
    count = 0
    for n in _walk(node):
        if not n.get("children"):
            count += 1
    return count


def _max_depth(node: Any, depth: int = 0) -> int:
    if not isinstance(node, dict):
        return depth
    children = node.get("children") or []
    if not children:
        return depth
    return max(_max_depth(child, depth + 1) for child in children if isinstance(child, dict))


def _text_blob(node: Any, limit: int = 12000) -> str:
    parts: list[str] = []
    for n in _walk(node):
        for key in ("name", "type", "characters"):
            val = n.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        if sum(len(p) for p in parts) > limit:
            break
    return " ".join(parts)[:limit]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\wА-Яа-яЁё+%-]+", text.lower(), re.UNICODE)


def _hash_token(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % EMBED_DIM


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def frame_embedding(node: dict[str, Any]) -> list[float]:
    b = _bounds(node)
    width = b.get("width", 1.0)
    height = b.get("height", 1.0)
    aspect = width / height if width > 0 and height > 0 else 1.0
    area = max(width * height, 1.0)
    leaf_count = _leaf_count(node)
    node_count = sum(1 for _ in _walk(node))
    depth = _max_depth(node)

    vec = [0.0] * EMBED_DIM

    # Strong numeric channels. The query only knows aspect ratio, so these are
    # intentionally weighted to dominate over weak text/name signals.
    numeric = [
        math.log(aspect),
        math.log(width + 1.0) / 10.0,
        math.log(height + 1.0) / 10.0,
        math.log(area + 1.0) / 20.0,
        min(leaf_count, 500) / 500.0,
        min(node_count, 2000) / 2000.0,
        min(depth, 20) / 20.0,
        1.0 if aspect >= 1 else 0.0,
        1.0 if aspect < 1 else 0.0,
    ]
    for i, val in enumerate(numeric):
        vec[i] = val * 6.0

    for token in _tokens(_text_blob(node)):
        idx = 32 + (_hash_token(token) % (EMBED_DIM - 32))
        vec[idx] += 1.0

    return _normalize(vec)


def aspect_query_embedding(aspect_ratio: str | float | int) -> list[float]:
    aspect = parse_aspect_ratio(aspect_ratio)
    # Use normalized pseudo-size so retrieval is mostly aspect based.
    width = max(aspect, 1.0)
    height = max(1.0 / aspect, 1.0)
    fake = {
        "name": f"query_aspect_{aspect:.6f}",
        "type": "aspect_query",
        "bounds": {"x": 0, "y": 0, "width": width, "height": height},
        "children": [],
    }
    return frame_embedding(fake)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _candidate_meta(source_file: Path, class_number: int, frame_index: int, node: dict[str, Any]) -> dict[str, Any]:
    b = _bounds(node)
    width = b.get("width", 0.0)
    height = b.get("height", 0.0)
    aspect = width / height if width > 0 and height > 0 else None
    return {
        "class_number": class_number,
        "source_file": str(source_file),
        "frame_index": frame_index,
        "id": node.get("id"),
        "name": node.get("name"),
        "type": node.get("type"),
        "bounds": node.get("bounds"),
        "aspect_ratio": aspect,
        "node_count": sum(1 for _ in _walk(node)),
        "leaf_count": _leaf_count(node),
    }


def _load_frames(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        return [raw]
    raise ValueError(f"{path} must contain a JSON object or array of objects")


def frames_from_raw(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        frames = [item for item in raw if isinstance(item, dict)]
    elif isinstance(raw, dict):
        frames = [raw]
    else:
        raise ValueError("Raw JSON must be an object or an array of objects")
    if not frames:
        raise ValueError("Raw JSON contains no top-level frame/object")
    return frames


def select_frame(raw: Any, frame_index: int = 0) -> dict[str, Any]:
    frames = frames_from_raw(raw)
    if frame_index < 0 or frame_index >= len(frames):
        raise ValueError(f"frame_index {frame_index} is out of range (0..{len(frames) - 1})")
    return frames[frame_index]


def build_class_index(class_number: int, raw_dir: Path = RAW_JSON_DIR, out_dir: Path = EMBEDDING_DIR) -> dict[str, Any]:
    if class_number not in VALID_CLASSES:
        raise ValueError(f"class_number must be one of {sorted(VALID_CLASSES)}")

    source = raw_dir / f"{class_number}.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing raw JSON file: {source}")

    frames = _load_frames(source)
    items: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        items.append(
            {
                "meta": _candidate_meta(source, class_number, idx, frame),
                "embedding": frame_embedding(frame),
            }
        )

    index = {
        "version": 1,
        "embedding": "local_hash_aspect_v1",
        "embedding_dim": EMBED_DIM,
        "class_number": class_number,
        "source_file": str(source),
        "count": len(items),
        "items": items,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"class_{class_number}.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    return index


def build_all_indexes(raw_dir: Path = RAW_JSON_DIR, out_dir: Path = EMBEDDING_DIR) -> list[dict[str, Any]]:
    return [build_class_index(n, raw_dir=raw_dir, out_dir=out_dir) for n in sorted(VALID_CLASSES)]


def load_class_index(class_number: int, out_dir: Path = EMBEDDING_DIR) -> dict[str, Any]:
    path = out_dir / f"class_{class_number}.json"
    if not path.exists():
        raise FileNotFoundError(f"Embedding index does not exist: {path}. Build it first.")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def search_index(class_number: int, aspect_ratio: str | float | int, top_k: int = 3, out_dir: Path = EMBEDDING_DIR) -> list[dict[str, Any]]:
    index = load_class_index(class_number, out_dir=out_dir)
    query = aspect_query_embedding(aspect_ratio)
    target_aspect = parse_aspect_ratio(aspect_ratio)

    scored: list[dict[str, Any]] = []
    for item in index.get("items", []):
        emb = item.get("embedding")
        meta = item.get("meta", {})
        if not isinstance(emb, list):
            continue
        aspect = meta.get("aspect_ratio")
        aspect_error = abs(math.log((aspect or 1.0) / target_aspect)) if aspect else float("inf")
        score = cosine(query, emb)
        # Aspect ratio is the user's retrieval input, so combine vector similarity
        # with an explicit aspect-distance penalty.
        final_score = score - min(aspect_error, 10.0) * 0.35
        row = {
            **meta,
            "score": final_score,
            "embedding_score": score,
            "aspect_error": aspect_error,
        }
        scored.append(row)

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]


def attach_full_json(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the full top-level Figma frame JSON for each retrieved candidate."""
    cache: dict[str, list[dict[str, Any]]] = {}
    out: list[dict[str, Any]] = []
    for row in candidates:
        source = str(row.get("source_file") or "")
        frame_index = row.get("frame_index")
        enriched = dict(row)
        if not source or not isinstance(frame_index, int):
            enriched["full_json"] = None
            out.append(enriched)
            continue

        if source not in cache:
            cache[source] = _load_frames(Path(source))
        frames = cache[source]
        if 0 <= frame_index < len(frames):
            enriched["full_json"] = frames[frame_index]
        else:
            enriched["full_json"] = None
        out.append(enriched)
    return out


def rerank_candidates_by_raw_similarity(raw_frame: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rerank already retrieved candidates by structural/text similarity to uploaded raw JSON."""
    raw_embedding = frame_embedding(raw_frame)
    ranked: list[dict[str, Any]] = []
    for row in candidates:
        full_json = row.get("full_json")
        if not isinstance(full_json, dict):
            continue
        candidate_embedding = frame_embedding(full_json)
        raw_similarity = cosine(raw_embedding, candidate_embedding)
        enriched = dict(row)
        enriched["raw_similarity"] = raw_similarity
        # Keep aspect/retrieval score as a secondary tie-breaker.
        enriched["selection_score"] = raw_similarity * 0.85 + float(row.get("score", 0.0)) * 0.15
        ranked.append(enriched)
    ranked.sort(key=lambda r: r["selection_score"], reverse=True)
    return ranked


def resize_figma_json_to_resolution(node: dict[str, Any], target_width: float, target_height: float) -> dict[str, Any]:
    """Deep-copy a Figma tree and scale every bounds object to target root resolution."""
    out = copy.deepcopy(node)
    root_bounds = _bounds(out)
    source_width = root_bounds.get("width") or target_width
    source_height = root_bounds.get("height") or target_height
    sx = target_width / source_width if source_width else 1.0
    sy = target_height / source_height if source_height else 1.0

    def scale(n: Any) -> None:
        if not isinstance(n, dict):
            return
        bounds = n.get("bounds")
        if isinstance(bounds, dict):
            for key, factor in (("x", sx), ("width", sx), ("y", sy), ("height", sy)):
                val = bounds.get(key)
                if isinstance(val, int | float):
                    bounds[key] = val * factor
        for child in n.get("children") or []:
            scale(child)

    scale(out)
    if isinstance(out.get("bounds"), dict):
        out["bounds"]["x"] = 0
        out["bounds"]["y"] = 0
        out["bounds"]["width"] = target_width
        out["bounds"]["height"] = target_height
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/query local Figma JSON embedding indexes.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="Build all class indexes from raw_jsons/{1,2,3,4}.json")

    q = sub.add_parser("query", help="Query top candidates by class number and aspect ratio")
    q.add_argument("class_number", type=int, choices=sorted(VALID_CLASSES))
    q.add_argument("aspect_ratio", help="Examples: 16:9, 1080x1920, 1.777")
    q.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()
    if args.cmd == "build":
        indexes = build_all_indexes()
        print(json.dumps([{"class_number": i["class_number"], "count": i["count"]} for i in indexes], ensure_ascii=False, indent=2))
    elif args.cmd == "query":
        print(json.dumps(search_index(args.class_number, args.aspect_ratio, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
