"""RALF dataset and collate with retrieved examples."""

from __future__ import annotations

import math
import gzip
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from layout_training.pairs import read_jsonl

RALF_NUMERIC_FEATURE_DIM = 16

try:
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    TorchDataset = object


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for RALF dataset.") from exc
    return torch


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def build_role_vocab(pairs: list[dict[str, Any]]) -> dict[str, int]:
    roles = set()
    types = set()
    for pair in pairs:
        for tok in pair.get("tokens") or []:
            src = tok.get("source") or {}
            roles.add(str(tok.get("train_role") or "unknown_group"))
            types.add(str(src.get("type") or "unknown"))
        for ret in pair.get("retrieved") or []:
            for tok in ret.get("tokens") or []:
                roles.add(str(tok.get("train_role") or "unknown_group"))
                types.add(str((tok.get("features") or {}).get("type") or "unknown"))
    return (
        {"<pad>": 0, **{r: i + 1 for i, r in enumerate(sorted(roles))}},
        {"<pad>": 0, **{t: i + 1 for i, t in enumerate(sorted(types))}},
    )


def _torch_load(path: str | Path):
    torch = _torch()
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        try:
            return torch.load(fh, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(fh, map_location="cpu")


def _small_canvas(canvas: dict[str, Any] | None) -> dict[str, float]:
    canvas = canvas or {}
    width = _safe_float(canvas.get("width"), 0.0)
    height = _safe_float(canvas.get("height"), 0.0)
    aspect = _safe_float(canvas.get("aspect"), 0.0)
    if aspect <= 0.0:
        aspect = width / max(height, 1e-6) if width > 0 and height > 0 else 1.0
    return {"width": width, "height": height, "aspect": aspect}


def preprocess_ralf_pair_for_cache(pair: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw RALF pair into the compact tensor-cache schema."""
    src_num: list[list[float]] = []
    src_roles: list[str] = []
    src_types: list[str] = []
    src_center: list[list[float]] = []
    target: list[list[float]] = []
    target_mask: list[bool] = []

    source_canvas = _small_canvas(pair.get("source_canvas"))
    target_canvas = _small_canvas(pair.get("target_canvas"))
    for tok in pair.get("tokens") or []:
        source_token = tok.get("source") or tok
        center = (source_token.get("center_size_norm") or [0.5, 0.5, 0.1, 0.1])[:4]
        tgt = (tok.get("target_center_size_norm") or [0.0, 0.0, 0.0, 0.0])[:4]
        src_num.append(
            build_ralf_numeric_features(
                source_token,
                canvas=source_canvas,
                target_canvas=target_canvas,
                retrieval_score=0.0,
            )
        )
        src_roles.append(str(tok.get("train_role") or "unknown_group"))
        src_types.append(str(source_token.get("type") or "unknown"))
        src_center.append([_safe_float(x, 0.0) for x in center])
        target.append([_safe_float(x, 0.0) for x in tgt])
        target_mask.append(bool(tok.get("has_target", True)))

    ret_num: list[list[list[float]]] = []
    ret_roles: list[list[str]] = []
    ret_types: list[list[str]] = []
    ret_scores: list[float] = []
    ret_record_ids: list[str] = []
    for ret in pair.get("retrieved") or []:
        score = _safe_float(ret.get("score"), 0.0)
        ret_canvas = _small_canvas(ret.get("canvas"))
        nums: list[list[float]] = []
        roles: list[str] = []
        types: list[str] = []
        for tok in ret.get("tokens") or []:
            nums.append(
                build_ralf_numeric_features(
                    tok,
                    canvas=ret_canvas,
                    target_canvas=target_canvas,
                    retrieval_score=score,
                )
            )
            roles.append(str(tok.get("train_role") or "unknown_group"))
            types.append(str((tok.get("features") or {}).get("type") or "unknown"))
        ret_num.append(nums)
        ret_roles.append(roles)
        ret_types.append(types)
        ret_scores.append(score)
        ret_record_ids.append(str(ret.get("record_id") or ""))

    return {
        "pair_id": str(pair.get("pair_id") or ""),
        "family_id": str(pair.get("family_id") or ""),
        "source_id": str(pair.get("source_id") or ""),
        "target_id": str(pair.get("target_id") or ""),
        "source_canvas": source_canvas,
        "target_canvas": target_canvas,
        "src_num": src_num,
        "src_roles": src_roles,
        "src_types": src_types,
        "src_center_size_norm": src_center,
        "target": target,
        "target_mask": target_mask,
        "ret_num": ret_num,
        "ret_roles": ret_roles,
        "ret_types": ret_types,
        "ret_scores": ret_scores,
        "ret_record_ids": ret_record_ids,
    }


def build_ralf_numeric_features(
    token: dict[str, Any],
    canvas: dict[str, Any] | None = None,
    target_canvas: dict[str, Any] | None = None,
    retrieval_score: float = 0.0,
) -> list[float]:
    meta = token.get("features") if isinstance(token.get("features"), dict) else token
    center = token.get("center_size_norm")
    if not isinstance(center, list):
        center = (token.get("source") or {}).get("center_size_norm")
    if not isinstance(center, list):
        center = [0.5, 0.5, 0.1, 0.1]
    center = [
        _safe_float(center[0] if len(center) > 0 else 0.5, 0.5),
        _safe_float(center[1] if len(center) > 1 else 0.5, 0.5),
        _safe_float(center[2] if len(center) > 2 else 0.1, 0.1),
        _safe_float(center[3] if len(center) > 3 else 0.1, 0.1),
    ]

    area_ratio = _safe_float(meta.get("area_ratio", token.get("area_ratio", 0.0)), 0.0)
    has_text = 1.0 if bool(meta.get("has_text", token.get("has_text", False))) else 0.0
    has_image = 1.0 if bool(meta.get("has_image", token.get("has_image", False))) else 0.0
    has_gradient = 1.0 if bool(meta.get("has_gradient", token.get("has_gradient", False))) else 0.0
    has_star = 1.0 if bool(meta.get("has_star", token.get("has_star", False))) else 0.0
    discount_text = 1.0 if bool(meta.get("discount_text", token.get("discount_text", False))) else 0.0
    rotation_deg = _safe_float(meta.get("rotation_deg", token.get("rotation_deg", 0.0)), 0.0)

    if "is_rotated" in meta:
        is_rotated = 1.0 if bool(meta.get("is_rotated")) else 0.0
    elif "is_rotated" in token:
        is_rotated = 1.0 if bool(token.get("is_rotated")) else 0.0
    else:
        is_rotated = 1.0 if abs(rotation_deg) > 0.01 else 0.0

    instance_count = _safe_float(meta.get("instance_count", token.get("instance_count", 1.0)), 1.0)
    instance_count_norm = min(max(instance_count, 0.0), 10.0) / 10.0
    descendant_count = _safe_float(meta.get("descendant_count", token.get("descendant_count", 0.0)), 0.0)
    descendant_count_log_norm = math.log1p(max(descendant_count, 0.0)) / 10.0

    canvas = canvas or {}
    cw = _safe_float(canvas.get("width"), 0.0)
    ch = _safe_float(canvas.get("height"), 0.0)
    canvas_aspect = _safe_float(canvas.get("aspect"), 0.0)
    if canvas_aspect <= 0:
        canvas_aspect = cw / max(ch, 1e-6) if cw > 0 and ch > 0 else 1.0

    features = [
        center[0],
        center[1],
        center[2],
        center[3],
        area_ratio,
        has_text,
        has_image,
        has_gradient,
        has_star,
        discount_text,
        is_rotated,
        rotation_deg / 180.0,
        instance_count_norm,
        descendant_count_log_norm,
        canvas_aspect,
        _safe_float(retrieval_score, 0.0),
    ]
    clean = [_safe_float(x, 0.0) for x in features]
    if len(clean) != RALF_NUMERIC_FEATURE_DIM:
        raise AssertionError(f"RALF feature dim mismatch: got {len(clean)} expected {RALF_NUMERIC_FEATURE_DIM}")
    return clean


class RalfLayoutPairDataset(TorchDataset):
    def __init__(self, path: str, role_vocab: dict[str, int] | None = None, type_vocab: dict[str, int] | None = None):
        self.pairs = read_jsonl(path)
        if role_vocab is None or type_vocab is None:
            rv, tv = build_role_vocab(self.pairs)
            self.role_vocab = role_vocab or rv
            self.type_vocab = type_vocab or tv
        else:
            self.role_vocab = role_vocab
            self.type_vocab = type_vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.pairs[idx]
        src_rows = []
        for tok in pair.get("tokens") or []:
            source_token = tok.get("source") or tok
            src_rows.append(
                {
                    "num": build_ralf_numeric_features(
                        source_token,
                        canvas=pair.get("source_canvas"),
                        target_canvas=pair.get("target_canvas"),
                        retrieval_score=0.0,
                    ),
                    "center": [float(x) for x in (source_token.get("center_size_norm") or [0.5, 0.5, 0.1, 0.1])[:4]],
                    "role_id": self.role_vocab.get(str(tok.get("train_role") or "unknown_group"), 0),
                    "type_id": self.type_vocab.get(str(source_token.get("type") or "unknown"), 0),
                    "target": [float(x) for x in (tok.get("target_center_size_norm") or [0.0, 0.0, 0.0, 0.0])[:4]],
                    "has_target": bool(tok.get("has_target", True)),
                    "train_role": str(tok.get("train_role") or "unknown_group"),
                    "token_id": str(tok.get("token_id") or ""),
                }
            )
        ret_rows = []
        for ret in pair.get("retrieved") or []:
            toks = []
            for tok in ret.get("tokens") or []:
                toks.append(
                    {
                        "num": build_ralf_numeric_features(
                            tok,
                            canvas=ret.get("canvas"),
                            target_canvas=pair.get("target_canvas"),
                            retrieval_score=_safe_float(ret.get("score"), 0.0),
                        ),
                        "role_id": self.role_vocab.get(str(tok.get("train_role") or "unknown_group"), 0),
                        "type_id": self.type_vocab.get(str((tok.get("features") or {}).get("type") or "unknown"), 0),
                        "token_id": str(tok.get("token_id") or ""),
                    }
                )
            ret_rows.append({"score": float(ret.get("score") or 0.0), "tokens": toks})
        return {"pair_id": pair.get("pair_id"), "source_tokens": src_rows, "retrieved": ret_rows}


class CachedRalfLayoutPairDataset(TorchDataset):
    def __init__(
        self,
        cache_dir: str,
        max_open_shards: int = 2,
        load_all: bool = False,
        role_vocab: dict[str, int] | None = None,
        type_vocab: dict[str, int] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing RALF cache manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.shard_meta = list(self.manifest.get("shards") or [])
        self.max_open_shards = max(1, int(max_open_shards))
        self.load_all = bool(load_all)
        self.role_vocab = role_vocab
        self.type_vocab = type_vocab
        self.index: list[tuple[int, int]] = []
        for shard_id, meta in enumerate(self.shard_meta):
            for local_idx in range(int(meta.get("samples") or 0)):
                self.index.append((shard_id, local_idx))
        self._shard_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        if self.load_all:
            for shard_id in range(len(self.shard_meta)):
                self._shard_cache[shard_id] = self._load_shard(shard_id)

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, shard_id: int) -> list[dict[str, Any]]:
        path = self.cache_dir / str(self.shard_meta[shard_id]["file"])
        data = _torch_load(path)
        if not isinstance(data, list):
            raise ValueError(f"RALF cache shard must contain a list: {path}")
        return data

    def _get_shard(self, shard_id: int) -> list[dict[str, Any]]:
        cached = self._shard_cache.get(shard_id)
        if cached is not None:
            self._shard_cache.move_to_end(shard_id)
            return cached
        shard = self._load_shard(shard_id)
        self._shard_cache[shard_id] = shard
        self._shard_cache.move_to_end(shard_id)
        while not self.load_all and len(self._shard_cache) > self.max_open_shards:
            self._shard_cache.popitem(last=False)
        return shard

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_id, local_idx = self.index[idx]
        item = dict(self._get_shard(shard_id)[local_idx])
        if self.role_vocab is not None:
            item["_role_vocab"] = self.role_vocab
        if self.type_vocab is not None:
            item["_type_vocab"] = self.type_vocab
        return item


def build_vocabs_from_dataset(dataset) -> tuple[dict[str, int], dict[str, int]]:
    roles: set[str] = set()
    types: set[str] = set()
    if hasattr(dataset, "pairs"):
        return build_role_vocab(dataset.pairs)
    if isinstance(dataset, CachedRalfLayoutPairDataset):
        for shard_id in range(len(dataset.shard_meta)):
            for sample in dataset._get_shard(shard_id):
                roles.update(str(x) for x in sample.get("src_roles") or [])
                types.update(str(x) for x in sample.get("src_types") or [])
                for row in sample.get("ret_roles") or []:
                    roles.update(str(x) for x in row)
                for row in sample.get("ret_types") or []:
                    types.update(str(x) for x in row)
        if not dataset.load_all:
            dataset._shard_cache.clear()
        return (
            {"<pad>": 0, **{r: i + 1 for i, r in enumerate(sorted(roles))}},
            {"<pad>": 0, **{t: i + 1 for i, t in enumerate(sorted(types))}},
        )
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if "src_roles" in sample:
            roles.update(str(x) for x in sample.get("src_roles") or [])
            types.update(str(x) for x in sample.get("src_types") or [])
            for row in sample.get("ret_roles") or []:
                roles.update(str(x) for x in row)
            for row in sample.get("ret_types") or []:
                types.update(str(x) for x in row)
    return (
        {"<pad>": 0, **{r: i + 1 for i, r in enumerate(sorted(roles))}},
        {"<pad>": 0, **{t: i + 1 for i, t in enumerate(sorted(types))}},
    )


def ralf_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch = _torch()
    bsz = len(batch)
    cached_mode = bool(batch and "src_num" in batch[0])
    if cached_mode:
        max_t = max((len(x.get("src_num") or []) for x in batch), default=0)
        max_k = max((len(x.get("ret_num") or []) for x in batch), default=0)
        max_r = max((max((len(tokens) for tokens in (x.get("ret_num") or [])), default=0) for x in batch), default=0)
        role_vocab = batch[0].get("_role_vocab") or {"<pad>": 0}
        type_vocab = batch[0].get("_type_vocab") or {"<pad>": 0}
    else:
        max_t = max((len(x["source_tokens"]) for x in batch), default=0)
        max_k = max((len(x["retrieved"]) for x in batch), default=0)
        max_r = max((max((len(k["tokens"]) for k in x["retrieved"]), default=0) for x in batch), default=0)
        role_vocab = {"<pad>": 0}
        type_vocab = {"<pad>": 0}

    src_num = torch.zeros((bsz, max_t, RALF_NUMERIC_FEATURE_DIM), dtype=torch.float32)
    src_center = torch.zeros((bsz, max_t, 4), dtype=torch.float32)
    src_role_ids = torch.zeros((bsz, max_t), dtype=torch.long)
    src_type_ids = torch.zeros((bsz, max_t), dtype=torch.long)
    src_mask = torch.zeros((bsz, max_t), dtype=torch.bool)
    target = torch.zeros((bsz, max_t, 4), dtype=torch.float32)
    target_mask = torch.zeros((bsz, max_t), dtype=torch.bool)

    ret_num = torch.zeros((bsz, max_k, max_r, RALF_NUMERIC_FEATURE_DIM), dtype=torch.float32)
    ret_role_ids = torch.zeros((bsz, max_k, max_r), dtype=torch.long)
    ret_type_ids = torch.zeros((bsz, max_k, max_r), dtype=torch.long)
    ret_mask = torch.zeros((bsz, max_k, max_r), dtype=torch.bool)
    ret_scores = torch.zeros((bsz, max_k), dtype=torch.float32)
    train_roles: list[list[str]] = []
    token_ids: list[list[str]] = []

    for bi, item in enumerate(batch):
        roles = []
        ids = []
        if cached_mode:
            for ti, nums in enumerate(item.get("src_num") or []):
                if len(nums) != RALF_NUMERIC_FEATURE_DIM:
                    raise ValueError(
                        f"Bad RALF feature length: pair_id={item.get('pair_id')}, "
                        f"len={len(nums)}, expected={RALF_NUMERIC_FEATURE_DIM}"
                    )
                src_num[bi, ti] = torch.tensor(nums, dtype=torch.float32)
                src_center[bi, ti] = torch.tensor((item.get("src_center_size_norm") or [])[ti], dtype=torch.float32)
                role = str((item.get("src_roles") or [])[ti])
                typ = str((item.get("src_types") or [])[ti])
                src_role_ids[bi, ti] = int(role_vocab.get(role, 0))
                src_type_ids[bi, ti] = int(type_vocab.get(typ, 0))
                src_mask[bi, ti] = True
                target[bi, ti] = torch.tensor((item.get("target") or [])[ti], dtype=torch.float32)
                target_mask[bi, ti] = bool((item.get("target_mask") or [])[ti])
                roles.append(role)
                ids.append("")
            train_roles.append(roles)
            token_ids.append(ids)

            for ki, rows in enumerate(item.get("ret_num") or []):
                ret_scores[bi, ki] = float((item.get("ret_scores") or [0.0] * (ki + 1))[ki])
                row_roles = (item.get("ret_roles") or [])[ki]
                row_types = (item.get("ret_types") or [])[ki]
                for ri, nums in enumerate(rows):
                    if len(nums) != RALF_NUMERIC_FEATURE_DIM:
                        raise ValueError(
                            f"Bad RALF feature length: pair_id={item.get('pair_id')}, "
                            f"len={len(nums)}, expected={RALF_NUMERIC_FEATURE_DIM}"
                        )
                    ret_num[bi, ki, ri] = torch.tensor(nums, dtype=torch.float32)
                    ret_role_ids[bi, ki, ri] = int(role_vocab.get(str(row_roles[ri]), 0))
                    ret_type_ids[bi, ki, ri] = int(type_vocab.get(str(row_types[ri]), 0))
                    ret_mask[bi, ki, ri] = True
            continue

        for ti, tok in enumerate(item["source_tokens"]):
            if len(tok["num"]) != RALF_NUMERIC_FEATURE_DIM:
                raise ValueError(
                    f"Bad RALF feature length: pair_id={item.get('pair_id')}, token_id={tok.get('token_id','')}, "
                    f"len={len(tok['num'])}, expected={RALF_NUMERIC_FEATURE_DIM}"
                )
            src_num[bi, ti] = torch.tensor(tok["num"], dtype=torch.float32)
            src_center[bi, ti] = torch.tensor(tok["center"], dtype=torch.float32)
            src_role_ids[bi, ti] = int(tok["role_id"])
            src_type_ids[bi, ti] = int(tok["type_id"])
            src_mask[bi, ti] = True
            target[bi, ti] = torch.tensor(tok["target"], dtype=torch.float32)
            target_mask[bi, ti] = bool(tok["has_target"])
            roles.append(tok["train_role"])
            ids.append(tok["token_id"])
        train_roles.append(roles)
        token_ids.append(ids)

        for ki, ret in enumerate(item["retrieved"]):
            ret_scores[bi, ki] = float(ret["score"])
            for ri, tok in enumerate(ret["tokens"]):
                if len(tok["num"]) != RALF_NUMERIC_FEATURE_DIM:
                    raise ValueError(
                        f"Bad RALF feature length: pair_id={item.get('pair_id')}, token_id={tok.get('token_id','')}, "
                        f"len={len(tok['num'])}, expected={RALF_NUMERIC_FEATURE_DIM}"
                    )
                ret_num[bi, ki, ri] = torch.tensor(tok["num"], dtype=torch.float32)
                ret_role_ids[bi, ki, ri] = int(tok["role_id"])
                ret_type_ids[bi, ki, ri] = int(tok["type_id"])
                ret_mask[bi, ki, ri] = True

    return {
        "src_num": src_num,
        "src_center": src_center,
        "src_role_ids": src_role_ids,
        "src_type_ids": src_type_ids,
        "src_mask": src_mask,
        "ret_num": ret_num,
        "ret_role_ids": ret_role_ids,
        "ret_type_ids": ret_type_ids,
        "ret_mask": ret_mask,
        "ret_scores": ret_scores,
        "target": target,
        "target_mask": target_mask,
        "train_roles": train_roles,
        "token_ids": token_ids,
    }
