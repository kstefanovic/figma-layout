"""Build within-family semantic layout transformation pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from .extract import flatten_semantic_nodes, get_bbox_norm, get_canvas_size
from .roles import NUM_ROLES, ROLE_TO_ID, TRAIN_ROLES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    families = load_families(args.input_dir)
    samples = build_samples(families)
    dataset = split_samples(samples, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    dataset["roles"] = TRAIN_ROLES
    dataset["num_roles"] = NUM_ROLES
    dataset["family_files"] = {family["family_id"]: family["path"].name for family in families}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.out)
    print(f"Families: {len(families)}")
    print(f"Frames: {sum(len(f['frames']) for f in families)}")
    print(f"Pairs: {len(samples)}")
    for split in ("train", "val", "test"):
        print(f"{split}: {len(dataset[split]['source_frame_id'])}")
    print(f"Wrote: {args.out}")


def load_families(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("*_clean_fixed_semantic.json"))
    if not paths:
        raise FileNotFoundError(f"no *_clean_fixed_semantic.json files found in {input_dir}")

    families: list[dict[str, Any]] = []
    for path in paths:
        data = _load_json(path)
        frames = _coerce_frames(data)
        family_id = path.stem
        families.append({"family_id": family_id, "path": path, "frames": frames})
    return families


def build_samples(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for family in families:
        prepared = [_prepare_frame(frame) for frame in family["frames"]]
        for source in prepared:
            for target in prepared:
                if source["frame_id"] == target["frame_id"]:
                    continue
                role_mask = source["mask"] * target["mask"]
                samples.append(
                    {
                        "family_id": family["family_id"],
                        "source_frame_id": source["frame_id"],
                        "target_frame_id": target["frame_id"],
                        "source_size": source["size"],
                        "target_size": target["size"],
                        "source_bboxes": source["bboxes"],
                        "target_bboxes": target["bboxes"],
                        "role_mask": role_mask,
                        "source_json": source["json"],
                        "target_json": target["json"],
                    }
                )
    return samples


def split_samples(
    samples: list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, Any]:
    target_ids = sorted({sample["target_frame_id"] for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(target_ids)
    train_cut = int(len(target_ids) * train_ratio)
    val_cut = train_cut + int(len(target_ids) * val_ratio)
    if len(target_ids) >= 3:
        train_cut = max(1, min(train_cut, len(target_ids) - 2))
        val_cut = max(train_cut + 1, min(val_cut, len(target_ids) - 1))

    split_ids = {
        "train": set(target_ids[:train_cut]),
        "val": set(target_ids[train_cut:val_cut]),
        "test": set(target_ids[val_cut:]),
    }
    out = {}
    for split, ids in split_ids.items():
        rows = [sample for sample in samples if sample["target_frame_id"] in ids]
        out[split] = _tensorize(rows)
    return out


def _prepare_frame(frame: dict[str, Any]) -> dict[str, Any]:
    canvas_w, canvas_h = get_canvas_size(frame)
    bboxes = torch.zeros((NUM_ROLES, 4), dtype=torch.float32)
    mask = torch.zeros((NUM_ROLES,), dtype=torch.float32)
    nodes = flatten_semantic_nodes(frame)
    for role in TRAIN_ROLES:
        node = nodes.get(role)
        if node is None:
            continue
        role_id = ROLE_TO_ID[role]
        bboxes[role_id] = torch.tensor(get_bbox_norm(node, canvas_w, canvas_h), dtype=torch.float32)
        mask[role_id] = 1.0
    return {
        "frame_id": _frame_id(frame),
        "size": torch.tensor([canvas_w, canvas_h], dtype=torch.float32),
        "bboxes": bboxes,
        "mask": mask,
        "json": frame,
    }


def _tensorize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "source_size": torch.empty((0, 2), dtype=torch.float32),
            "target_size": torch.empty((0, 2), dtype=torch.float32),
            "source_bboxes": torch.empty((0, NUM_ROLES, 4), dtype=torch.float32),
            "target_bboxes": torch.empty((0, NUM_ROLES, 4), dtype=torch.float32),
            "role_mask": torch.empty((0, NUM_ROLES), dtype=torch.float32),
            "family_id_list": [],
            "source_frame_id": [],
            "target_frame_id": [],
            "source_json": [],
            "target_json": [],
        }
    return {
        "source_size": torch.stack([row["source_size"] for row in rows]).float(),
        "target_size": torch.stack([row["target_size"] for row in rows]).float(),
        "source_bboxes": torch.stack([row["source_bboxes"] for row in rows]).float(),
        "target_bboxes": torch.stack([row["target_bboxes"] for row in rows]).float(),
        "role_mask": torch.stack([row["role_mask"] for row in rows]).float(),
        "family_id_list": [row["family_id"] for row in rows],
        "source_frame_id": [row["source_frame_id"] for row in rows],
        "target_frame_id": [row["target_frame_id"] for row in rows],
        "source_json": [row["source_json"] for row in rows],
        "target_json": [row["target_json"] for row in rows],
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_frames(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("frames", "banners", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    raise ValueError("clean semantic file must contain a JSON object or list")


def _frame_id(frame: dict[str, Any]) -> str:
    return str(frame.get("id") or frame.get("name") or id(frame))


if __name__ == "__main__":
    main()
