"""Build parent, child, and floating rich semantic pair datasets."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch

from .rich_utils import (
    flatten_role_nodes,
    get_canvas_size,
    load_frames,
    node_flags,
    normalized_bbox,
    relative_bbox,
    safe_float,
)
from .schema import (
    ALIGN_H_TO_ID,
    ALIGN_V_TO_ID,
    CHILD_PARENT,
    CHILD_ROLES,
    DEFAULT_NODE_TYPES,
    FLOATING_ROLES,
    PARENT_ROLES,
    UNKNOWN_NODE_TYPE,
    orientation_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    families = load_families(args.input_dir)
    node_type_to_id = build_node_type_vocab(families)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("parent", PARENT_ROLES, "target_bbox", args.out_dir / "parent_layout_pairs.pt"),
        ("child", CHILD_ROLES, "target_relative_bbox", args.out_dir / "child_layout_pairs.pt"),
        ("floating", FLOATING_ROLES, "target_bbox", args.out_dir / "floating_layout_pairs.pt"),
    ]
    for dataset_type, roles, target_key, out_path in specs:
        samples = build_role_samples(families, roles, dataset_type, node_type_to_id)
        dataset = split_samples(samples, args.seed, args.train_ratio, args.val_ratio)
        dataset.update(
            {
                "dataset_type": dataset_type,
                "roles": roles,
                "role_to_id": {role: idx for idx, role in enumerate(roles)},
                "node_type_to_id": node_type_to_id,
                "target_key": target_key,
            }
        )
        torch.save(dataset, out_path)
        print(f"{dataset_type}: samples={len(samples)} train={len(dataset['train']['role_id'])} val={len(dataset['val']['role_id'])} test={len(dataset['test']['role_id'])}")
        print(f"Wrote: {out_path}")


def load_families(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("*_clean_fixed_semantic_rich.json"))
    if not paths:
        raise FileNotFoundError(f"no *_clean_fixed_semantic_rich.json files found in {input_dir}")
    return [{"path": path, "frames": load_frames(path)} for path in paths]


def build_node_type_vocab(families: list[dict[str, Any]]) -> dict[str, int]:
    node_types = set(DEFAULT_NODE_TYPES)
    for family in families:
        for frame in family["frames"]:
            for node in flatten_role_nodes(frame).values():
                node_types.add(str(node.get("type") or UNKNOWN_NODE_TYPE).lower())
    ordered = [UNKNOWN_NODE_TYPE] + sorted(t for t in node_types if t != UNKNOWN_NODE_TYPE)
    return {node_type: idx for idx, node_type in enumerate(ordered)}


def build_role_samples(
    families: list[dict[str, Any]],
    roles: list[str],
    dataset_type: str,
    node_type_to_id: dict[str, int],
) -> list[dict[str, torch.Tensor | str]]:
    samples: list[dict[str, torch.Tensor | str]] = []
    role_to_id = {role: idx for idx, role in enumerate(roles)}
    for family in families:
        prepared = [prepare_frame(frame, roles, dataset_type, node_type_to_id, role_to_id) for frame in family["frames"]]
        for source in prepared:
            for target in prepared:
                if source["frame_id"] == target["frame_id"]:
                    continue
                for role in roles:
                    if role not in source["roles"] or role not in target["roles"]:
                        continue
                    source_row = source["roles"][role]
                    target_row = target["roles"][role]
                    row = {
                        "source_frame_id": source["frame_id"],
                        "target_frame_id": target["frame_id"],
                        "role": role,
                        "source_canvas": source["canvas"],
                        "target_canvas": target["canvas"],
                        "source_orientation": source["orientation"],
                        "target_orientation": target["orientation"],
                        "role_id": source_row["role_id"],
                        "node_type_id": source_row["node_type_id"],
                        "source_bbox": source_row["source_bbox"],
                        "source_relative_bbox": source_row["source_relative_bbox"],
                        "flags": source_row["flags"],
                        "font_size": source_row["font_size"],
                        "align_h": source_row["align_h"],
                        "align_v": source_row["align_v"],
                        "target_visibility": target_row["visibility"],
                    }
                    if dataset_type == "child":
                        row["target_relative_bbox"] = target_row["target_relative_bbox"]
                        row["target_font_size"] = target_row["font_size"]
                        row["target_align_h"] = target_row["align_h"]
                        row["target_align_v"] = target_row["align_v"]
                    else:
                        row["target_bbox"] = target_row["target_bbox"]
                    samples.append(row)
    return samples


def prepare_frame(
    frame: dict[str, Any],
    roles: list[str],
    dataset_type: str,
    node_type_to_id: dict[str, int],
    role_to_id: dict[str, int],
) -> dict[str, Any]:
    canvas_w, canvas_h = get_canvas_size(frame)
    canvas_area = max(canvas_w * canvas_h, 1.0)
    nodes = flatten_role_nodes(frame)
    rows: dict[str, dict[str, torch.Tensor]] = {}
    for role in roles:
        node = nodes.get(role)
        if node is None:
            continue
        parent = nodes.get(CHILD_PARENT.get(role, ""))
        rel = relative_bbox(node, parent) if parent is not None else [0.0, 0.0, 0.0, 0.0]
        node_type = str(node.get("type") or UNKNOWN_NODE_TYPE).lower()
        row = {
            "role_id": torch.tensor(role_to_id[role], dtype=torch.long),
            "node_type_id": torch.tensor(node_type_to_id.get(node_type, node_type_to_id[UNKNOWN_NODE_TYPE]), dtype=torch.long),
            "source_bbox": torch.tensor(normalized_bbox(node, canvas_w, canvas_h), dtype=torch.float32),
            "source_relative_bbox": torch.tensor(rel, dtype=torch.float32),
            "target_bbox": torch.tensor(normalized_bbox(node, canvas_w, canvas_h), dtype=torch.float32),
            "target_relative_bbox": torch.tensor(rel, dtype=torch.float32),
            "flags": torch.tensor(node_flags(node), dtype=torch.float32),
            "font_size": torch.tensor(safe_float(node.get("fontSize")) / (canvas_area**0.5), dtype=torch.float32),
            "align_h": torch.tensor(ALIGN_H_TO_ID.get(str(node.get("textAlignHorizontal") or "UNKNOWN").upper(), ALIGN_H_TO_ID["UNKNOWN"]), dtype=torch.long),
            "align_v": torch.tensor(ALIGN_V_TO_ID.get(str(node.get("textAlignVertical") or "UNKNOWN").upper(), ALIGN_V_TO_ID["UNKNOWN"]), dtype=torch.long),
            "visibility": torch.tensor(1.0 if node.get("visible", True) else 0.0, dtype=torch.float32),
        }
        rows[role] = row
    return {
        "frame_id": str(frame.get("id") or frame.get("name") or id(frame)),
        "canvas": torch.tensor([canvas_w, canvas_h, canvas_w / canvas_h], dtype=torch.float32),
        "orientation": torch.tensor(orientation_id(canvas_w, canvas_h), dtype=torch.long),
        "roles": rows,
    }


def split_samples(samples: list[dict[str, Any]], seed: int, train_ratio: float, val_ratio: float) -> dict[str, Any]:
    rows = list(samples)
    random.Random(seed).shuffle(rows)
    train_cut = int(len(rows) * train_ratio)
    val_cut = train_cut + int(len(rows) * val_ratio)
    if len(rows) >= 3:
        train_cut = max(1, min(train_cut, len(rows) - 2))
        val_cut = max(train_cut + 1, min(val_cut, len(rows) - 1))
    return {
        "train": tensorize(rows[:train_cut]),
        "val": tensorize(rows[train_cut:val_cut]),
        "test": tensorize(rows[val_cut:]),
    }


def tensorize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "source_canvas",
        "target_canvas",
        "source_orientation",
        "target_orientation",
        "role_id",
        "node_type_id",
        "source_bbox",
        "source_relative_bbox",
        "flags",
        "font_size",
        "align_h",
        "align_v",
        "target_visibility",
    ]
    optional_keys = ["target_bbox", "target_relative_bbox", "target_font_size", "target_align_h", "target_align_v"]
    if not rows:
        return empty_split()
    out: dict[str, Any] = {key: torch.stack([row[key] for row in rows]) for key in tensor_keys}
    for key in optional_keys:
        if key in rows[0]:
            out[key] = torch.stack([row[key] for row in rows])
    out["source_frame_id"] = [str(row["source_frame_id"]) for row in rows]
    out["target_frame_id"] = [str(row["target_frame_id"]) for row in rows]
    out["role"] = [str(row["role"]) for row in rows]
    return out


def empty_split() -> dict[str, Any]:
    return {
        "source_canvas": torch.empty((0, 3), dtype=torch.float32),
        "target_canvas": torch.empty((0, 3), dtype=torch.float32),
        "source_orientation": torch.empty((0,), dtype=torch.long),
        "target_orientation": torch.empty((0,), dtype=torch.long),
        "role_id": torch.empty((0,), dtype=torch.long),
        "node_type_id": torch.empty((0,), dtype=torch.long),
        "source_bbox": torch.empty((0, 4), dtype=torch.float32),
        "source_relative_bbox": torch.empty((0, 4), dtype=torch.float32),
        "flags": torch.empty((0, 5), dtype=torch.float32),
        "font_size": torch.empty((0,), dtype=torch.float32),
        "align_h": torch.empty((0,), dtype=torch.long),
        "align_v": torch.empty((0,), dtype=torch.long),
        "target_visibility": torch.empty((0,), dtype=torch.float32),
        "source_frame_id": [],
        "target_frame_id": [],
        "role": [],
    }


if __name__ == "__main__":
    main()

