"""Evaluate a trained GNN layout predictor on held-out families."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from .dataset import GNNLayoutDataset
from .graph_builder import FEATURE_NAMES
from .model import GNNLayoutPredictor
from .roles import ROLES
from .train import evaluate_loader, split_rows_by_family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = GNNLayoutDataset._read_rows(args.pairs)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    test_families = set(config.get("test_families") or [])
    if test_families:
        eval_rows = [row for row in rows if str(row.get("family_key") or "unknown") in test_families]
        split_note = f"checkpoint test split ({len(test_families)} families)"
    else:
        splits = split_rows_by_family(rows, seed=args.seed)
        eval_rows = splits["test"] or rows
        split_note = "recomputed held-out split" if splits["test"] else "all rows (no held-out split available)"

    if not eval_rows:
        raise SystemExit("No rows available for evaluation")

    hidden = int(config.get("hidden", 128))
    dropout = float(config.get("dropout", 0.15))
    in_channels = int(config.get("in_channels", len(FEATURE_NAMES)))
    model = GNNLayoutPredictor(in_channels=in_channels, hidden=hidden, dropout=dropout)
    model.load_state_dict(checkpoint["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(GNNLayoutDataset(args.pairs, eval_rows), batch_size=args.batch_size, shuffle=False)
    loss, metrics = evaluate_loader(model, loader, nn.SmoothL1Loss(reduction="none"), device)

    print(f"Evaluation split: {split_note}")
    print(f"Total samples: {len(eval_rows)}")
    print(f"Loss: {loss:.6f}")
    print(f"Mean L1: {metrics['mean_l1']:.6f}")
    print("Role-wise L1:")
    for role in ROLES:
        print(f"  {role}: {metrics['role_l1'][role]:.6f}")
    print("Role-wise IoU:")
    for role in ROLES:
        print(f"  {role}: {metrics['role_iou'][role]:.6f}")
    print("Missing role counts:")
    for role in ROLES:
        print(f"  {role}: {metrics['missing_counts'][role]}")


if __name__ == "__main__":
    main()
