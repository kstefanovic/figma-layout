"""Train the GNN layout prior predictor."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .dataset import GNNLayoutDataset
from .graph_builder import FEATURE_NAMES
from .model import GNNLayoutPredictor
from .roles import IDX_TO_ROLE, NUM_ROLES, ROLES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--out", type=Path, default=Path("gnn_layout/data/checkpoints/gnn_layout.pt"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = GNNLayoutDataset._read_rows(args.pairs)
    if not rows:
        raise SystemExit(f"No training pairs found in {args.pairs}")

    set_seed(args.seed)
    splits = split_rows_by_family(rows, seed=args.seed)
    train_rows = splits["train"]
    val_rows = splits["val"] or train_rows
    test_rows = splits["test"]
    print_split_summary(splits)

    train_loader = DataLoader(GNNLayoutDataset(args.pairs, train_rows), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(GNNLayoutDataset(args.pairs, val_rows), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNLayoutPredictor(
        in_channels=len(FEATURE_NAMES),
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss(reduction="none")

    best_val = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "pairs": str(args.pairs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "lr": args.lr,
        "dropout": args.dropout,
        "seed": args.seed,
        "in_channels": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "roles": ROLES,
        "train_families": splits["train_families"],
        "val_families": splits["val_families"],
        "test_families": splits["test_families"],
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metrics = evaluate_loader(model, val_loader, criterion, device)
        role_l1 = ", ".join(
            f"{role}:{val_metrics['role_l1'].get(role, float('nan')):.4f}" for role in ROLES
        )
        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | role_l1: {role_l1}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                args.out,
            )
            with args.out.with_suffix(args.out.suffix + ".config.json").open("w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Best checkpoint: {args.out} (val_loss={best_val:.6f})")
    if not test_rows:
        print("No held-out test rows were available for this dataset size.")


def run_epoch(model, loader, criterion, device, optimizer) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        y, mask = batch_targets(batch, pred.size(0))
        loss = masked_smooth_l1(pred, y, mask, criterion)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * pred.size(0)
        total_count += pred.size(0)
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device) -> tuple[float, dict[str, Any]]:
    model.eval()
    losses: list[float] = []
    all_pred: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    all_mask: list[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y, mask = batch_targets(batch, pred.size(0))
        loss = masked_smooth_l1(pred, y, mask, criterion)
        losses.append(float(loss.detach().cpu()))
        all_pred.append(pred.detach().cpu())
        all_y.append(y.detach().cpu())
        all_mask.append(mask.detach().cpu())
    if not all_pred:
        return float("inf"), empty_metrics()
    metrics = compute_metrics(torch.cat(all_pred), torch.cat(all_y), torch.cat(all_mask))
    return float(np.mean(losses)), metrics


def batch_targets(batch, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    y = batch.y_boxes.view(batch_size, NUM_ROLES, 4)
    mask = batch.y_mask.view(batch_size, NUM_ROLES)
    return y.to(batch.x.device), mask.to(batch.x.device)


def masked_smooth_l1(pred, y, mask, criterion) -> torch.Tensor:
    raw = criterion(pred, y)
    weighted = raw * mask.unsqueeze(-1)
    denom = torch.clamp(mask.sum() * 4.0, min=1.0)
    return weighted.sum() / denom


def compute_metrics(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    abs_err = (pred - y).abs()
    role_l1: dict[str, float] = {}
    role_iou: dict[str, float] = {}
    missing_counts: dict[str, int] = {}
    valid_values: list[float] = []
    for role_idx, role in IDX_TO_ROLE.items():
        role_mask = mask[:, role_idx] > 0.5
        missing_counts[role] = int((~role_mask).sum().item())
        if role_mask.any():
            role_abs = abs_err[role_mask, role_idx, :]
            role_l1[role] = float(role_abs.mean().item())
            valid_values.extend(role_abs.reshape(-1).tolist())
            role_iou[role] = float(
                torch.stack([bbox_iou(a, b) for a, b in zip(pred[role_mask, role_idx], y[role_mask, role_idx])])
                .mean()
                .item()
            )
        else:
            role_l1[role] = float("nan")
            role_iou[role] = float("nan")
    return {
        "mean_l1": float(np.mean(valid_values)) if valid_values else float("nan"),
        "role_l1": role_l1,
        "role_iou": role_iou,
        "missing_counts": missing_counts,
    }


def bbox_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = torch.maximum(ax, bx), torch.maximum(ay, by)
    ix2, iy2 = torch.minimum(ax2, bx2), torch.minimum(ay2, by2)
    iw, ih = torch.clamp(ix2 - ix1, min=0), torch.clamp(iy2 - iy1, min=0)
    inter = iw * ih
    union = torch.clamp(aw * ah + bw * bh - inter, min=1e-8)
    return inter / union


def empty_metrics() -> dict[str, Any]:
    return {
        "mean_l1": float("nan"),
        "role_l1": {role: float("nan") for role in ROLES},
        "role_iou": {role: float("nan") for role in ROLES},
        "missing_counts": {role: 0 for role in ROLES},
    }


def split_rows_by_family(rows: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("family_key") or "unknown")].append(row)
    families = sorted(by_family)
    rng = random.Random(seed)
    rng.shuffle(families)
    n = len(families)
    if n == 1:
        train_f, val_f, test_f = families, [], []
    elif n == 2:
        train_f, val_f, test_f = families[:1], families[1:], []
    else:
        n_train = max(1, int(round(n * 0.8)))
        n_val = max(1, int(round(n * 0.1)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train_f = families[:n_train]
        val_f = families[n_train : n_train + n_val]
        test_f = families[n_train + n_val :]
    return {
        "train": [row for fam in train_f for row in by_family[fam]],
        "val": [row for fam in val_f for row in by_family[fam]],
        "test": [row for fam in test_f for row in by_family[fam]],
        "train_families": train_f,
        "val_families": val_f,
        "test_families": test_f,
    }


def print_split_summary(splits: dict[str, Any]) -> None:
    print(
        "Split rows: "
        f"train={len(splits['train'])} ({len(splits['train_families'])} families), "
        f"val={len(splits['val'])} ({len(splits['val_families'])} families), "
        f"test={len(splits['test'])} ({len(splits['test_families'])} families)"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
