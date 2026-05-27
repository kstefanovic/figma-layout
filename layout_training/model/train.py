"""Training loop for the top-level layout transformer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .dataset import LayoutPairDataset, collate_fn
from .losses import total_layout_loss
from .model import TopLevelLayoutTransformer


def _tqdm(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def _torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training. Install torch to use this command.") from exc
    return torch, DataLoader


def resolve_device(value: str):
    torch, _ = _torch()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def run_epoch(model, loader, optimizer, device, train: bool, *, epoch: int, epochs: int):
    torch, _ = _torch()
    model.train(train)
    total = 0.0
    count = 0
    phase = "train" if train else "val"
    progress = _tqdm(loader, desc=f"{phase} {epoch}/{epochs}", leave=False, dynamic_ncols=True)
    for batch in progress:
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            pred = model(batch["x_num"], batch["role_ids"], batch["type_ids"], batch["mask"], batch["source_center"])
            loss = total_layout_loss(pred, batch["target"], batch["target_mask"], batch["train_roles"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{total / max(1, count):.5f}")
    return total / max(1, count)


def train_model(
    *,
    train_path: str,
    val_path: str,
    output_path: str,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 5e-4,
    device: str = "auto",
    reports_dir: str | Path = "layout_training/reports",
    patience: int = 25,
) -> dict[str, Any]:
    torch, DataLoader = _torch()
    dev = resolve_device(device)
    if dev.type == "cuda":
        print(f"using device: {dev} ({torch.cuda.get_device_name(dev)})")
    else:
        print(f"using device: {dev}")
    train_ds = LayoutPairDataset(train_path)
    val_ds = LayoutPairDataset(val_path, role_vocab=train_ds.role_vocab, type_vocab=train_ds.type_vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    sample = train_ds[0]
    numeric_dim = len(sample["tokens"][0]["x_num"]) if sample["tokens"] else 0
    if numeric_dim <= 0:
        raise ValueError("Training data contains no token numeric features")

    model = TopLevelLayoutTransformer(
        num_roles=max(train_ds.role_vocab.values()) + 1,
        num_types=max(train_ds.type_vocab.values()) + 1,
        numeric_dim=numeric_dim,
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    log_path = reports / "training_log.csv"

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, dev, train=True, epoch=epoch, epochs=epochs)
        val_loss = run_epoch(model, val_loader, optimizer, dev, train=False, epoch=epoch, epochs=epochs)
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
        rows.append(row)
        print(f"epoch={epoch}/{epochs} train={train_loss:.6f} val={val_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6g}")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "engine": "top_level_layout_transformer_v1",
                    "model_state": model.state_dict(),
                    "model_config": {
                        "num_roles": max(train_ds.role_vocab.values()) + 1,
                        "num_types": max(train_ds.type_vocab.values()) + 1,
                        "numeric_dim": numeric_dim,
                        "d_model": 128,
                        "nhead": 4,
                        "num_layers": 4,
                        "dropout": 0.1,
                    },
                    "role_vocab": train_ds.role_vocab,
                    "type_vocab": train_ds.type_vocab,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                },
                output,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    with log_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(rows)
    return {"output": str(output), "best_val_loss": best_val, "best_epoch": best_epoch, "log": str(log_path)}
