"""Training loop for the simplified CORE top-level layout model."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .dataset import CoreLayoutPairDataset, collate_fn
from .losses import total_core_layout_loss
from .model import CoreTopLevelLayoutTransformer


CORE_ALLOWED_PATH_HINTS = (
    "layout_training/data/layout_records/core_layout_records.jsonl",
    "layout_training/data/layout_pairs/core_pairs.jsonl",
    "layout_training/data/layout_pairs/core_train.jsonl",
    "layout_training/data/layout_pairs/core_val.jsonl",
)
NON_CORE_FORBIDDEN_HINTS = (
    "ralf",
    "ralf_pairs",
    "ralf_train",
    "top_level_pairs",
    "top_level_records",
)


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
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def run_epoch(model, loader, optimizer, scaler, device, train: bool, *, epoch: int, epochs: int):
    torch, _ = _torch()
    model.train(train)
    total = 0.0
    count = 0
    phase = "train" if train else "val"
    progress = _tqdm(loader, desc=f"{phase} {epoch}/{epochs}", leave=False, dynamic_ncols=True)
    for batch in progress:
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(batch["x_num"], batch["role_ids"], batch["mask"], batch["source_center"])
                loss = total_core_layout_loss(pred, batch["target"], batch["target_bottom_y"], batch["target_mask"], batch["train_roles"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{total / max(1, count):.5f}")
    return total / max(1, count)


def validate_core_training_paths(*paths: str, allow_non_core_path: bool = False) -> None:
    bad_paths: list[str] = []
    for path in paths:
        low = str(path or "").lower()
        if any(marker in low for marker in NON_CORE_FORBIDDEN_HINTS):
            bad_paths.append(path)
    if bad_paths and not allow_non_core_path:
        raise ValueError(
            "CORE training must use only core records/pairs/train/val files. "
            f"Refusing suspicious non-core path(s): {bad_paths}. "
            "Pass --allow-non-core-path only if you really intend to override this safety check."
        )


def train_model(
    *,
    train_path: str,
    val_path: str,
    output_path: str,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 5e-4,
    device: str = "auto",
    reports_dir: str | Path = "layout_training/reports",
    patience: int = 30,
    num_workers: int = 2,
    compile_model: bool = False,
    allow_non_core_path: bool = False,
) -> dict[str, Any]:
    torch, DataLoader = _torch()
    validate_core_training_paths(train_path, val_path, output_path, allow_non_core_path=allow_non_core_path)
    dev = resolve_device(device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    train_ds = CoreLayoutPairDataset(train_path)
    val_ds = CoreLayoutPairDataset(val_path, role_vocab=train_ds.role_vocab, type_vocab=train_ds.type_vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    sample = train_ds[0]
    numeric_dim = len(sample["tokens"][0]["x_num"]) if sample["tokens"] else 0
    if numeric_dim <= 0:
        raise ValueError("Training data contains no token numeric features")
    model = CoreTopLevelLayoutTransformer(
        num_roles=max(train_ds.role_vocab.values()) + 1,
        numeric_dim=numeric_dim,
    ).to(dev)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    log_path = reports / "core_training_log.csv"

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, dev, train=True, epoch=epoch, epochs=epochs)
        val_loss = run_epoch(model, val_loader, optimizer, scaler, dev, train=False, epoch=epoch, epochs=epochs)
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
        rows.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(
                {
                    "engine": "core_top_level_layout_transformer_v1",
                    "model_state": state_model.state_dict(),
                    "model_config": {
                        "num_roles": max(train_ds.role_vocab.values()) + 1,
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
                break
    with log_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(rows)
    return {"output": str(output), "best_val_loss": best_val, "best_epoch": best_epoch, "log": str(log_path)}
