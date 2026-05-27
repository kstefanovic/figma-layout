"""Training loop for full RALF model."""

from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path
from typing import Any

from .dataset import (
    RALF_NUMERIC_FEATURE_DIM,
    CachedRalfLayoutPairDataset,
    RalfLayoutPairDataset,
    build_vocabs_from_dataset,
    ralf_collate_fn,
)
from .losses import total_ralf_loss
from .model import RalfTopLevelLayoutTransformer


def _torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for RALF training.") from exc
    return torch, DataLoader


def _tqdm(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def _resolve_device(torch, value: str):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _move_batch_to_device(batch: dict[str, Any], device):
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _run_epoch(model, loader, optimizer, device, train: bool, epoch: int, epochs: int, use_amp: bool, scaler):
    torch, _ = _torch()
    model.train(train)
    total = 0.0
    count = 0
    phase = "train" if train else "val"
    prog = _tqdm(loader, desc=f"{phase} {epoch}/{epochs}", leave=False, dynamic_ncols=True)
    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for batch in prog:
            batch = _move_batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(
                    batch["src_num"],
                    batch["src_role_ids"],
                    batch["src_type_ids"],
                    batch["src_mask"],
                    batch["ret_num"],
                    batch["ret_role_ids"],
                    batch["ret_type_ids"],
                    batch["ret_mask"],
                    batch["ret_scores"],
                    batch["src_center"],
                )
                loss = total_ralf_loss(pred, batch["target"], batch["target_mask"], batch["train_roles"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp and scaler is not None:
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
            if hasattr(prog, "set_postfix"):
                prog.set_postfix(loss=f"{total / max(1, count):.5f}")
            del batch, pred, loss
    return total / max(1, count)


def train_ralf_model(
    *,
    train_path: str | None = None,
    val_path: str | None = None,
    train_cache: str | None = None,
    val_cache: str | None = None,
    output_path: str,
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 3e-4,
    device: str = "auto",
    d_model: int = 128,
    nhead: int = 4,
    retrieval_k: int = 5,
    reports_dir: str | Path = "layout_training/reports",
    patience: int = 25,
    resume: str | None = None,
    num_workers: int | None = None,
    cache_max_open_shards: int = 2,
    cache_load_all: bool = False,
    amp: bool = True,
    tf32: bool = True,
    compile_model: bool = False,
    save_optimizer_state: bool = False,
) -> dict[str, Any]:
    torch, DataLoader = _torch()
    dev = _resolve_device(torch, device)
    use_cuda = dev.type == "cuda"
    if use_cuda and tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    using_cache = bool(train_cache or val_cache)
    if train_cache:
        print(f"Using cached RALF dataset: {train_cache}")
        train_ds = CachedRalfLayoutPairDataset(
            train_cache,
            max_open_shards=cache_max_open_shards,
            load_all=cache_load_all,
        )
        role_vocab, type_vocab = build_vocabs_from_dataset(train_ds)
        train_ds.role_vocab = role_vocab
        train_ds.type_vocab = type_vocab
    else:
        if not train_path:
            raise ValueError("--train is required when --train-cache is not provided")
        print("Training from raw JSONL may be CPU/RAM bottlenecked; consider build_ralf_tensor_cache.")
        train_ds = RalfLayoutPairDataset(train_path)
        role_vocab, type_vocab = train_ds.role_vocab, train_ds.type_vocab

    if val_cache:
        print(f"Using cached RALF dataset: {val_cache}")
        val_ds = CachedRalfLayoutPairDataset(
            val_cache,
            max_open_shards=cache_max_open_shards,
            load_all=cache_load_all,
            role_vocab=role_vocab,
            type_vocab=type_vocab,
        )
    else:
        if not val_path:
            raise ValueError("--val is required when --val-cache is not provided")
        if not using_cache:
            print("Training from raw JSONL may be CPU/RAM bottlenecked; consider build_ralf_tensor_cache.")
        val_ds = RalfLayoutPairDataset(val_path, role_vocab=role_vocab, type_vocab=type_vocab)

    if num_workers is None:
        num_workers = 4 if using_cache else 0
    cpu_count = os.cpu_count() or 1
    if int(num_workers) > max(0, cpu_count - 2):
        warnings.warn(f"num_workers={int(num_workers)} is high for cpu_count={cpu_count}; this may increase RAM pressure.")
    if batch_size >= 256 and not using_cache:
        warnings.warn("batch_size>=256 with raw JSONL mode can be RAM-heavy; consider build_ralf_tensor_cache.")
    dl_kwargs: dict[str, Any] = {
        "num_workers": max(0, int(num_workers)),
        "pin_memory": use_cuda,
        "persistent_workers": int(num_workers) > 0,
        "drop_last": False,
    }
    if int(num_workers) > 0:
        dl_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=ralf_collate_fn, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=ralf_collate_fn, **dl_kwargs)
    src_numeric_dim = RALF_NUMERIC_FEATURE_DIM
    ret_numeric_dim = RALF_NUMERIC_FEATURE_DIM

    model = RalfTopLevelLayoutTransformer(
        num_roles=max(train_ds.role_vocab.values()) + 1,
        num_types=max(train_ds.type_vocab.values()) + 1,
        src_numeric_dim=src_numeric_dim,
        ret_numeric_dim=ret_numeric_dim,
        d_model=d_model,
        nhead=nhead,
        source_layers=3,
        retrieved_layers=2,
    ).to(dev)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    use_amp = bool(amp and use_cuda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_cuda else None
    printed_batch_shapes = False
    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location="cpu")
        model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
        model_to_load.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("last_epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))
        best_epoch = int(ckpt.get("best_epoch", 0))

    gpu_name = torch.cuda.get_device_name(dev.index or 0) if use_cuda else "cpu"
    print(
        f"device={dev} gpu={gpu_name} amp={use_amp} tf32={bool(tf32 and use_cuda)} "
        f"num_workers={int(num_workers)} pin_memory={use_cuda} compile={compile_model}"
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    log_path = reports / "ralf_training_log.csv"
    rows = []
    bad = 0
    for epoch in range(start_epoch, epochs + 1):
        if not printed_batch_shapes:
            first_batch = next(iter(train_loader))
            print(
                "first_batch_shapes:"
                f" src_num={tuple(first_batch['src_num'].shape)}"
                f" ret_num={tuple(first_batch['ret_num'].shape)}"
                f" target={tuple(first_batch['target'].shape)}"
                f" src_mask={tuple(first_batch['src_mask'].shape)}"
                f" ret_mask={tuple(first_batch['ret_mask'].shape)}"
            )
            printed_batch_shapes = True
        tr = _run_epoch(model, train_loader, optimizer, dev, True, epoch, epochs, use_amp, scaler)
        va = _run_epoch(model, val_loader, optimizer, dev, False, epoch, epochs, use_amp, scaler)
        scheduler.step(va)
        rows.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "lr": optimizer.param_groups[0]["lr"]})
        print(f"epoch={epoch}/{epochs} train={tr:.6f} val={va:.6f} lr={optimizer.param_groups[0]['lr']:.6g}")
        if va < best_val:
            best_val = va
            best_epoch = epoch
            bad = 0
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            checkpoint = {
                    "model_type": "ralf_top_level_layout_transformer_v1",
                    "model_state": model_to_save.state_dict(),
                    "config": {
                        "num_roles": max(train_ds.role_vocab.values()) + 1,
                        "num_types": max(train_ds.type_vocab.values()) + 1,
                        "src_numeric_dim": src_numeric_dim,
                        "ret_numeric_dim": ret_numeric_dim,
                        "d_model": d_model,
                        "nhead": nhead,
                        "source_layers": 3,
                        "retrieved_layers": 2,
                        "dropout": 0.1,
                        "retrieval_k": retrieval_k,
                    },
                    "role_vocab": train_ds.role_vocab,
                    "type_vocab": train_ds.type_vocab,
                    "train_args": {
                        "train": train_path,
                        "val": val_path,
                        "train_cache": train_cache,
                        "val_cache": val_cache,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "device": str(dev),
                        "retrieval_k": retrieval_k,
                        "num_workers": int(num_workers),
                        "amp": bool(use_amp),
                        "tf32": bool(tf32 and use_cuda),
                        "compile_model": bool(compile_model),
                        "cache_max_open_shards": int(cache_max_open_shards),
                        "cache_load_all": bool(cache_load_all),
                    },
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "last_epoch": epoch,
                    "val_loss": va,
                }
            if save_optimizer_state:
                checkpoint["optimizer_state"] = optimizer.state_dict()
            torch.save(checkpoint, out)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stopping at epoch {epoch}")
                break
    with log_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        w.writeheader()
        w.writerows(rows)
    return {"output": str(out), "best_val_loss": best_val, "best_epoch": best_epoch, "log": str(log_path)}
