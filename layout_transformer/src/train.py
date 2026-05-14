"""Train the LayoutTransformer model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .model import LayoutTransformer
from .roles import NUM_ROLES, ROLE_TO_ID, TRAIN_ROLES

PRINT_ROLES = ["hero_image", "background_shape", "brand_group", "headline_group", "legal_text"]

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is not installed.
    tqdm = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--scheduler", choices=["cosine", "plateau", "none"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training on CUDA")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"Using GPU: {torch.cuda.get_device_name(torch.device(args.device))}")
    else:
        print(f"Using device: {args.device}")

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    train_data = LayoutPairDataset(data["train"])
    val_data = LayoutPairDataset(data["val"])
    if len(train_data) == 0:
        raise ValueError("training split is empty")
    if len(val_data) == 0:
        raise ValueError("validation split is empty")

    model_kwargs = {
        "num_roles": NUM_ROLES,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    model = LayoutTransformer(**model_kwargs).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch=max(1, math.ceil(len(train_data) / args.batch_size)))
    amp_enabled = use_cuda and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    role_ids = torch.arange(NUM_ROLES, dtype=torch.long, device=args.device)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    best_val = float("inf")
    bad_epochs = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            role_ids,
            args.device,
            epoch,
            args.grad_clip,
            amp_enabled,
        )
        val_metrics = evaluate(model, val_loader, role_ids, args.device, epoch, amp_enabled)
        step_epoch_scheduler(scheduler, args.scheduler, val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_norm_mae={val_metrics['norm_mae']:.6f} "
            f"val_pixel_mae={val_metrics['pixel_mae']:.2f} "
            f"lr={current_lr:.8f}"
        )
        print_role_errors(val_metrics["role_norm_mae"])
        improved = val_metrics["loss"] < best_val - args.min_delta
        if improved:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "roles": TRAIN_ROLES,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "train_args": sanitize_args(args),
                },
                args.out,
            )
            print(f"saved best checkpoint: {args.out}")
        else:
            bad_epochs += 1
            print(f"no val improvement: {bad_epochs}/{args.patience}")
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"early stopping at epoch {epoch} (best_val_loss={best_val:.6f})")
                break


class LayoutPairDataset(Dataset):
    def __init__(self, split: dict[str, Any]) -> None:
        self.split = split

    def __len__(self) -> int:
        return int(self.split["source_size"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "source_size": self.split["source_size"][idx],
            "target_size": self.split["target_size"][idx],
            "source_bboxes": self.split["source_bboxes"][idx],
            "target_bboxes": self.split["target_bboxes"][idx],
            "role_mask": self.split["role_mask"][idx],
        }


def train_one_epoch(
    model: LayoutTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scaler: torch.amp.GradScaler,
    role_ids: torch.Tensor,
    device: str,
    epoch: int,
    grad_clip: float,
    amp_enabled: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    progress = make_progress(loader, desc=f"train {epoch:03d}")
    for batch in progress:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pred = model(
                role_ids,
                batch["source_bboxes"],
                batch["source_size"],
                batch["target_size"],
                batch["role_mask"],
            )
            loss = masked_smooth_l1(pred, batch["target_bboxes"], batch["role_mask"])
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step_batch_scheduler(scheduler)
        total_loss += float(loss.item()) * int(batch["source_size"].shape[0])
        total_count += int(batch["source_size"].shape[0])
        set_progress_postfix(progress, loss=total_loss / max(1, total_count), lr=optimizer.param_groups[0]["lr"])
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(
    model: LayoutTransformer,
    loader: DataLoader,
    role_ids: torch.Tensor,
    device: str,
    epoch: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_abs = 0.0
    total_abs_count = 0.0
    total_pixel_abs = 0.0
    total_pixel_count = 0.0
    role_abs = torch.zeros(NUM_ROLES, dtype=torch.float64, device=device)
    role_count = torch.zeros(NUM_ROLES, dtype=torch.float64, device=device)
    progress = make_progress(loader, desc=f"val   {epoch:03d}")
    for batch in progress:
        batch = _to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pred = model(
                role_ids,
                batch["source_bboxes"],
                batch["source_size"],
                batch["target_size"],
                batch["role_mask"],
            )
            loss = masked_smooth_l1(pred, batch["target_bboxes"], batch["role_mask"])
        mask = batch["role_mask"].unsqueeze(-1)
        abs_err = (pred - batch["target_bboxes"]).abs() * mask
        pixel_scale = batch["target_size"][:, None, [0, 1, 0, 1]]
        pixel_abs_err = abs_err * pixel_scale

        batch_size = int(batch["source_size"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        total_abs += float(abs_err.sum().item())
        total_abs_count += float(mask.sum().item() * 4)
        total_pixel_abs += float(pixel_abs_err.sum().item())
        total_pixel_count += float(mask.sum().item() * 4)
        role_abs += abs_err.sum(dim=(0, 2)).double()
        role_count += (batch["role_mask"].sum(dim=0) * 4).double()
        set_progress_postfix(progress, loss=total_loss / max(1, total_examples))

    role_norm_mae = (role_abs / role_count.clamp_min(1.0)).detach().cpu()
    return {
        "loss": total_loss / max(1, total_examples),
        "norm_mae": total_abs / max(1.0, total_abs_count),
        "pixel_mae": total_pixel_abs / max(1.0, total_pixel_count),
        "role_norm_mae": role_norm_mae,
    }


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, role_mask: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    mask = role_mask.unsqueeze(-1)
    return (loss * mask).sum() / (mask.sum() * 4).clamp_min(1.0)


def print_role_errors(role_norm_mae: torch.Tensor) -> None:
    pieces = []
    for role in PRINT_ROLES:
        idx = ROLE_TO_ID[role]
        pieces.append(f"{role}={float(role_norm_mae[idx]):.6f}")
    print("role_val_norm_mae " + " ".join(pieces))


def _to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if args.scheduler == "none":
        return None
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, args.patience // 3),
            min_lr=args.min_lr,
        )

    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)
    total_steps = max(1, args.epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        min_factor = args.min_lr / args.lr
        return max(min_factor, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def step_batch_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
) -> None:
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()


def step_epoch_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scheduler_name: str,
    val_loss: float,
) -> None:
    if scheduler is not None and scheduler_name == "plateau":
        scheduler.step(val_loss)


def make_progress(loader: DataLoader, desc: str):
    if tqdm is None:
        return loader
    return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)


def set_progress_postfix(progress: Any, **values: float) -> None:
    if hasattr(progress, "set_postfix"):
        progress.set_postfix({key: f"{value:.6g}" for key, value in values.items()})


def sanitize_args(args: argparse.Namespace) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in vars(args).items():
        clean[key] = str(value) if isinstance(value, Path) else value
    return clean


if __name__ == "__main__":
    main()
