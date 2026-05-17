"""Shared training utilities for V2 model scripts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .schema import ROLE_WEIGHTS

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


MODEL_INPUT_KEYS = [
    "role_id",
    "node_type_id",
    "source_canvas",
    "target_canvas",
    "source_orientation",
    "target_orientation",
    "source_bbox",
    "source_relative_bbox",
    "flags",
    "font_size",
    "align_h",
    "align_v",
]


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--iou-weight", type=float, default=0.0)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    *,
    args: argparse.Namespace,
    model_class: type[torch.nn.Module],
    dataset_type: str,
) -> None:
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

    payload = torch.load(args.dataset, map_location="cpu", weights_only=False)
    if payload.get("dataset_type") != dataset_type:
        raise ValueError(f"{args.dataset} contains {payload.get('dataset_type')!r}, expected {dataset_type!r}")
    train_data = LayoutRoleDataset(payload["train"])
    val_data = LayoutRoleDataset(payload["val"])
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("train and val splits must both be non-empty")

    model_kwargs = {
        "num_roles": len(payload["roles"]),
        "num_node_types": len(payload["node_type_to_id"]),
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    model = model_class(**model_kwargs).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args, max(1, math.ceil(len(train_data) / args.batch_size)))
    amp_enabled = bool(args.amp and use_cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    role_weights = torch.tensor([ROLE_WEIGHTS.get(role, 1.0) for role in payload["roles"]], dtype=torch.float32, device=args.device)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args, dataset_type, role_weights, optimizer, scheduler, scaler, amp_enabled, epoch)
        val_metrics = run_epoch(model, val_loader, args, dataset_type, role_weights, None, None, scaler, amp_enabled, epoch)
        print(
            f"epoch {epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_bbox_mae={val_metrics['bbox_mae']:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.8f}"
        )
        if val_metrics["loss"] < best_val - args.min_delta:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "dataset_type": dataset_type,
                    "roles": payload["roles"],
                    "role_to_id": payload["role_to_id"],
                    "node_type_to_id": payload["node_type_to_id"],
                    "best_val_loss": best_val,
                    "epoch": epoch,
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


class LayoutRoleDataset(Dataset):
    def __init__(self, split: dict[str, Any]) -> None:
        self.split = split
        self.keys = [key for key, value in split.items() if isinstance(value, torch.Tensor)]

    def __len__(self) -> int:
        return int(self.split["role_id"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: self.split[key][idx] for key in self.keys}


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    dataset_type: str,
    role_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    epoch: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "bbox_mae": 0.0}
    total_count = 0
    progress = make_progress(loader, ("train" if training else "val  ") + f" {epoch:03d}")
    for batch in progress:
        batch = to_device(batch, args.device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", enabled=amp_enabled):
            pred = model(**{key: batch[key] for key in MODEL_INPUT_KEYS})
            loss, bbox_mae = compute_loss(pred, batch, dataset_type, role_weights, args.iou_weight)
        if training:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        batch_size = int(batch["role_id"].shape[0])
        totals["loss"] += float(loss.detach().item()) * batch_size
        totals["bbox_mae"] += float(bbox_mae.detach().item()) * batch_size
        total_count += batch_size
        set_progress_postfix(progress, loss=totals["loss"] / max(1, total_count))
    return {key: value / max(1, total_count) for key, value in totals.items()}


def compute_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    dataset_type: str,
    role_weights: torch.Tensor,
    iou_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = role_weights[batch["role_id"]]
    if dataset_type == "child":
        bbox_pred = pred["relative_bbox"]
        bbox_target = batch["target_relative_bbox"]
    else:
        bbox_pred = pred["bbox"]
        bbox_target = batch["target_bbox"]
    bbox_loss = weighted_smooth_l1(bbox_pred, bbox_target, weights)
    bbox_mae = ((bbox_pred - bbox_target).abs().mean(dim=-1) * weights).sum() / weights.sum().clamp_min(1.0)
    loss = bbox_loss
    if iou_weight > 0:
        loss = loss + iou_weight * giou_loss(bbox_pred, bbox_target, weights)
    if "visibility" in pred:
        loss = loss + 0.05 * weighted_bce(pred["visibility"], batch["target_visibility"], weights)
    if dataset_type == "child":
        text_mask = batch["flags"][:, 0]
        text_weights = weights * text_mask
        loss = loss + 0.35 * weighted_smooth_l1(pred["font_size"], batch["target_font_size"], text_weights)
        if text_weights.sum() > 0:
            loss = loss + 0.1 * weighted_ce(pred["align_h"], batch["target_align_h"], text_weights)
            loss = loss + 0.1 * weighted_ce(pred["align_v"], batch["target_align_v"], text_weights)
    return loss, bbox_mae


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if loss.dim() > 1:
        loss = loss.mean(dim=-1)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def weighted_ce(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits, target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def giou_loss(pred_xywh: torch.Tensor, target_xywh: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    pred = xywh_to_xyxy(pred_xywh)
    target = xywh_to_xyxy(target_xywh)
    inter_lt = torch.maximum(pred[:, :2], target[:, :2])
    inter_rb = torch.minimum(pred[:, 2:], target[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp_min(0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    pred_area = ((pred[:, 2] - pred[:, 0]).clamp_min(0) * (pred[:, 3] - pred[:, 1]).clamp_min(0))
    target_area = ((target[:, 2] - target[:, 0]).clamp_min(0) * (target[:, 3] - target[:, 1]).clamp_min(0))
    union = pred_area + target_area - inter
    iou = inter / union.clamp_min(1e-6)
    enc_lt = torch.minimum(pred[:, :2], target[:, :2])
    enc_rb = torch.maximum(pred[:, 2:], target[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp_min(0)
    enc_area = (enc_wh[:, 0] * enc_wh[:, 1]).clamp_min(1e-6)
    giou = iou - (enc_area - union) / enc_area
    return ((1.0 - giou) * weights).sum() / weights.sum().clamp_min(1.0)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return torch.stack([boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], dim=-1)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if args.scheduler == "none":
        return None
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)
    total_steps = max(1, args.epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return max(args.min_lr / args.lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_progress(loader: DataLoader, desc: str):
    if tqdm is None:
        return loader
    return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)


def set_progress_postfix(progress: Any, **values: float) -> None:
    if hasattr(progress, "set_postfix"):
        progress.set_postfix({key: f"{value:.6g}" for key, value in values.items()})


def sanitize_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

