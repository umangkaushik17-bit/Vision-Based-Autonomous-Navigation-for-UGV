"""Complete GPU training pipeline for outdoor UGV traversability segmentation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import yaml

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.datasets.ugv_dataset import create_dataloader
from ai.models.traversability_net import create_traversability_model
from ai.training.early_stopping import EarlyStopping
from ai.traversability.loss import CombinedTraversabilityLoss
from ai.traversability.metrics import calculate_segmentation_metrics, calculate_navigation_safety_metrics
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("train_traversability")

def print_environment_banner(cfg: Dict, is_smoke_test: bool):
    """Print required startup diagnostic header with GPU/CUDA information."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None (CPU Execution)"
    gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB" if cuda_available else "N/A"
    cuda_ver = torch.version.cuda if cuda_available else "N/A"

    print("\n" + "=" * 65)
    print("       SIH26126 TRAVERSABILITY AI TRAINING INITIALIZATION        ")
    print("=" * 65)
    print(f"Mode                  : {'[SMOKE TEST]' if is_smoke_test else '[FULL GPU TRAINING]'}")
    print(f"GPU                   : {gpu_name}")
    print(f"GPU memory            : {gpu_mem}")
    print(f"CUDA version          : {cuda_ver}")
    print(f"PyTorch version       : {torch.__version__}")
    print(f"CUDA available        : {cuda_available}")
    print(f"Batch size            : {cfg['training']['batch_size']}")
    print(f"Image size            : {cfg['dataset']['target_size']}")
    print(f"Workers               : {cfg['training']['num_workers']}")
    print(f"AMP                   : {cfg['training']['mixed_precision'] and cuda_available}")
    print(f"Model                 : {cfg['model']['architecture']}")
    print(f"Dataset version       : {cfg['experiment']['version']}")
    print(f"Training configuration: {cfg['experiment']['name']}")
    print("=" * 65 + "\n")

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    use_amp: bool,
    grad_accum_steps: int = 4
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if use_amp and scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, masks) / grad_accum_steps
            scaler.scale(loss).backward()
            if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(images)
            loss = criterion(logits, masks) / grad_accum_steps
            loss.backward()
            if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += float(loss.item() * grad_accum_steps)
        num_batches += 1

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return total_loss / max(num_batches, 1)

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"] # cpu numpy

        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(masks.numpy())

    concat_preds = np.concatenate(all_preds, axis=0)
    concat_targets = np.concatenate(all_targets, axis=0)

    seg_metrics = calculate_segmentation_metrics(concat_preds, concat_targets)
    safety_metrics = calculate_navigation_safety_metrics(concat_preds, concat_targets)

    return {**seg_metrics, **safety_metrics}

def main():
    parser = argparse.ArgumentParser(description="Train Traversability AI model on external GPU or local smoke test")
    parser.add_argument("--config", "-c", type=str, default="configs/traversability_training.yaml", help="Config file")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick lightweight verification smoke test")
    parser.add_argument("--resume", type=str, default=None, help="Resume from existing checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--min-epochs", type=int, default=None, help="Minimum epochs before early stopping can trigger")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience (consecutive epochs without improvement)")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    if args.smoke_test:
        cfg["training"]["epochs"] = cfg["smoke_test"]["epochs"]
        cfg["training"]["batch_size"] = cfg["smoke_test"]["batch_size"]
        cfg["training"]["num_workers"] = 0

    print_environment_banner(cfg, args.smoke_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["training"]["mixed_precision"] and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    # 1. Instantiate DataLoaders
    manifest_path = cfg["dataset"]["manifest_path"]
    target_size = tuple(cfg["dataset"]["target_size"])
    max_samples = cfg.get("smoke_test", {}).get("max_samples", 8) if args.smoke_test else None
    train_loader = create_dataloader(
        split="train",
        manifest_path=manifest_path,
        batch_size=cfg["training"]["batch_size"],
        target_size=target_size,
        num_workers=cfg["training"]["num_workers"],
        max_samples=max_samples
    )
    val_loader = create_dataloader(
        split="val",
        manifest_path=manifest_path,
        batch_size=cfg["training"]["batch_size"],
        target_size=target_size,
        num_workers=cfg["training"]["num_workers"],
        max_samples=max_samples
    )

    # 2. Build Model
    model = create_traversability_model(cfg).to(device)

    # 3. Loss & Optimizer
    loss_weights = cfg["loss"]["class_weights"]
    criterion = CombinedTraversabilityLoss(
        class_weights=loss_weights,
        gamma=cfg["loss"]["gamma"],
        ignore_index=cfg["loss"]["ignore_index"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["scheduler"]["eta_min"]
    )

    # 4. Checkpoint management & Early Stopping
    save_dir = Path(cfg["checkpoint"]["save_dir"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    best_mIoU = -1.0
    history = []

    # Configure Early Stopping
    early_stop_cfg = cfg.get("training", {}).get("early_stopping", {})
    early_stop_enabled = early_stop_cfg.get("enabled", True) and not args.no_early_stopping and not args.smoke_test
    patience = args.patience if args.patience is not None else early_stop_cfg.get("patience", 7)
    min_delta = early_stop_cfg.get("min_delta", 0.001)
    monitor = early_stop_cfg.get("monitor", "val_mIoU")
    mode = early_stop_cfg.get("mode", "max")

    early_stopper: Optional[EarlyStopping] = None
    min_epochs = args.min_epochs if args.min_epochs is not None else early_stop_cfg.get("min_epochs", 25)
    epochs = cfg["training"]["epochs"]
    if early_stop_enabled:
        early_stopper = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            monitor=monitor,
            min_epochs=min_epochs
        )
        print(f"[EARLY STOPPING] Enabled | Min Epochs: {min_epochs} | Max Epochs: {epochs} | Patience: {patience} | Monitor: {monitor}")
    else:
        print("[EARLY STOPPING] Disabled (smoke-test or explicit flag)")

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mIoU = ckpt.get("best_mIoU", -1.0)
        if "early_stopping" in ckpt and early_stopper is not None and ckpt["early_stopping"]:
            early_stopper.load_state_dict(ckpt["early_stopping"])
        print(f"[RESUME] Resumed from epoch {start_epoch} (best_mIoU: {best_mIoU:.4f})")

    # 5. Training Loop
    epochs = cfg["training"]["epochs"]
    grad_accum_steps = cfg["training"].get("gradient_accumulation_steps", 4)
    early_stopped = False
    stopped_epoch = None

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, grad_accum_steps=grad_accum_steps
        )
        val_metrics = evaluate_epoch(model, val_loader, device)
        scheduler.step()
        epoch_time = time.time() - t0

        val_mIoU = val_metrics.get("mIoU", 0.0)
        val_fs_ratio = val_metrics.get("false_safe_pixel_ratio", 0.0)
        val_acc = val_metrics.get("pixel_accuracy", 0.0)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_mIoU": float(val_mIoU),
            "val_acc": float(val_acc),
            "val_false_safe_ratio": float(val_fs_ratio),
            "epoch_time_sec": float(epoch_time)
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch+1:02d}/{epochs:02d}] "
            f"Loss: {train_loss:.4f} | "
            f"Val mIoU: {val_mIoU:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"False-Safe: {val_fs_ratio*100:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        # Early Stopping Evaluation
        should_stop = False
        if early_stopper is not None:
            monitored_val = val_metrics.get(monitor, val_mIoU if monitor == "val_mIoU" else train_loss)
            should_stop = early_stopper.step(monitored_val, epoch=epoch + 1)
            if early_stopper.is_improved:
                print(f" -> [{monitor.upper()} IMPROVED] New best: {early_stopper.best_score:.4f} at epoch {epoch + 1}")
            else:
                print(
                    f" -> [EARLY STOPPING] Counter: {early_stopper.counter}/{early_stopper.patience} "
                    f"(best: {early_stopper.best_score:.4f} at epoch {early_stopper.best_epoch})"
                )

        # Save Latest Checkpoint
        latest_path = save_dir / "latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "history": history,
            "best_mIoU": best_mIoU,
            "early_stopping": early_stopper.state_dict() if early_stopper else None
        }, str(latest_path))

        # Save Best Checkpoint
        if val_mIoU > best_mIoU:
            best_mIoU = val_mIoU
            best_path = save_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "val_metrics": val_metrics,
                "history": history
            }, str(best_path))
            print(f" -> [BEST MODEL SAVED] Val mIoU improved to {best_mIoU:.4f}")

        if should_stop:
            early_stopped = True
            stopped_epoch = epoch + 1
            print(
                f"\n[EARLY STOP TRIGGERED] Stopping training early at epoch {stopped_epoch}. "
                f"No improvement in {monitor} for {early_stopper.patience} consecutive epochs. "
                f"Best {monitor}: {early_stopper.best_score:.4f} recorded at epoch {early_stopper.best_epoch}."
            )
            break

    # 6. Save Training Report
    report_dir = Path("outputs/reports").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": cfg["experiment"]["name"],
            "device": str(device),
            "is_smoke_test": args.smoke_test,
            "best_val_mIoU": best_mIoU,
            "total_epochs_configured": epochs,
            "epochs_completed": len(history),
            "early_stopped": early_stopped,
            "early_stopped_epoch": stopped_epoch,
            "early_stopping_best_epoch": early_stopper.best_epoch if early_stopper else None,
            "history": history
        }, f, indent=2)

    print(f"\n[DONE] Training finished. Report saved to: {report_path}")

if __name__ == "__main__":
    main()
