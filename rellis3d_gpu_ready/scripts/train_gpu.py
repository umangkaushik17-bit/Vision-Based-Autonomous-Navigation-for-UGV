#!/usr/bin/env python3
"""Strict GPU-Only Training Pipeline for RELLIS-3D Dense Traversability.

ENFORCEMENT:
- Terminates immediately if CUDA GPU is not available.
- Strictly refuses CPU fallback training.
- Uses Automatic Mixed Precision (AMP) and gradient accumulation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import yaml

# Add script directory parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.traversability_net import LightweightTraversabilityNet
from src.datasets.rellis_dataset import RELLIS3DDataset
from src.losses.traversability_loss import CombinedTraversabilityLoss
from src.metrics.safety_metrics import calculate_safety_metrics

def enforce_gpu_environment() -> str:
    """Enforce strict GPU execution policy."""
    if not torch.cuda.is_available():
        print("\n" + "!" * 65)
        print("  CRITICAL ERROR: GPU TRAINING REQUESTED BUT NO CUDA GPU DETECTED  ")
        print("!" * 65)
        print("Violation: Attempting to run train_gpu.py on a CPU-only environment.")
        print("Per system rules: CPU fallback training is strictly prohibited.")
        print("Action: Terminating training pipeline immediately with exit code 1.")
        print("!" * 65 + "\n")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"[OK] Verified GPU Hardware: {device_name} (CUDA {torch.version.cuda})")
    return "cuda"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Training YAML config")
    parser.add_argument("--dataset-root", type=str, default="./data/raw/public/RELLIS-3D", help="Extracted RELLIS-3D root")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-epoch smoke test")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 prevents container /dev/shm bus error)")
    args = parser.parse_args()

    # 1. Enforce GPU-only
    device = enforce_gpu_environment()

    # 2. Load Config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 65)
    print(f"      RELLIS-3D GPU TRAINING PIPELINE — {cfg['experiment']['name']}      ")
    print("=" * 65)

    # 3. Locate Dataset Samples
    ds_root = Path(args.dataset_root)
    if not ds_root.exists():
        print(f"[ERROR] RELLIS-3D root directory does not exist: {ds_root}")
        print("Please run scripts/discover_dataset.py or extract RELLIS-3D archives first.")
        sys.exit(1)

    paired_samples = []
    for seq in sorted(ds_root.iterdir()):
        if seq.is_dir() and (seq / "pylon_camera_node").exists():
            rgb_dir = seq / "pylon_camera_node"
            label_dir = seq / "pylon_camera_node_label_id"
            for img in sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png"))):
                mask = label_dir / f"{img.stem}.png"
                if mask.exists():
                    paired_samples.append((str(img), str(mask)))

    if not paired_samples:
        print(f"[ERROR] No valid image-label pairs found in {ds_root}.")
        sys.exit(1)

    print(f"Discovered {len(paired_samples)} paired training/validation samples.")

    # Split into train/val
    np.random.seed(42)
    np.random.shuffle(paired_samples)
    split_idx = int(len(paired_samples) * 0.8)
    train_samples = paired_samples[:split_idx]
    val_samples = paired_samples[split_idx:]

    if args.smoke_test:
        print("[SMOKE TEST ACTIVE] Bounding dataset to 32 train, 16 val samples.")
        train_samples = train_samples[:32]
        val_samples = val_samples[:16]

    train_ds = RELLIS3DDataset(train_samples, is_training=True)
    val_ds = RELLIS3DDataset(val_samples, is_training=False)

    batch_size = 4 if args.smoke_test else cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 4. Initialize Model
    model = LightweightTraversabilityNet(num_classes=cfg["model"]["num_classes"])
    
    init_ckpt = cfg["model"].get("initial_checkpoint")
    if init_ckpt and Path(init_ckpt).exists():
        print(f"Initializing weights from baseline: {init_ckpt}")
        ckpt = torch.load(init_ckpt, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Load compatible weights
        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    # 5. Optimizer, Loss & Scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )
    criterion = CombinedTraversabilityLoss(
        class_weights=cfg["loss"]["class_weights"],
        ignore_index=cfg["loss"]["ignore_index"],
        gamma=cfg["loss"]["focal_gamma"]
    ).to(device)
    scaler = GradScaler() if cfg["training"]["mixed_precision"] else None

    epochs = 2 if args.smoke_test else cfg["training"]["max_epochs"]
    grad_accum = cfg["training"].get("grad_accum_steps", 1)

    best_miou = 0.0
    out_dir = Path(cfg["checkpointing"]["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []

    print(f"\nStarting GPU Training on {device} ({epochs} Epochs)...")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0
        batches = 0

        for idx, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            if scaler is not None:
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, masks) / grad_accum
                scaler.scale(loss).backward()
                if (idx + 1) % grad_accum == 0 or (idx + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(imgs)
                loss = criterion(logits, masks) / grad_accum
                loss.backward()
                if (idx + 1) % grad_accum == 0 or (idx + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss += loss.item() * grad_accum
            batches += 1

        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].cpu().numpy()
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(masks)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        val_metrics = calculate_safety_metrics(all_preds, all_targets)
        dt = time.perf_counter() - t0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss/batches:.4f} | "
              f"Val mIoU: {val_metrics['mIoU']:.4f} | Unsafe Recall: {val_metrics['unsafe_recall']:.4f} | "
              f"False-Safe: {val_metrics['false_safe_ratio']:.4f} | Time: {dt:.1f}s")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / batches,
            "val_mIoU": val_metrics["mIoU"],
            "unsafe_recall": val_metrics["unsafe_recall"],
            "false_safe_ratio": val_metrics["false_safe_ratio"],
            "epoch_time_s": dt
        }
        history.append(epoch_record)

        # Checkpoint best
        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            best_path = out_dir / "best.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "mIoU": best_miou}, best_path)
            print(f"  --> Saved new best checkpoint to {best_path} (mIoU={best_miou:.4f})")

        # Always save latest
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "mIoU": val_metrics["mIoU"]}, out_dir / "latest.pt")

    # Save training report
    report_path = Path("reports/training_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# RELLIS-3D Training Report\n\n- Best Val mIoU: {best_miou:.4f}\n- Total Epochs: {epochs}\n")
    print(f"\n[COMPLETE] Training concluded successfully. Best mIoU: {best_miou:.4f}.")

if __name__ == "__main__":
    main()
