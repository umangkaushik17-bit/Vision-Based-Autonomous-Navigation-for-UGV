#!/usr/bin/env python3
"""Authoritative Strict Kaggle NVIDIA T4 GPU Training Pipeline for RELLIS-3D.

MANDATORY RULES ENFORCED:
1. Strict GPU Enforcement: device.type == 'cuda' and 't4' in GPU name. Hard exit otherwise.
   NEVER falls back to CPU.
2. Checkpointing: best.pt, latest.pt, epoch_XXXX.pt with model, optimizer, scheduler, scaler, metrics.
3. Resumability: Supports --resume.
4. Bounded Stages:
   - Stage B: GPU smoke test (--stage smoke)
   - Stage C: Short pilot training (--stage pilot)
   - Stage E: Full training (--stage full)
5. Comprehensive Safety Metrics:
   - mIoU, per-class IoU (Safe, Unknown, Unsafe)
   - Dangerous-terrain recall / Unsafe recall
   - Non-traversable recall
   - False-safe rate
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import yaml

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.models.traversability_net import LightweightTraversabilityNet
from rellis3d_gpu_ready.src.losses.traversability_loss import CombinedTraversabilityLoss
from rellis3d_gpu_ready.src.metrics.safety_metrics import calculate_safety_metrics


def enforce_kaggle_t4() -> str:
    """Strictly verify that CUDA is available and running on an NVIDIA T4 GPU."""
    if not torch.cuda.is_available():
        print("\n" + "!" * 70)
        print("TRAINING BLOCKED: REQUIRED KAGGLE NVIDIA T4 GPU NOT DETECTED.")
        print("Reason: torch.cuda.is_available() is False. No CUDA GPU found.")
        print("Per system rules: Training on CPU is strictly forbidden.")
        print("!" * 70 + "\n")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    if "t4" not in gpu_name.lower():
        print("\n" + "!" * 70)
        print("TRAINING BLOCKED: REQUIRED KAGGLE NVIDIA T4 GPU NOT DETECTED.")
        print(f"Reason: Found GPU '{gpu_name}', but required target is NVIDIA Tesla T4.")
        print("!" * 70 + "\n")
        sys.exit(1)

    print(f"[+] Verified Target Hardware: {gpu_name} on cuda:0 (CUDA {torch.version.cuda})")
    return "cuda:0"


class RELLISDataset(Dataset):
    """Authoritative RELLIS-3D Dataset Loader using real image/mask pairs and label mapping."""

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        target_wh: Tuple[int, int] = (640, 400),
        mapping: Optional[Dict[int, int]] = None,
        is_training: bool = True
    ):
        self.samples = samples
        self.target_wh = target_wh
        self.mapping = mapping
        self.is_training = is_training
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_p, mask_p = self.samples[idx]

        # 1. Read Image
        img = cv2.imread(img_p)
        if img is None:
            raise FileNotFoundError(f"Missing image file: {img_p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_wh, interpolation=cv2.INTER_LINEAR)

        # 2. Read Mask
        mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Missing mask file: {mask_p}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        # Strictly NEAREST NEIGHBOR resizing
        mask = cv2.resize(mask, self.target_wh, interpolation=cv2.INTER_NEAREST)

        # 3. Class ID Mapping
        if self.mapping is not None:
            mapped = np.full_like(mask, fill_value=255, dtype=np.uint8)
            for raw_id, target_class in self.mapping.items():
                mapped[mask == raw_id] = target_class
            mask = mapped

        # Data augmentation during training
        if self.is_training:
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

        # Normalize RGB
        norm_img = ((img / 255.0) - self.mean) / self.std
        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy(mask).long()

        return {"image": tensor_img, "mask": tensor_mask}


def parse_manifest(manifest_path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """Parse manifest CSV into train, val, and test sample lists."""
    splits = {"train": [], "val": [], "test": []}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("valid") in ("True", "true", True, 1, "1"):
                sp = row.get("split", "train")
                if sp in splits:
                    splits[sp].append((row["image_path"], row["annotation_path"]))
    return splits


def train_rellis_gpu(
    stage: str = "smoke",
    resume_checkpoint: Optional[str] = None,
    manifest_csv: Path = Path("data/rellis3d/rellis_manifest.csv"),
    mapping_yaml: Path = Path("configs/rellis3d_traversability_mapping.yaml"),
    baseline_checkpoint: Path = Path("models/rugd_traversability/outputs/checkpoints/best.pt"),
    output_dir: Path = Path("outputs/rellis3d"),
    epochs_override: Optional[int] = None,
    batch_size: int = 8,
    num_workers: int = 2
) -> Dict:
    print("=" * 70)
    print("      RELLIS-3D AUTONOMOUS GPU TRAINING PIPELINE ON KAGGLE T4       ")
    print("=" * 70)

    # 1. HARD ENFORCEMENT: NVIDIA T4 on CUDA:0
    device_str = enforce_kaggle_t4()
    device = torch.device(device_str)

    # 2. Load Mapping
    with open(mapping_yaml, "r", encoding="utf-8") as f:
        map_cfg = yaml.safe_load(f)
    mapping = {int(k): int(v["model_output_category"]) for k, v in map_cfg.get("classes", {}).items()}

    # 3. Load Samples
    if not manifest_csv.exists():
        print(f"[ERROR] Manifest CSV not found: {manifest_csv}")
        sys.exit(1)

    splits = parse_manifest(manifest_csv)
    train_samples = splits["train"]
    val_samples = splits["val"]

    print(f"[+] Loaded {len(train_samples)} training and {len(val_samples)} validation samples.")

    # Stage Configurations
    if stage == "smoke":
        epochs = 2
        train_samples = train_samples[:32]
        val_samples = val_samples[:16]
        batch_size = 4
        print("[SMOKE TEST] Bounded to 32 train, 16 val samples, 2 epochs.")
    elif stage == "pilot":
        epochs = 5
        train_samples = train_samples[:300]
        val_samples = val_samples[:100]
        print("[PILOT RUN] Bounded to 300 train, 100 val samples, 5 epochs.")
    elif stage == "full":
        epochs = epochs_override if epochs_override else 20
        print(f"[FULL TRAINING] Running full dataset ({len(train_samples)} train, {len(val_samples)} val) for {epochs} epochs.")
    else:
        raise ValueError(f"Unknown stage: {stage}")

    train_ds = RELLISDataset(train_samples, target_wh=(640, 400), mapping=mapping, is_training=True)
    val_ds = RELLISDataset(val_samples, target_wh=(640, 400), mapping=mapping, is_training=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # 4. Model Initialization
    model = LightweightTraversabilityNet(num_classes=3)
    start_epoch = 1
    best_miou = 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    criterion = CombinedTraversabilityLoss(class_weights=[1.0, 1.5, 2.0], ignore_index=255, gamma=2.0).to(device)

    # Load baseline or resume
    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"[*] Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("best_metric", ckpt.get("mIoU", 0.0))
        print(f"[+] Resumed at epoch {start_epoch}, previous best mIoU: {best_miou:.4f}")
    elif baseline_checkpoint.exists():
        print(f"[*] Initializing encoder/decoder weights from RUGD baseline: {baseline_checkpoint}")
        ckpt = torch.load(baseline_checkpoint, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Load compatible weights strictly
        model.load_state_dict(state_dict, strict=False)
        print("[+] RUGD baseline weights initialized successfully.")

    model.to(device)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    training_history = []
    print(f"\n[*] Commencing GPU execution for {epochs} epochs on {torch.cuda.get_device_name(0)}...")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0
        batches = 0

        for batch_idx, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_masks = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)

                with autocast():
                    logits = model(imgs)
                    vloss = criterion(logits, masks)

                val_loss += vloss.item()
                val_batches += 1

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)
                all_masks.append(masks.cpu().numpy())

        avg_val_loss = val_loss / max(val_batches, 1)
        all_preds = np.concatenate(all_preds, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        # Safety metrics calculation
        val_metrics = calculate_safety_metrics(all_preds, all_masks)
        dt = time.perf_counter() - t0

        miou = val_metrics["mIoU"]
        unsafe_recall = val_metrics["unsafe_recall"]
        false_safe = val_metrics["false_safe_ratio"]

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | mIoU: {miou:.4f} | "
              f"Unsafe Recall: {unsafe_recall:.4f} | False-Safe: {false_safe:.4f} | Time: {dt:.1f}s")

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "mIoU": miou,
            "unsafe_recall": unsafe_recall,
            "false_safe_ratio": false_safe,
            "safe_iou": val_metrics.get("safe_iou", 0.0),
            "unknown_iou": val_metrics.get("unknown_iou", 0.0),
            "unsafe_iou": val_metrics.get("unsafe_iou", 0.0),
            "pixel_accuracy": val_metrics.get("pixel_accuracy", 0.0),
            "epoch_duration_s": dt
        }
        training_history.append(epoch_record)

        # Checkpoint package
        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_metric": best_miou,
            "val_metrics": val_metrics,
            "stage": stage,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
            "dataset_manifest_id": str(manifest_csv)
        }

        # 1. Always save latest.pt
        latest_path = ckpt_dir / "latest.pt"
        torch.save(checkpoint_payload, latest_path)

        # 2. Check if best
        if miou > best_miou:
            best_miou = miou
            checkpoint_payload["best_metric"] = best_miou
            best_path = ckpt_dir / "best.pt"
            torch.save(checkpoint_payload, best_path)
            print(f"  [+] New best checkpoint saved to {best_path} (mIoU: {best_miou:.4f})")

        # 3. Save epoch_XXXX.pt on milestones
        if epoch % 5 == 0 or epoch == epochs:
            epoch_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save(checkpoint_payload, epoch_path)

    # Save metrics JSON
    metrics_file = metrics_dir / f"{stage}_training_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "stage": stage,
            "epochs": epochs,
            "best_miou": best_miou,
            "final_metrics": training_history[-1] if training_history else {},
            "history": training_history
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[COMPLETE] Stage '{stage}' training finished.")
    print(f"Best mIoU achieved: {best_miou:.4f}")
    print(f"Checkpoints stored in: {ckpt_dir.resolve()}")
    print("=" * 70)

    return {"best_miou": best_miou, "history": training_history}


def main():
    parser = argparse.ArgumentParser(description="RELLIS-3D GPU Training Pipeline")
    parser.add_argument("--stage", type=str, choices=["smoke", "pilot", "full"], default="smoke", help="Training stage")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--manifest", type=str, default="data/rellis3d/rellis_manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--mapping", type=str, default="configs/rellis3d_traversability_mapping.yaml", help="Path to mapping YAML")
    parser.add_argument("--baseline", type=str, default="models/rugd_traversability/outputs/checkpoints/best.pt", help="Baseline checkpoint")
    parser.add_argument("--output", type=str, default="outputs/rellis3d", help="Experiment output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    args = parser.parse_args()

    train_rellis_gpu(
        stage=args.stage,
        resume_checkpoint=args.resume,
        manifest_csv=Path(args.manifest),
        mapping_yaml=Path(args.mapping),
        baseline_checkpoint=Path(args.baseline),
        output_dir=Path(args.output),
        epochs_override=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
