"""Comprehensive evaluation engine disaggregating synthetic, public, and own-real data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.datasets.base import TraversabilityTargetClass
from ai.datasets.split_manager import SplitManager
from ai.datasets.ugv_dataset import UGVDataset
from ai.models.traversability_net import create_traversability_model
from ai.traversability.baseline_classifier import BaselineTerrainClassifier
from ai.traversability.metrics import calculate_segmentation_metrics, calculate_navigation_safety_metrics
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("evaluate_traversability")

def evaluate_subset(
    model: torch.nn.Module,
    samples: List,
    device: torch.device,
    target_size=(480, 640)
) -> Dict[str, float]:
    """Evaluate deep-learning model on a subset of samples."""
    if not samples:
        return {"mIoU": 0.0, "pixel_accuracy": 0.0, "false_safe_pixel_ratio": 0.0}

    dataset = UGVDataset(samples, target_size=target_size, is_training=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"]
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(masks.numpy())

    concat_preds = np.concatenate(all_preds, axis=0)
    concat_targets = np.concatenate(all_targets, axis=0)

    seg_m = calculate_segmentation_metrics(concat_preds, concat_targets)
    safe_m = calculate_navigation_safety_metrics(concat_preds, concat_targets)
    return {**seg_m, **safe_m}

def evaluate_baseline(samples: List, target_size=(480, 640)) -> Dict[str, float]:
    """Evaluate non-deep-learning classical baseline on test samples for comparison."""
    if not samples:
        return {"mIoU": 0.0, "pixel_accuracy": 0.0, "false_safe_pixel_ratio": 0.0}

    import cv2
    baseline = BaselineTerrainClassifier()
    all_preds = []
    all_targets = []

    for s in samples[:15]: # Representative subset
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        h_t, w_t = target_size
        img_res = cv2.resize(img, (w_t, h_t))
        pred = baseline.predict(img_res)

        if s.mask_path and Path(s.mask_path).exists():
            mask = cv2.imread(s.mask_path, cv2.IMREAD_GRAYSCALE)
            mask_res = cv2.resize(mask, (w_t, h_t), interpolation=cv2.INTER_NEAREST)
        else:
            mask_res = np.full((h_t, w_t), TraversabilityTargetClass.UNKNOWN, dtype=np.uint8)

        all_preds.append(pred)
        all_targets.append(mask_res)

    if not all_preds:
        return {"mIoU": 0.0, "pixel_accuracy": 0.0, "false_safe_pixel_ratio": 0.0}

    concat_preds = np.array(all_preds)
    concat_targets = np.array(all_targets)

    seg_m = calculate_segmentation_metrics(concat_preds, concat_targets)
    safe_m = calculate_navigation_safety_metrics(concat_preds, concat_targets)
    return {**seg_m, **safe_m}

def main():
    parser = argparse.ArgumentParser(description="Evaluate traversability models and compute domain gap metrics")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt", help="Model checkpoint path")
    parser.add_argument("--manifest", type=str, default="data/manifests/dataset_manifest.json", help="Manifest path")
    parser.add_argument("--output-report", type=str, default="outputs/reports/evaluation_report.json")
    args = parser.parse_args()

    manager = SplitManager(args.manifest)
    manager.load_manifest()
    test_samples = manager.get_samples_by_split("test")

    print(f"\n[EVALUATION] Evaluating on {len(test_samples)} Held-Out Final Test Samples...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_traversability_model().to(device)

    # Load weights if checkpoint exists, else evaluate initialization/baseline
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[EVALUATION] Loaded weights from: {ckpt_path}")
    else:
        print(f"[NOTICE] Checkpoint {ckpt_path} not found. Running baseline comparison mode.")

    # 1. Classical Non-DL Baseline
    baseline_metrics = evaluate_baseline(test_samples)

    # 2. DL Model Overall Evaluation
    dl_overall = evaluate_subset(model, test_samples, device)

    # 3. Disaggregated Evaluation: Synthetic vs Own Real
    syn_samples = [s for s in test_samples if s.source_type == "synthetic"]
    own_samples = [s for s in test_samples if s.source_type == "own_real"]

    dl_syn = evaluate_subset(model, syn_samples, device)
    dl_own = evaluate_subset(model, own_samples, device)

    print("\n" + "=" * 65)
    print("        SIH26126 FINAL TRAVERSABILITY BENCHMARK REPORT          ")
    print("=" * 65)
    print(f"{'Method / Split':<28} | {'mIoU':<8} | {'Accuracy':<8} | {'False-Safe':<10}")
    print("-" * 65)
    print(f"{'1. Classical CV Baseline':<28} | {baseline_metrics.get('mIoU', 0):.4f}   | {baseline_metrics.get('pixel_accuracy', 0):.4f}   | {baseline_metrics.get('false_safe_pixel_ratio', 0)*100:.2f}%")
    print(f"{'2. DL Model (Overall)':<28} | {dl_overall.get('mIoU', 0):.4f}   | {dl_overall.get('pixel_accuracy', 0):.4f}   | {dl_overall.get('false_safe_pixel_ratio', 0)*100:.2f}%")
    print(f"{'   - Synthetic Test Split':<28} | {dl_syn.get('mIoU', 0):.4f}   | {dl_syn.get('pixel_accuracy', 0):.4f}   | {dl_syn.get('false_safe_pixel_ratio', 0)*100:.2f}%")
    print(f"{'   - Own-Real Test Split':<28} | {dl_own.get('mIoU', 0):.4f}   | {dl_own.get('pixel_accuracy', 0):.4f}   | {dl_own.get('false_safe_pixel_ratio', 0)*100:.2f}%")
    print("=" * 65)

    # Domain Gap Analysis
    domain_gap_mIoU = dl_syn.get("mIoU", 0.0) - dl_own.get("mIoU", 0.0)
    print(f"Observed Domain Gap (Synthetic mIoU - Own Real mIoU): {domain_gap_mIoU:+.4f}\n")

    # Save report
    out_p = Path(args.output_report).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({
            "test_samples": len(test_samples),
            "baseline_metrics": baseline_metrics,
            "dl_overall_metrics": dl_overall,
            "synthetic_metrics": dl_syn,
            "own_real_metrics": dl_own,
            "domain_gap_mIoU": domain_gap_mIoU
        }, f, indent=2)

    print(f"[REPORT] Evaluation metrics saved to: {out_p}")

if __name__ == "__main__":
    main()
