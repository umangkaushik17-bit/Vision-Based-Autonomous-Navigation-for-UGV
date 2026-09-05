"""Comprehensive evaluation and benchmarking suite for SAM-TP traversability on multi-domain splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image

from ai.traversability.sam_tp import SAMTPTraversabilityPredictor
from ai.traversability.visualization import (
    create_difference_map,
    create_heatmap,
    create_traversability_overlay
)
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("benchmark_sam_tp")


def evaluate_frame_pair(
    pred_prob: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
    ignore_index: int = 255
) -> Dict[str, float]:
    """Calculate safety metrics for a single prediction and ground truth pair.

    GT mask conventions:
      - 3-class: 0 = SAFE, 1 = UNKNOWN, 2 = UNSAFE/OBSTACLE, 255 = VOID.
      - Binary: 1 = SAFE, 0 = UNSAFE.
    """
    pred_safe = pred_prob >= threshold

    if gt_mask.max() > 1:
        valid = gt_mask != ignore_index
        gt_safe = (gt_mask == 0) & valid
        gt_unsafe = (gt_mask == 2) & valid
    else:
        valid = np.ones_like(pred_safe, dtype=bool)
        gt_safe = (gt_mask == 1) & valid
        gt_unsafe = (gt_mask == 0) & valid

    tp = np.logical_and(pred_safe & valid, gt_safe).sum()
    fp = np.logical_and(pred_safe & valid, gt_unsafe).sum()  # FALSE-SAFE: Critical hazard!
    fn = np.logical_and((~pred_safe) & valid, gt_safe).sum()  # FALSE-UNSAFE: Pessimistic
    tn = np.logical_and((~pred_safe) & valid, gt_unsafe).sum()

    total_valid = valid.sum()
    if total_valid == 0:
        return {}

    # Standard segmentation metrics
    intersection = tp
    union = tp + fp + fn
    iou_safe = float(intersection / max(union, 1))

    intersection_unsafe = tn
    union_unsafe = tn + fp + fn
    iou_unsafe = float(intersection_unsafe / max(union_unsafe, 1))

    miou = float((iou_safe + iou_unsafe) / 2.0)
    pixel_acc = float((tp + tn) / max(total_valid, 1))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-6))

    # Safety-critical error ratios
    total_unsafe = fp + tn
    false_safe_pixel_ratio = float(fp / max(total_valid, 1))
    false_safe_area_ratio = float(fp / max(total_unsafe, 1)) if total_unsafe > 0 else 0.0
    false_unsafe_pixel_ratio = float(fn / max(total_valid, 1))

    return {
        "iou_safe": iou_safe,
        "iou_unsafe": iou_unsafe,
        "miou": miou,
        "pixel_accuracy": pixel_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_safe_pixel_ratio": false_safe_pixel_ratio,
        "false_safe_area_ratio": false_safe_area_ratio,
        "false_unsafe_pixel_ratio": false_unsafe_pixel_ratio,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_valid": int(total_valid)
    }


def find_dataset_pairs(base_dir: Path, max_samples_per_domain: int = 20) -> Dict[str, List[Tuple[Path, Path]]]:
    """Locate matching (RGB, GT mask) image pairs across Synthetic, Public, and Own data."""
    domains = {
        "synthetic": [],
        "public_rugd": [],
        "own_real": []
    }

    seq_dir = base_dir / "data" / "processed" / "sequences"
    if not seq_dir.exists():
        logger.warning(f"Sequences directory does not exist: {seq_dir}")
        return domains

    for p in seq_dir.rglob("*.png"):
        if "rgb" in p.parts:
            mask_candidates = [
                p.parent.parent / "masks" / p.name,
                p.parent.parent / "masks" / p.name.replace("_rgb.png", "_mask.png"),
                p.parent.parent / "masks" / p.name.replace(".png", "_mask.png")
            ]
            for m_cand in mask_candidates:
                if m_cand.exists():
                    path_str = str(p).lower()
                    if "synthetic" in path_str:
                        domains["synthetic"].append((p, m_cand))
                    elif "rugd" in path_str:
                        domains["public_rugd"].append((p, m_cand))
                    else:
                        domains["own_real"].append((p, m_cand))
                    break

    # Subsample per domain
    for k in domains:
        if len(domains[k]) > max_samples_per_domain:
            domains[k] = domains[k][:max_samples_per_domain]

    return domains


def run_benchmark(
    max_samples: int = 15,
    output_dir: Path = Path("outputs/traversability/baseline"),
    failures_dir: Path = Path("outputs/traversability/failures")
) -> Dict:
    """Execute complete multi-domain evaluation benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path.cwd()
    pairs = find_dataset_pairs(base_dir, max_samples_per_domain=max_samples)

    logger.info(f"Dataset pairs discovered:")
    for d, pair_list in pairs.items():
        logger.info(f"  {d}: {len(pair_list)} pairs")

    predictor = SAMTPTraversabilityPredictor(contrast_refine=True)
    predictor.warmup()

    all_results = {}
    overall_metrics: Dict[str, List[float]] = {
        "miou": [],
        "iou_safe": [],
        "pixel_accuracy": [],
        "precision": [],
        "recall": [],
        "false_safe_pixel_ratio": [],
        "false_safe_area_ratio": [],
        "false_unsafe_pixel_ratio": []
    }

    failure_records = []
    saved_visuals = 0

    for domain_name, pair_list in pairs.items():
        domain_metrics: Dict[str, List[float]] = {k: [] for k in overall_metrics}
        logger.info(f"Evaluating domain: {domain_name} ({len(pair_list)} frames)...")

        for idx, (img_path, mask_path) in enumerate(pair_list):
            res = predictor.predict(img_path)
            gt = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

            if gt.shape[:2] != res.mask.shape:
                gt = cv2.resize(gt, (res.mask.shape[1], res.mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            metrics = evaluate_frame_pair(res.mask, gt)
            if not metrics:
                continue

            for k in domain_metrics:
                domain_metrics[k].append(metrics[k])
                overall_metrics[k].append(metrics[k])

            # Safety failure tracking (False-Safe area > 10%)
            is_failure = metrics["false_safe_area_ratio"] > 0.10 or metrics["false_safe_pixel_ratio"] > 0.05
            if is_failure:
                failure_records.append({
                    "domain": domain_name,
                    "frame": img_path.name,
                    "false_safe_pixel_ratio": metrics["false_safe_pixel_ratio"],
                    "false_safe_area_ratio": metrics["false_safe_area_ratio"],
                    "miou": metrics["miou"]
                })

            # Save qualitative visual panel
            if saved_visuals < 12 or is_failure:
                target_save_dir = failures_dir if is_failure else output_dir
                prefix = f"fail_{domain_name}_{idx:03d}" if is_failure else f"bench_{domain_name}_{idx:03d}"

                # Generate 4-panel visual audit
                rgb = res.image
                gt_vis = cv2.applyColorMap((gt * 80).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                pred_heat = create_heatmap(res.mask)
                diff_map = create_difference_map(res.mask, gt)

                h, w = rgb.shape[:2]
                panel = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
                panel[0:h, 0:w] = rgb
                panel[0:h, w:2*w] = gt_vis
                panel[h:2*h, 0:w] = pred_heat
                panel[h:2*h, w:2*w] = diff_map

                # Add informative text labels
                cv2.putText(panel, f"RGB: {img_path.name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(panel, "Ground Truth Mask", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(panel, f"SAM-TP Probability (Mean: {res.mask.mean():.3f})", (20, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(panel, f"Safety Audit (FS: {metrics['false_safe_area_ratio']*100:.1f}%) Red=Hazard", (w + 20, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                panel_path = target_save_dir / f"{prefix}_audit_panel.png"
                Image.fromarray(panel).save(panel_path)
                saved_visuals += 1

        all_results[domain_name] = {
            k: float(np.mean(v)) if len(v) > 0 else 0.0
            for k, v in domain_metrics.items()
        }
        all_results[domain_name]["num_samples"] = len(pair_list)

    # Aggregated overall summary
    summary = {
        "model_name": predictor.model_name,
        "device": predictor.device,
        "overall": {
            k: float(np.mean(v)) if len(v) > 0 else 0.0
            for k, v in overall_metrics.items()
        },
        "domains": all_results,
        "failure_modes": {
            "total_evaluated_frames": sum(len(v) for v in pairs.values()),
            "total_failure_frames": len(failure_records),
            "failure_rate": float(len(failure_records) / max(sum(len(v) for v in pairs.values()), 1)),
            "recorded_failures": failure_records
        }
    }

    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Benchmark completed successfully! Results written to: {results_file}")
    logger.info(f"Overall mIoU: {summary['overall']['miou']:.4f} | False-Safe Pixel Ratio: {summary['overall']['false_safe_pixel_ratio']:.4f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM-TP Multi-Domain Benchmark")
    parser.add_argument("--max-samples", type=int, default=15, help="Max samples per domain")
    args = parser.parse_args()

    run_benchmark(max_samples=args.max_samples)
