"""Evaluation metrics engine for segmentation accuracy and navigation safety."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import torch

from ai.datasets.base import TraversabilityTargetClass

def calculate_segmentation_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 3,
    ignore_index: int = 255
) -> Dict[str, float]:
    """Calculate mIoU, per-class IoU, and overall pixel accuracy."""
    valid_mask = (targets != ignore_index)
    pred_valid = predictions[valid_mask]
    target_valid = targets[valid_mask]

    if len(target_valid) == 0:
        return {"mIoU": 0.0, "pixel_accuracy": 0.0}

    pixel_accuracy = float(np.mean(pred_valid == target_valid))

    class_names = ["SAFE", "UNKNOWN", "UNSAFE"]
    ious: List[float] = []
    metrics: Dict[str, float] = {"pixel_accuracy": pixel_accuracy}

    for c in range(num_classes):
        pred_c = (pred_valid == c)
        target_c = (target_valid == c)

        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()

        if union == 0:
            iou = 1.0 # No pixels in ground truth or prediction
        else:
            iou = float(intersection) / float(union)

        c_name = class_names[c] if c < len(class_names) else f"class_{c}"
        metrics[f"iou_{c_name.lower()}"] = iou
        ious.append(iou)

    metrics["mIoU"] = float(np.mean(ious))
    return metrics

def calculate_navigation_safety_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = 255
) -> Dict[str, float]:
    """Calculate critical safety metrics prioritizing hazard detection over raw accuracy."""
    valid_mask = (targets != ignore_index)
    pred_valid = predictions[valid_mask]
    target_valid = targets[valid_mask]

    total_valid_pixels = len(target_valid)
    if total_valid_pixels == 0:
        return {
            "false_safe_pixel_ratio": 0.0,
            "false_unsafe_ratio": 0.0,
            "unsafe_recall": 0.0,
            "traversable_precision": 0.0,
            "unknown_ratio": 0.0
        }

    # True labels
    true_safe = (target_valid == TraversabilityTargetClass.SAFE)
    true_unsafe = (target_valid == TraversabilityTargetClass.UNSAFE)

    # Predicted labels
    pred_safe = (pred_valid == TraversabilityTargetClass.SAFE)
    pred_unsafe = (pred_valid == TraversabilityTargetClass.UNSAFE)
    pred_unknown = (pred_valid == TraversabilityTargetClass.UNKNOWN)

    # 1. False-Safe: The most critical failure in UGV autonomy.
    # True = UNSAFE, but Predicted = SAFE.
    total_true_unsafe = true_unsafe.sum()
    if total_true_unsafe > 0:
        false_safe_pixels = np.logical_and(true_unsafe, pred_safe).sum()
        false_safe_pixel_ratio = float(false_safe_pixels) / float(total_true_unsafe)
        unsafe_recall = float(np.logical_and(true_unsafe, pred_unsafe).sum()) / float(total_true_unsafe)
    else:
        false_safe_pixel_ratio = 0.0
        unsafe_recall = 1.0

    # 2. False-Unsafe: Overly conservative behavior.
    # True = SAFE, but Predicted = UNSAFE.
    total_true_safe = true_safe.sum()
    if total_true_safe > 0:
        false_unsafe_pixels = np.logical_and(true_safe, pred_unsafe).sum()
        false_unsafe_ratio = float(false_unsafe_pixels) / float(total_true_safe)
        traversable_precision = float(np.logical_and(true_safe, pred_safe).sum()) / float(max(pred_safe.sum(), 1))
    else:
        false_unsafe_ratio = 0.0
        traversable_precision = 0.0

    # 3. Unknown Ratio
    unknown_ratio = float(pred_unknown.sum()) / float(total_valid_pixels)

    return {
        "false_safe_pixel_ratio": false_safe_pixel_ratio,
        "false_unsafe_ratio": false_unsafe_ratio,
        "unsafe_recall": unsafe_recall,
        "traversable_precision": traversable_precision,
        "unknown_ratio": unknown_ratio
    }
