"""Navigation safety and semantic segmentation evaluation metrics."""

from __future__ import annotations

from typing import Dict
import numpy as np

def calculate_safety_metrics(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 3, ignore_index: int = 255) -> Dict:
    """Compute mIoU, per-class IoUs, unsafe recall, and false-safe pixel ratio.

    Classes:
      0: SAFE
      1: UNKNOWN
      2: UNSAFE (Hazard / Obstacle)
    """
    valid = (targets != ignore_index)
    p = predictions[valid]
    t = targets[valid]

    if len(t) == 0:
        return {"mIoU": 0.0, "pixel_acc": 0.0, "unsafe_recall": 0.0, "false_safe_ratio": 0.0}

    pixel_acc = float((p == t).mean())

    ious = {}
    for c in range(num_classes):
        p_c = (p == c)
        t_c = (t == c)
        intersection = np.logical_and(p_c, t_c).sum()
        union = np.logical_or(p_c, t_c).sum()
        ious[f"iou_class_{c}"] = float(intersection / union) if union > 0 else 0.0

    miou = float(np.mean(list(ious.values())))

    # Safety Critical Metrics
    # Unsafe Recall = (True Unsafe) / (All True Unsafe)
    true_unsafe = (t == 2)
    unsafe_recall = float((p[true_unsafe] == 2).sum() / true_unsafe.sum()) if true_unsafe.sum() > 0 else 1.0

    # False Safe Ratio = (Predicted SAFE when True is UNSAFE) / (All True UNSAFE)
    false_safe_count = np.logical_and(p == 0, t == 2).sum()
    false_safe_ratio = float(false_safe_count / true_unsafe.sum()) if true_unsafe.sum() > 0 else 0.0

    return {
        "mIoU": miou,
        "pixel_acc": pixel_acc,
        "iou_safe": ious.get("iou_class_0", 0.0),
        "iou_unknown": ious.get("iou_class_1", 0.0),
        "iou_unsafe": ious.get("iou_class_2", 0.0),
        "unsafe_recall": unsafe_recall,
        "false_safe_ratio": false_safe_ratio
    }
