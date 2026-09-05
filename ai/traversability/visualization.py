"""Visualization utilities for traversability probability maps, overlays, and evaluation error maps."""

from __future__ import annotations

import cv2
import matplotlib.cm as cm
import numpy as np


def create_traversability_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45
) -> np.ndarray:
    """Blend an RGB image with green (drivable) and red (non-drivable) color overlay.

    Args:
        image: RGB frame (H, W, 3) uint8.
        mask: Continuous traversability mask (H, W) float32 in [0.0, 1.0].
        alpha: Blending weight of the colored overlay [0.0, 1.0].

    Returns:
        Blended RGB frame (H, W, 3) uint8.
    """
    m = np.clip(mask, 0.0, 1.0)[..., None]
    # Green for drivable (m -> 1), Red for non-drivable (m -> 0)
    color = np.concatenate([255.0 * (1.0 - m), 255.0 * m, np.zeros_like(m)], axis=-1)
    blended = (1.0 - alpha) * image.astype(np.float32) + alpha * color
    return np.clip(blended, 0, 255).astype(np.uint8)


def create_heatmap(prob_map: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    """Convert a 2D float probability map into a vivid RGB heatmap.

    Args:
        prob_map: Float array (H, W) in [0.0, 1.0].
        colormap: Matplotlib colormap name ('turbo', 'jet', 'viridis').

    Returns:
        RGB heatmap array (H, W, 3) uint8.
    """
    normalized = np.clip(prob_map, 0.0, 1.0)
    rgba = cm.get_cmap(colormap)(normalized)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def create_difference_map(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
    ignore_index: int = 255
) -> np.ndarray:
    """Generate visual safety audit difference map comparing predictions against ground truth.

    Color coding:
      - GREEN: True Positive (Correctly classified SAFE)
      - DARK GRAY: True Negative (Correctly classified OBSTACLE/UNSAFE)
      - BRIGHT RED: FALSE-SAFE ERROR (CRITICAL HAZARD - Ground truth was UNSAFE, predicted SAFE!)
      - YELLOW: FALSE-UNSAFE ERROR (Pessimistic - Ground truth was SAFE, predicted UNSAFE)
      - BLACK: Ignored / Void pixels (sky, unannotated)

    Args:
        pred_mask: Predicted continuous probability (H, W) in [0.0, 1.0].
        gt_mask: Ground truth integer mask (H, W), where:
                 0 = SAFE, 1 = UNKNOWN/UNSURE, 2 = UNSAFE/HAZARD, 255 = VOID.
                 OR binary mask where 1 = SAFE, 0 = UNSAFE.
        threshold: Decision threshold for predicted traversability.
        ignore_index: Void index to exclude from error coloring.

    Returns:
        RGB difference map (H, W, 3) uint8.
    """
    h, w = pred_mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)

    pred_safe = pred_mask >= threshold

    # Determine GT safe vs unsafe based on 3-class or binary GT convention
    if gt_mask.max() > 1:
        valid = gt_mask != ignore_index
        gt_safe = (gt_mask == 0) & valid
        gt_unsafe = (gt_mask == 2) & valid
    else:
        valid = np.ones((h, w), dtype=bool)
        gt_safe = (gt_mask == 1) & valid
        gt_unsafe = (gt_mask == 0) & valid

    # 1. True Safe (Green)
    tp = pred_safe & gt_safe
    out[tp] = [40, 200, 40]

    # 2. True Unsafe (Dark Gray)
    tn = (~pred_safe) & gt_unsafe
    out[tn] = [60, 60, 60]

    # 3. False-Safe: Predicted SAFE when actual is UNSAFE (CRITICAL SAFETY ERROR -> BRIGHT RED)
    false_safe = pred_safe & gt_unsafe
    out[false_safe] = [255, 20, 20]

    # 4. False-Unsafe: Predicted UNSAFE when actual is SAFE (Pessimistic -> YELLOW)
    false_unsafe = (~pred_safe) & gt_safe
    out[false_unsafe] = [255, 215, 0]

    # 5. Void / Unknown (Dark Blue)
    void = ~valid
    out[void] = [20, 20, 40]

    return out
