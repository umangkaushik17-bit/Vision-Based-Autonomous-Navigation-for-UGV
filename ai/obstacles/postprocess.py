"""Postprocessing utilities for obstacle detection and hazard extraction."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from ai.obstacles.base import ObstacleDetection


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union_area = float(area_a + area_b - inter_area)

    if union_area <= 0.0:
        return 0.0
    return float(inter_area) / union_area


def apply_nms(
    detections: List[ObstacleDetection],
    iou_threshold: float = 0.45
) -> List[ObstacleDetection]:
    """Perform Non-Maximum Suppression (NMS) over a list of detections."""
    if not detections:
        return []

    # Sort descending by confidence
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept_dets: List[ObstacleDetection] = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        kept_dets.append(best)

        # Filter out remaining detections that overlap too heavily with best
        filtered = []
        for det in sorted_dets:
            iou = compute_iou(best.bbox, det.bbox)
            if iou < iou_threshold:
                filtered.append(det)
        sorted_dets = filtered

    return kept_dets


def build_composite_hazard_mask(
    detections: List[ObstacleDetection],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """Render a binary 2D hazard mask (H, W) from a list of detections."""
    h, w = image_shape[:2]
    hazard_mask = np.zeros((h, w), dtype=bool)

    for det in detections:
        if det.mask is not None:
            hazard_mask |= det.mask
        else:
            x1, y1, x2, y2 = det.bbox
            hazard_mask[y1:y2, x1:x2] = True

    return hazard_mask
