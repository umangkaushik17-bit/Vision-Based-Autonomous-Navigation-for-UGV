"""Visualization utilities for obstacles and hazard detections."""

from __future__ import annotations

from typing import Dict, List, Tuple
import cv2
import numpy as np

from ai.obstacles.base import ObstacleDetection, RiskLevel

# BGR colors for risk levels
RISK_COLORS: Dict[RiskLevel, Tuple[int, int, int]] = {
    RiskLevel.CRITICAL: (0, 0, 255),    # Red
    RiskLevel.HIGH: (0, 140, 255),      # Orange
    RiskLevel.MEDIUM: (0, 220, 255),    # Yellow
    RiskLevel.LOW: (255, 180, 0),       # Cyan / Light Blue
    RiskLevel.UNKNOWN: (220, 0, 220)    # Purple
}


def draw_obstacle_detections(
    image: np.ndarray,
    detections: List[ObstacleDetection],
    draw_masks: bool = True
) -> np.ndarray:
    """Draw labeled bounding boxes, risk badges, and optional masks on an RGB image."""
    vis = image.copy()
    h, w = vis.shape[:2]

    # Overlay masks if requested
    if draw_masks:
        mask_overlay = np.zeros_like(vis)
        for det in detections:
            color = RISK_COLORS.get(det.risk_level, (0, 0, 255))
            if det.mask is not None:
                mask_overlay[det.mask] = color
            else:
                x1, y1, x2, y2 = det.bbox
                mask_overlay[y1:y2, x1:x2] = color

        # Alpha blend mask overlay
        has_any_mask = (mask_overlay.sum(axis=-1) > 0)
        vis[has_any_mask] = cv2.addWeighted(vis, 0.65, mask_overlay, 0.35, 0)[has_any_mask]

    # Draw bounding boxes and text badges
    for det in detections:
        color = RISK_COLORS.get(det.risk_level, (0, 0, 255))
        x1, y1, x2, y2 = det.bbox

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label text
        label = f"{det.hazard_type.value}: {det.confidence:.2f} ({det.risk_level.value})"
        (txt_w, txt_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        # Header background badge
        badge_y1 = max(0, y1 - txt_h - 6)
        badge_y2 = y1
        badge_x2 = min(w, x1 + txt_w + 6)
        cv2.rectangle(vis, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

        # Text in white/black for contrast
        text_color = (0, 0, 0) if det.risk_level in [RiskLevel.MEDIUM, RiskLevel.LOW] else (255, 255, 255)
        cv2.putText(vis, label, (x1 + 3, badge_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    # Draw summary HUD in top left corner
    hud_text = f"Hazards Detected: {len(detections)}"
    cv2.putText(vis, hud_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return vis
