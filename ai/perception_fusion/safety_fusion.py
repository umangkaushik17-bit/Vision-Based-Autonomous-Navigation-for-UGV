"""Safety-Aware Perception Fusion Engine.

Fuses:
1. Model A (SAM-TP) learned traversability
2. Model B (RUGD) learned traversability
3. Explicit Obstacle & Terrain Hazard Detections
4. Configurable Hazard Policy
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from ai.obstacles.base import HazardType, ObstacleDetection, RiskLevel
from ai.obstacles.postprocess import build_composite_hazard_mask
from ai.obstacles.visualization import draw_obstacle_detections
from ai.perception_fusion.hazard_policy import HazardPolicy
from ai.traversability.base import TraversabilityResult


@dataclass
class SafetyAwarePerceptionResult:
    """Consolidated safety-aware perception output for downstream costmaps and planning."""

    final_traversability: np.ndarray  # (H, W) float32 [0.0, 1.0], 1.0 = SAFE
    hazard_mask: np.ndarray           # (H, W) bool, True = active obstacle/hazard
    hazard_classes: List[str]         # Unique list of detected hazard names
    risk_map: np.ndarray              # (H, W) float32 [0.0, 1.0], 1.0 = LETHAL
    unknown_mask: np.ndarray          # (H, W) bool, True = uncertain / disagreement
    confidence_map: np.ndarray        # (H, W) float32 [0.0, 1.0], 1.0 = HIGH CONFIDENCE
    detections: List[ObstacleDetection]
    overlay: np.ndarray               # (H, W, 3) RGB visual overlay
    latency_ms: float
    result_a: Optional[TraversabilityResult] = None
    result_b: Optional[TraversabilityResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def safe_pixel_ratio(self) -> float:
        """Percentage of pixels evaluated as drivable."""
        return float(np.mean(self.final_traversability >= 0.5))

    @property
    def hazard_pixel_ratio(self) -> float:
        """Percentage of pixels covered by explicit obstacles/hazards."""
        return float(np.mean(self.hazard_mask))

    @property
    def unknown_pixel_ratio(self) -> float:
        """Percentage of pixels classified as UNKNOWN."""
        return float(np.mean(self.unknown_mask))


class SafetyPerceptionFusion:
    """Fuses dual traversability predictions with explicit hazard perception."""

    def __init__(
        self,
        policy: Optional[HazardPolicy] = None,
        disagreement_threshold: float = 0.40,
        weight_a: float = 0.50,
        weight_b: float = 0.50
    ) -> None:
        self.policy = policy or HazardPolicy()
        self.disagreement_threshold = disagreement_threshold
        self.weight_a = weight_a
        self.weight_b = weight_b

    def fuse(
        self,
        image: np.ndarray,
        result_a: Optional[TraversabilityResult],
        result_b: Optional[TraversabilityResult],
        detections: List[ObstacleDetection]
    ) -> SafetyAwarePerceptionResult:
        """Execute multi-source safety fusion."""
        t0 = time.perf_counter()
        h, w = image.shape[:2]

        # 1. Align Base Traversability Maps
        if result_a is not None and result_b is not None:
            mask_a = result_a.mask
            mask_b = result_b.mask
            conf_a = result_a.confidence if result_a.confidence is not None else np.ones((h, w), dtype=np.float32)
            conf_b = result_b.confidence if result_b.confidence is not None else np.ones((h, w), dtype=np.float32)

            # Check model disagreement
            diff = np.abs(mask_a - mask_b)
            unknown_mask = (diff > self.disagreement_threshold)

            # Weighted average where models roughly agree
            base_trav = (self.weight_a * mask_a + self.weight_b * mask_b)
            # In disagreement regions, penalize traversability towards UNKNOWN (0.50) or conservative minimum
            base_trav[unknown_mask] = np.minimum(mask_a[unknown_mask], mask_b[unknown_mask]) * 0.70

            base_confidence = (conf_a + conf_b) / 2.0
            base_confidence[unknown_mask] *= 0.40  # Confidence drops in disagreement zones

        elif result_a is not None:
            base_trav = result_a.mask.copy()
            unknown_mask = result_a.unknown_mask if result_a.unknown_mask is not None else np.zeros((h, w), dtype=bool)
            base_confidence = result_a.confidence if result_a.confidence is not None else np.ones((h, w), dtype=np.float32)
        elif result_b is not None:
            base_trav = result_b.mask.copy()
            unknown_mask = result_b.unknown_mask if result_b.unknown_mask is not None else np.zeros((h, w), dtype=bool)
            base_confidence = result_b.confidence if result_b.confidence is not None else np.ones((h, w), dtype=np.float32)
        else:
            base_trav = np.full((h, w), 0.50, dtype=np.float32)
            unknown_mask = np.ones((h, w), dtype=bool)
            base_confidence = np.zeros((h, w), dtype=np.float32)

        # 2. Process Explicit Obstacle & Hazard Detections
        final_trav = base_trav.copy()
        hazard_mask = np.zeros((h, w), dtype=bool)
        risk_map = np.zeros((h, w), dtype=np.float32)
        detected_hazard_classes = set()

        for det in detections:
            det_mask = det.mask if det.mask is not None else np.zeros((h, w), dtype=bool)
            if det.mask is None:
                x1, y1, x2, y2 = det.bbox
                det_mask[y1:y2, x1:x2] = True

            hazard_mask |= det_mask
            detected_hazard_classes.add(det.hazard_type.value)
            risk_cost = self.policy.get_risk_cost(det.risk_level)

            # Override traversability based on risk tier
            if det.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                # Critical / High: Hard obstacle override (traversability forced to 0.0)
                final_trav[det_mask] = 0.0
                risk_map[det_mask] = np.maximum(risk_map[det_mask], risk_cost)
                base_confidence[det_mask] = max(0.85, det.confidence)
            elif det.risk_level == RiskLevel.MEDIUM:
                final_trav[det_mask] = np.minimum(final_trav[det_mask], 0.35)
                risk_map[det_mask] = np.maximum(risk_map[det_mask], 0.50)
            elif det.risk_level == RiskLevel.UNKNOWN:
                unknown_mask |= det_mask
                final_trav[det_mask] = np.minimum(final_trav[det_mask], 0.40)
                risk_map[det_mask] = np.maximum(risk_map[det_mask], 0.50)

        # 3. Derive Background Risk Map from Traversability
        # Risk = 1.0 - traversability, merged with active obstacle risk
        background_risk = 1.0 - final_trav
        risk_map = np.maximum(risk_map, background_risk)

        # 4. Generate Composite Safety HUD Overlay
        overlay = self._create_safety_hud(image, final_trav, hazard_mask, unknown_mask, detections)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return SafetyAwarePerceptionResult(
            final_traversability=final_trav.astype(np.float32),
            hazard_mask=hazard_mask,
            hazard_classes=sorted(list(detected_hazard_classes)),
            risk_map=risk_map.astype(np.float32),
            unknown_mask=unknown_mask,
            confidence_map=base_confidence.astype(np.float32),
            detections=detections,
            overlay=overlay,
            latency_ms=latency_ms,
            result_a=result_a,
            result_b=result_b,
            metadata={
                "disagreement_threshold": self.disagreement_threshold,
                "hazard_count": len(detections),
                "has_model_a": result_a is not None,
                "has_model_b": result_b is not None
            }
        )

    def _create_safety_hud(
        self,
        image: np.ndarray,
        traversability: np.ndarray,
        hazard_mask: np.ndarray,
        unknown_mask: np.ndarray,
        detections: List[ObstacleDetection]
    ) -> np.ndarray:
        """Render multi-layer safety HUD overlay."""
        h, w = image.shape[:2]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Drivable safe zones in green
        safe_zone = (traversability >= 0.5) & (~hazard_mask) & (~unknown_mask)
        color_mask[safe_zone] = [0, 200, 0]

        # Unknown / uncertainty zones in yellow
        color_mask[unknown_mask & (~hazard_mask)] = [255, 200, 0]

        # Active hazards / impassable zones in red
        color_mask[hazard_mask] = [220, 0, 0]

        # Blend base mask
        blended = cv2.addWeighted(image, 0.60, color_mask, 0.40, 0)

        # Draw labeled bounding boxes on top
        vis = draw_obstacle_detections(blended, detections, draw_masks=False)

        # Draw HUD safety status badge
        mean_trav = float(traversability.mean())
        if len(detections) > 0 and any(d.risk_level == RiskLevel.CRITICAL for d in detections):
            status_text = "STATUS: CRITICAL HAZARD DETECTED"
            badge_color = (0, 0, 255)
        elif unknown_mask.mean() > 0.20:
            status_text = "STATUS: HIGH UNCERTAINTY (CAUTION)"
            badge_color = (0, 200, 255)
        else:
            status_text = f"STATUS: NOMINAL (Drivability: {mean_trav:.2f})"
            badge_color = (0, 200, 0)

        cv2.rectangle(vis, (10, h - 45), (420, h - 10), (0, 0, 0), -1)
        cv2.putText(vis, status_text, (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_color, 2, cv2.LINE_AA)

        return vis
