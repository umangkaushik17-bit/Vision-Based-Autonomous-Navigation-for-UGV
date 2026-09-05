"""Unified Obstacle and Hazard Detector.

Combines neural discrete object detection with specialized terrain hazard analyzers
(water reflection, low/dark obstacles, negative ditch edges) into a unified detector.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

from ai.obstacles.base import ObstacleDetection, ObstacleDetectorBase
from ai.obstacles.model import NeuralObjectDetector
from ai.obstacles.postprocess import apply_nms
from ai.obstacles.terrain_hazards import LowDarkObstacleDetector, NegativeObstacleDetector, WaterReflectionDetector
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.obstacles.detector")


class UnifiedHazardDetector(ObstacleDetectorBase):
    """Unified detector coordinating object models and terrain hazard extractors."""

    def __init__(
        self,
        enable_neural: bool = True,
        enable_water: bool = True,
        enable_dark_obstacles: bool = True,
        enable_ditches: bool = True,
        nms_threshold: float = 0.45,
        device: Optional[str] = None
    ) -> None:
        self.nms_threshold = nms_threshold

        self.neural_detector = NeuralObjectDetector(device=device) if enable_neural else None
        self.water_detector = WaterReflectionDetector() if enable_water else None
        self.dark_detector = LowDarkObstacleDetector() if enable_dark_obstacles else None
        self.ditch_detector = NegativeObstacleDetector() if enable_ditches else None

        logger.info(
            f"UnifiedHazardDetector initialized: neural={enable_neural}, "
            f"water={enable_water}, dark_obs={enable_dark_obstacles}, ditches={enable_ditches}"
        )

    def detect(self, image: Union[np.ndarray, str, Path]) -> List[ObstacleDetection]:
        """Detect all hazards in the given image."""
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                return []
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            rgb = image
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        all_detections: List[ObstacleDetection] = []

        # 1. Neural discrete object detection (persons, vehicles, large obstacles)
        if self.neural_detector is not None:
            try:
                all_detections.extend(self.neural_detector.detect(rgb))
            except Exception as e:
                logger.warning(f"Neural detector failed: {e}")

        # 2. Water and reflection detection (critical creek safety)
        if self.water_detector is not None:
            try:
                all_detections.extend(self.water_detector.detect(rgb))
            except Exception as e:
                logger.warning(f"Water detector failed: {e}")

        # 3. Low/dark protruding obstacle extraction
        if self.dark_detector is not None:
            try:
                all_detections.extend(self.dark_detector.detect(rgb))
            except Exception as e:
                logger.warning(f"Dark obstacle detector failed: {e}")

        # 4. Negative obstacle (ditch/pothole) extraction
        if self.ditch_detector is not None:
            try:
                all_detections.extend(self.ditch_detector.detect(rgb))
            except Exception as e:
                logger.warning(f"Negative obstacle detector failed: {e}")

        # 5. Apply Non-Maximum Suppression to eliminate duplicate bounding boxes
        final_detections = apply_nms(all_detections, iou_threshold=self.nms_threshold)
        return final_detections

    def get_detector_info(self) -> Dict[str, Any]:
        """Return system configuration."""
        return {
            "detector_name": "UnifiedHazardDetector",
            "has_neural": self.neural_detector is not None,
            "has_water_analyzer": self.water_detector is not None,
            "has_dark_obstacle_analyzer": self.dark_detector is not None,
            "has_ditch_analyzer": self.ditch_detector is not None,
            "nms_threshold": self.nms_threshold
        }
