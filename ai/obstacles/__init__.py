"""Explicit Obstacle & Hazard Perception Module."""

from ai.obstacles.base import (
    HazardType,
    ObstacleDetection,
    ObstacleDetectorBase,
    RiskLevel,
)
from ai.obstacles.detector import UnifiedHazardDetector
from ai.obstacles.model import NeuralObjectDetector
from ai.obstacles.postprocess import apply_nms, build_composite_hazard_mask
from ai.obstacles.terrain_hazards import (
    LowDarkObstacleDetector,
    NegativeObstacleDetector,
    WaterReflectionDetector,
)
from ai.obstacles.visualization import draw_obstacle_detections

__all__ = [
    "HazardType",
    "RiskLevel",
    "ObstacleDetection",
    "ObstacleDetectorBase",
    "UnifiedHazardDetector",
    "NeuralObjectDetector",
    "WaterReflectionDetector",
    "LowDarkObstacleDetector",
    "NegativeObstacleDetector",
    "apply_nms",
    "build_composite_hazard_mask",
    "draw_obstacle_detections",
]
