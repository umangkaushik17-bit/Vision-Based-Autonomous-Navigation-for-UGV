"""Unit tests for Obstacle and Hazard Detection subsystems."""

import pytest
import numpy as np

from ai.obstacles.base import HazardType, ObstacleDetection, RiskLevel
from ai.obstacles.detector import UnifiedHazardDetector
from ai.obstacles.postprocess import apply_nms, compute_iou, build_composite_hazard_mask
from ai.obstacles.terrain_hazards import LowDarkObstacleDetector, WaterReflectionDetector


def test_obstacle_detection_dataclass():
    """Verify ObstacleDetection properties, areas, and serialization."""
    det = ObstacleDetection(
        class_name="rock",
        confidence=0.88,
        bbox=(100, 150, 180, 210),
        hazard_type=HazardType.ROCK,
        risk_level=RiskLevel.HIGH,
        source_model="test_model"
    )

    assert det.width == 80
    assert det.height == 60
    assert det.area == 4800
    assert det.center == (140, 180)

    d = det.to_dict()
    assert d["class_name"] == "rock"
    assert d["confidence"] == 0.88
    assert d["hazard_type"] == "ROCK"
    assert d["risk_level"] == "HIGH"


def test_iou_and_nms():
    """Verify box IoU calculation and Non-Maximum Suppression."""
    box_a = (10, 10, 50, 50)
    box_b = (10, 10, 50, 50)  # Identical -> IoU = 1.0
    box_c = (100, 100, 150, 150)  # Disjoint -> IoU = 0.0

    assert compute_iou(box_a, box_b) == 1.0
    assert compute_iou(box_a, box_c) == 0.0

    # Overlapping detections with different confidences
    det1 = ObstacleDetection("rock", 0.90, (10, 10, 50, 50), HazardType.ROCK, RiskLevel.HIGH, "m")
    det2 = ObstacleDetection("rock", 0.70, (15, 15, 55, 55), HazardType.ROCK, RiskLevel.HIGH, "m")
    det3 = ObstacleDetection("rock", 0.85, (100, 100, 150, 150), HazardType.ROCK, RiskLevel.HIGH, "m")

    kept = apply_nms([det1, det2, det3], iou_threshold=0.5)
    assert len(kept) == 2
    assert kept[0].confidence == 0.90
    assert kept[1].confidence == 0.85


def test_water_reflection_detector():
    """Verify detection of specular reflection water cues."""
    detector = WaterReflectionDetector(min_water_area=50)

    # Create dummy image with a bright specular reflection spot in the ground region
    img = np.full((480, 640, 3), 100, dtype=np.uint8)
    # Bright highlight patch (water reflection: high brightness, low saturation)
    img[300:360, 200:300] = [245, 245, 255]

    dets = detector.detect(img)
    assert len(dets) >= 1
    assert any(d.hazard_type == HazardType.WATER for d in dets)
    assert any(d.risk_level == RiskLevel.CRITICAL for d in dets)


def test_low_dark_obstacle_detector():
    """Verify detection of anomalous dark rock on light ground."""
    detector = LowDarkObstacleDetector(min_area=50)

    # Create bright light terrain
    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    # Insert dark obstacle patch (e.g. dark boulder)
    img[320:360, 250:300] = [40, 40, 40]

    dets = detector.detect(img)
    assert len(dets) >= 1
    assert any(d.hazard_type in [HazardType.ROCK, HazardType.STEP] for d in dets)


def test_unified_hazard_detector():
    """Verify unified detector integrates all components."""
    detector = UnifiedHazardDetector()
    info = detector.get_detector_info()
    assert info["detector_name"] == "UnifiedHazardDetector"
    assert info["has_water_analyzer"] is True
    assert info["has_dark_obstacle_analyzer"] is True

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = detector.detect(dummy_img)
    assert isinstance(dets, list)
