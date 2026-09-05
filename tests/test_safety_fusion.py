"""Unit tests for Safety Perception Fusion Engine."""

import pytest
import numpy as np

from ai.obstacles.base import HazardType, ObstacleDetection, RiskLevel
from ai.perception_fusion.safety_fusion import SafetyPerceptionFusion, SafetyAwarePerceptionResult
from ai.traversability.base import TraversabilityResult


@pytest.fixture
def fusion():
    return SafetyPerceptionFusion()


def make_dummy_result(prob_val: float, h: int = 100, w: int = 100) -> TraversabilityResult:
    """Helper to create dummy TraversabilityResult."""
    mask = np.full((h, w), prob_val, dtype=np.float32)
    logits = np.zeros((h, w), dtype=np.float32)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    image = np.zeros((h, w, 3), dtype=np.uint8)
    return TraversabilityResult(
        mask=mask,
        logits=logits,
        overlay=overlay,
        image=image,
        input_shape=(h, w),
        inference_s=0.01,
        device="cpu",
        model_name="dummy"
    )


def test_safety_fusion_agreement_safe(fusion):
    """When both models agree on SAFE and no hazard is present, result is SAFE."""
    res_a = make_dummy_result(0.90)
    res_b = make_dummy_result(0.85)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    fused = fusion.fuse(rgb, res_a, res_b, detections=[])

    assert isinstance(fused, SafetyAwarePerceptionResult)
    assert fused.safe_pixel_ratio == 1.0
    assert fused.hazard_pixel_ratio == 0.0
    assert fused.unknown_pixel_ratio == 0.0
    assert fused.final_traversability.mean() > 0.80
    assert fused.risk_map.mean() < 0.20


def test_safety_fusion_disagreement_as_unknown(fusion):
    """When Model A says SAFE (0.90) and Model B says UNSAFE (0.10), disagreement is flagged as UNKNOWN."""
    res_a = make_dummy_result(0.90)
    res_b = make_dummy_result(0.10)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    fused = fusion.fuse(rgb, res_a, res_b, detections=[])

    # Disagreement is NOT forced to SAFE!
    assert fused.unknown_mask.all()
    assert fused.unknown_pixel_ratio == 1.0
    assert fused.safe_pixel_ratio == 0.0  # Conservatively penalized
    assert fused.final_traversability.mean() <= 0.50


def test_safety_fusion_obstacle_override(fusion):
    """When an obstacle is detected, it overrides traversability to 0.0 (UNSAFE) even if both models say SAFE."""
    # Both models falsely predict SAFE (e.g. water reflection failure)
    res_a = make_dummy_result(0.95)
    res_b = make_dummy_result(0.95)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    # Obstacle detector finds a critical water hazard in a 40x40 patch
    hazard_mask = np.zeros((100, 100), dtype=bool)
    hazard_mask[30:70, 30:70] = True
    det = ObstacleDetection(
        class_name="water_reflection",
        confidence=0.99,
        bbox=(30, 30, 70, 70),
        hazard_type=HazardType.WATER,
        risk_level=RiskLevel.CRITICAL,
        source_model="water_detector",
        mask=hazard_mask
    )

    fused = fusion.fuse(rgb, res_a, res_b, detections=[det])

    # In the hazard patch, traversability MUST be forced to 0.0
    assert (fused.final_traversability[30:70, 30:70] == 0.0).all()
    assert (fused.risk_map[30:70, 30:70] == 1.0).all()
    assert fused.hazard_mask[30:70, 30:70].all()
    assert "WATER" in fused.hazard_classes


def test_safety_fusion_overlay_shape(fusion):
    """Verify visual overlay has correct shape and type."""
    res_a = make_dummy_result(0.70)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    fused = fusion.fuse(rgb, res_a, None, detections=[])
    assert fused.overlay.shape == (100, 100, 3)
    assert fused.overlay.dtype == np.uint8
