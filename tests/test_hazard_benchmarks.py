"""Unit tests verifying measurable false-safe error reduction under hazard stress."""

import pytest
import numpy as np

from ai.obstacles.base import HazardType, ObstacleDetection, RiskLevel
from ai.obstacles.terrain_hazards import LowDarkObstacleDetector, WaterReflectionDetector
from ai.perception_fusion.safety_fusion import SafetyPerceptionFusion
from ai.traversability.base import TraversabilityResult


def test_false_safe_reduction_on_water():
    """Verify that hazard perception overrides false-safe prediction on water reflections."""
    # Synthetic frame where lower half has a reflective water pool
    h, w = 480, 640
    rgb = np.full((h, w, 3), 110, dtype=np.uint8)
    # Bright specular water pool
    rgb[300:400, 200:400] = [245, 245, 255]

    # Ground truth: 0=SAFE, 2=UNSAFE (water pool is UNSAFE = 2)
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    gt_mask[300:400, 200:400] = 2

    # Simulated SAM-TP failure: mistakenly predicts 0.95 (SAFE) on the water pool!
    fake_sam_tp_mask = np.full((h, w), 0.90, dtype=np.float32)
    res_a = TraversabilityResult(
        mask=fake_sam_tp_mask,
        logits=fake_sam_tp_mask,
        overlay=rgb,
        image=rgb,
        input_shape=(h, w),
        inference_s=0.1,
        device="cpu",
        model_name="fake_sam_tp"
    )

    # Calculate SAM-TP alone False-Safe error on the water pool
    water_pixels = (gt_mask == 2)
    sam_tp_false_safe_pixels = (res_a.mask[water_pixels] >= 0.5).sum()
    sam_tp_false_safe_rate = float(sam_tp_false_safe_pixels) / float(water_pixels.sum())
    assert sam_tp_false_safe_rate == 1.0  # 100% false-safe error!

    # Now apply water detector and fusion
    water_detector = WaterReflectionDetector(min_water_area=100)
    dets = water_detector.detect(rgb)
    assert len(dets) >= 1

    fusion = SafetyPerceptionFusion()
    fused = fusion.fuse(rgb, res_a, None, dets)

    # After fusion, water pool traversability MUST be 0.0
    fused_false_safe_pixels = (fused.final_traversability[water_pixels] >= 0.5).sum()
    fused_false_safe_rate = float(fused_false_safe_pixels) / float(water_pixels.sum())

    # Verify dramatic false-safe error reduction
    assert fused_false_safe_rate < 0.05
    assert fused.hazard_mask[water_pixels].mean() > 0.90


def test_false_safe_reduction_on_dark_rock():
    """Verify that low-dark obstacle detection overrides false-safe prediction on rocks."""
    h, w = 480, 640
    # Bright sunlit dirt trail
    rgb = np.full((h, w, 3), 190, dtype=np.uint8)
    # Dark rock sitting on trail
    rgb[350:390, 280:330] = [35, 35, 35]

    # Ground truth: rock is UNSAFE (2)
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    gt_mask[350:390, 280:330] = 2

    # Simulated flawed model predicting SAFE on the rock
    fake_mask = np.full((h, w), 0.85, dtype=np.float32)
    res_model = TraversabilityResult(
        mask=fake_mask,
        logits=fake_mask,
        overlay=rgb,
        image=rgb,
        input_shape=(h, w),
        inference_s=0.1,
        device="cpu",
        model_name="flawed_model"
    )

    rock_pixels = (gt_mask == 2)
    assert (res_model.mask[rock_pixels] >= 0.5).all()

    # Detect dark obstacle
    dark_detector = LowDarkObstacleDetector(min_area=50)
    dets = dark_detector.detect(rgb)
    assert len(dets) >= 1

    fusion = SafetyPerceptionFusion()
    fused = fusion.fuse(rgb, res_model, None, dets)

    # Verify rock region is neutralized to 0.0
    fused_false_safe_pixels = (fused.final_traversability[rock_pixels] >= 0.5).sum()
    assert fused_false_safe_pixels == 0
    assert fused.risk_map[rock_pixels].mean() >= 0.85
