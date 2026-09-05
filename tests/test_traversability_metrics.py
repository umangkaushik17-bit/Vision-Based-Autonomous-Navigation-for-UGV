"""Unit tests for segmentation metrics and false-safe navigation safety calculations."""

import numpy as np
import pytest

from ai.datasets.base import TraversabilityTargetClass
from ai.traversability.metrics import calculate_segmentation_metrics, calculate_navigation_safety_metrics

def test_segmentation_metrics_perfect_prediction():
    targets = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    preds = np.array([[0, 1], [2, 0]], dtype=np.uint8) # The 255 position is ignored

    metrics = calculate_segmentation_metrics(preds, targets, num_classes=3, ignore_index=255)
    assert metrics["mIoU"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0

def test_false_safe_calculation():
    # Ground truth: 10 UNSAFE pixels (class 2)
    # Predicted: 3 SAFE (class 0), 7 UNSAFE (class 2)
    targets = np.full((10, 1), TraversabilityTargetClass.UNSAFE, dtype=np.uint8)
    preds = np.full((10, 1), TraversabilityTargetClass.UNSAFE, dtype=np.uint8)
    preds[:3, 0] = TraversabilityTargetClass.SAFE # 3 False-Safe errors

    safety = calculate_navigation_safety_metrics(preds, targets, ignore_index=255)
    assert pytest.approx(safety["false_safe_pixel_ratio"], 0.01) == 0.30
    assert pytest.approx(safety["unsafe_recall"], 0.01) == 0.70
