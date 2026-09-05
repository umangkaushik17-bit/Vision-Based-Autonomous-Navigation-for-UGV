"""Unit tests for non-deep-learning classical computer vision baseline."""

import numpy as np
import pytest

from ai.datasets.base import TraversabilityTargetClass
from ai.traversability.baseline_classifier import BaselineTerrainClassifier

def test_baseline_classifier_shapes_and_classes():
    classifier = BaselineTerrainClassifier()
    dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    pred = classifier.predict(dummy_bgr)
    assert pred.shape == (480, 640)
    assert pred.dtype == np.uint8

    unique_classes = set(np.unique(pred))
    allowed_classes = {
        TraversabilityTargetClass.SAFE,
        TraversabilityTargetClass.UNKNOWN,
        TraversabilityTargetClass.UNSAFE,
        TraversabilityTargetClass.VOID
    }
    assert unique_classes.issubset(allowed_classes)

def test_baseline_sky_detection():
    classifier = BaselineTerrainClassifier()
    dummy_bgr = np.full((480, 640, 3), 240, dtype=np.uint8) # Bright sky-like image

    pred = classifier.predict(dummy_bgr)
    # Top section should contain VOID pixels
    horizon_y = int(480 * 0.35)
    assert np.any(pred[:horizon_y, :] == TraversabilityTargetClass.VOID)
