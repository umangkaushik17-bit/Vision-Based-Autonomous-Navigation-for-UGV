"""Unit tests for safety failure modes, contrast refinement, temporal filtering, and difference mapping."""

import numpy as np
import pytest

from ai.traversability.postprocess import (
    TemporalTraversabilityFilter,
    refine_traversability_by_contrast
)
from ai.traversability.visualization import (
    create_difference_map,
    create_heatmap,
    create_traversability_overlay
)


def test_contrast_refinement_downgrades_dark_obstacle():
    """Verify that a dark obstacle (e.g. black rock / shadow / ditch) on bright ground is downgraded."""
    h, w = 100, 100
    # Bright ground (RGB ~ 180)
    rgb = np.full((h, w, 3), 180, dtype=np.uint8)
    # Dark obstacle in center (RGB ~ 30)
    rgb[40:60, 40:60] = 30

    # SAM-TP mistakenly labeled everything as drivable (1.0)
    trav = np.full((h, w), 1.0, dtype=np.float32)

    refined = refine_traversability_by_contrast(
        trav=trav,
        rgb=rgb,
        drivable_thresh=0.5,
        darkness_ratio=0.65,
        min_reference_pixels=100
    )

    # Center obstacle must be downgraded to 0.0
    assert (refined[40:60, 40:60] == 0.0).all()
    # Surrounding bright ground should remain 1.0
    assert (refined[0:20, 0:20] == 1.0).all()


def test_contrast_refinement_preserves_uniform_ground():
    """Verify that uniform ground without dark anomalies is untouched."""
    h, w = 50, 50
    rgb = np.full((h, w, 3), 150, dtype=np.uint8)
    trav = np.full((h, w), 0.9, dtype=np.float32)

    refined = refine_traversability_by_contrast(
        trav=trav,
        rgb=rgb,
        min_reference_pixels=50
    )

    np.testing.assert_allclose(refined, trav)


def test_temporal_traversability_filter():
    """Verify exponential moving average dampens abrupt single-frame fluctuations."""
    filter_ema = TemporalTraversabilityFilter(alpha=0.6)

    frame1 = np.full((20, 20), 1.0, dtype=np.float32)
    s1 = filter_ema.update(frame1)
    np.testing.assert_allclose(s1, frame1)

    # Frame 2 has abrupt drop to 0.0 (flicker)
    frame2 = np.full((20, 20), 0.0, dtype=np.float32)
    s2 = filter_ema.update(frame2)

    # Expected: 0.6 * 0.0 + 0.4 * 1.0 = 0.4
    np.testing.assert_allclose(s2, 0.4, atol=1e-5)

    # Test reset
    filter_ema.reset()
    s3 = filter_ema.update(frame2)
    np.testing.assert_allclose(s3, frame2)


def test_difference_map_color_coding():
    """Verify audit difference map assigns correct safety classification colors."""
    pred = np.array([[0.9, 0.9], [0.1, 0.1]], dtype=np.float32)
    gt = np.array([[0, 2], [0, 2]], dtype=np.uint8)  # 0=safe, 2=unsafe

    diff = create_difference_map(pred, gt, threshold=0.5)

    assert diff.shape == (2, 2, 3)
    # [0, 0]: Pred safe (0.9), GT safe (0) -> True Safe (Green [40, 200, 40])
    np.testing.assert_array_equal(diff[0, 0], [40, 200, 40])

    # [0, 1]: Pred safe (0.9), GT unsafe (2) -> FALSE-SAFE HAZARD (Bright Red [255, 20, 20])
    np.testing.assert_array_equal(diff[0, 1], [255, 20, 20])

    # [1, 0]: Pred unsafe (0.1), GT safe (0) -> False-Unsafe Pessimistic (Yellow [255, 215, 0])
    np.testing.assert_array_equal(diff[1, 0], [255, 215, 0])

    # [1, 1]: Pred unsafe (0.1), GT unsafe (2) -> True Unsafe (Dark Gray [60, 60, 60])
    np.testing.assert_array_equal(diff[1, 1], [60, 60, 60])
