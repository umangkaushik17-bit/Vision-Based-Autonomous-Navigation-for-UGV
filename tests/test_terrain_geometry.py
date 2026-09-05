"""Unit tests for terrain geometry extraction, SVD plane fitting, slope, and flatness."""

import numpy as np
import pytest

from ai.perception_fusion.interfaces import TerrainCell


def test_terrain_cell_flat_plane():
    """Verify horizontal planar points yield slope ~ 0.0 and flatness ~ 0.0."""
    # 20 flat points at z = 0.0
    x, y = np.mgrid[-0.5:0.5:5j, -0.5:0.5:4j]
    pts = np.stack([x.flatten(), y.flatten(), np.zeros_like(x.flatten())], axis=-1)

    cell = TerrainCell.from_points(pts)

    assert cell.is_valid is True
    assert cell.point_count == 20
    assert np.isclose(cell.slope_deg, 0.0, atol=1e-3)
    assert np.isclose(cell.flatness, 0.0, atol=1e-5)
    assert np.isclose(cell.height_variation, 0.0, atol=1e-5)
    # Normal should point straight up [0, 0, 1]
    np.testing.assert_allclose(cell.normal, [0.0, 0.0, 1.0], atol=1e-3)


def test_terrain_cell_inclined_slope():
    """Verify points on an inclined ramp (45 degree incline) produce slope ~ 45.0 degrees."""
    x, y = np.mgrid[-0.5:0.5:5j, -0.5:0.5:4j]
    # z = x (45 deg incline along X)
    pts = np.stack([x.flatten(), y.flatten(), x.flatten()], axis=-1)

    cell = TerrainCell.from_points(pts)

    assert cell.is_valid is True
    assert np.isclose(cell.slope_deg, 45.0, atol=1e-1)
    assert np.isclose(cell.flatness, 0.0, atol=1e-5)


def test_terrain_cell_sparse_fallback():
    """Verify that insufficient points (< 3) gracefully degrade to invalid fallback cell."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # Only 2 points

    cell = TerrainCell.from_points(pts)

    assert cell.is_valid is False
    assert cell.point_count == 2
    assert cell.slope_deg == 0.0


def test_terrain_cell_roughness():
    """Verify noisy 3D points show non-zero flatness/roughness."""
    np.random.seed(42)
    x, y = np.mgrid[-0.5:0.5:10j, -0.5:0.5:10j]
    # Flat ground + random vertical roughness
    z = np.random.normal(0.0, 0.1, size=x.shape)
    pts = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

    cell = TerrainCell.from_points(pts)

    assert cell.is_valid is True
    assert cell.flatness > 0.01  # Measurable roughness
    assert cell.height_variation > 0.1
