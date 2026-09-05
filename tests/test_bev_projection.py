"""Unit tests for BEV projection, camera geometry, and depth source abstractions."""

import numpy as np
import pytest

from ai.traversability.bev_projection import BEVProjector, BEVTraversabilityMap
from ai.traversability.depth_source import (
    EstimatedMonocularDepthSource,
    PlanarGroundDepthSource,
    SimulatedDepthSource
)
from ai.traversability.geometry import CameraExtrinsics, CameraIntrinsics


def test_camera_intrinsics():
    """Verify camera intrinsic matrix and inverse inversion."""
    intrinsics = CameraIntrinsics.default_640x480(fov_deg=60.0)
    assert intrinsics.width == 640
    assert intrinsics.height == 480
    assert intrinsics.K.shape == (3, 3)

    # K @ K_inv must equal 3x3 identity
    eye = intrinsics.K @ intrinsics.K_inv
    np.testing.assert_allclose(eye, np.eye(3), atol=1e-4)


def test_camera_extrinsics():
    """Verify camera extrinsics rotation matrix properties."""
    extrinsics = CameraExtrinsics(pitch_deg=20.0, height_m=0.40)
    R = extrinsics.rotation_matrix
    assert R.shape == (3, 3)
    # Rotation matrices are orthogonal: R @ R.T = I
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-4)


def test_planar_ground_depth_source():
    """Verify analytical ground depth calculation produces physically realistic distances."""
    intrinsics = CameraIntrinsics.default_640x480()
    extrinsics = CameraExtrinsics(pitch_deg=25.0, height_m=0.40)
    depth_src = PlanarGroundDepthSource(intrinsics, extrinsics, max_depth_m=10.0)

    depth = depth_src.get_depth_meters((480, 640))
    assert depth.shape == (480, 640)

    # Bottom pixels (closest ground in front of rover) should have small positive depth
    bottom_depth = depth[479, 320]
    assert 0.1 < bottom_depth < 2.0

    # Top pixels looking toward horizon or sky should be nan or greater than bottom
    mid_depth = depth[250, 320]
    assert np.isnan(mid_depth) or mid_depth > bottom_depth


def test_bev_projector_mapping():
    """Verify coordinate mapping round-trip."""
    bev_map = BEVTraversabilityMap(
        grid=np.ones((40, 40), dtype=np.float32),
        unknown_mask=np.zeros((40, 40), dtype=bool),
        resolution_m=0.1,
        x_range_m=(0.5, 4.5),
        y_range_m=(-2.0, 2.0)
    )
    row, col = bev_map.meter_to_grid(2.5, 0.0)
    assert 0 <= row < 40
    assert 0 <= col < 40

    x_m, y_m = bev_map.grid_to_meter(row, col)
    assert np.isclose(x_m, 2.5, atol=0.1)
    assert np.isclose(y_m, 0.0, atol=0.1)


def test_bev_projector_execution():
    """Verify full projection execution on synthetic probability mask."""
    projector = BEVProjector(
        resolution_m=0.1,
        x_range_m=(0.5, 3.5),
        y_range_m=(-1.5, 1.5)
    )

    # 480x640 mask: uniform 0.9 safe ground
    mask = np.full((480, 640), 0.9, dtype=np.float32)

    bev_map = projector.project(mask)

    assert isinstance(bev_map, BEVTraversabilityMap)
    assert bev_map.shape == (30, 30)
    assert bev_map.grid.dtype == np.float32

    # Observed cells should reflect the safe scores
    known = ~bev_map.unknown_mask
    assert known.any()
    assert bev_map.grid[known].mean() > 0.5
