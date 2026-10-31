"""Tests for Phase 14.3 Camera Projection & Inverse Perspective Mapping (IPM).

Verifies camera intrinsics, extrinsics, analytical ground ray intersection,
metric scaling, frame transformations, and precomputed projection LUT.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.traversability.geometry import CameraExtrinsics, CameraIntrinsics
from ai.traversability.depth_source import PlanarGroundDepthSource
from production.perception.vision_costmap_integrator import (
    LocalBEVCostmap,
    VisionCostmapConfig,
    VisionCostmapIntegrator,
)


def test_simulation_intrinsics_calibration():
    """Verify camera intrinsics calibrated for Gazebo demo_outdoor camera."""
    # Gazebo camera: 640x480, HFOV = 1.3962634 rad (80.0 deg)
    hfov_rad = 1.3962634
    expected_fx = (640 / 2.0) / math.tan(hfov_rad / 2.0)

    integrator = VisionCostmapIntegrator.create_simulation_default()
    intr = integrator.config.intrinsics

    assert intr.width == 640
    assert intr.height == 480
    assert math.isclose(intr.fx, expected_fx, rel_tol=1e-3)
    assert math.isclose(intr.fy, expected_fx, rel_tol=1e-3)
    assert intr.cx == 320.0
    assert intr.cy == 240.0

    # Orthogonality of K @ K_inv
    eye = intr.K @ intr.K_inv
    np.testing.assert_allclose(eye, np.eye(3), atol=1e-4)


def test_simulation_extrinsics_and_rotation_matrix():
    """Verify camera extrinsics properties and rotation matrix validity."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    extr = integrator.config.extrinsics

    assert extr.pitch_deg == 15.0
    assert extr.height_m == 0.525
    assert extr.forward_offset_m == 0.15

    R = extr.rotation_matrix
    assert R.shape == (3, 3)
    # Rotation matrix must be orthogonal and det == 1.0
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_ground_plane_ray_intersection_accuracy():
    """Verify analytical ground plane intersection produces physically accurate metric coordinates."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    intr = integrator.config.intrinsics
    extr = integrator.config.extrinsics

    ds = PlanarGroundDepthSource(intr, extr)
    depth = ds.get_depth_meters((480, 640))

    # Center pixel (row=240, col=320) corresponds to optical axis
    # Optical axis is tilted downward by 15 deg from horizontal
    # Optical center height = 0.525m
    # Distance along ray to ground: z_c = height / sin(15 deg) = 0.525 / sin(0.2618) ~= 2.028m
    center_zc = depth[240, 320]
    expected_zc = 0.525 / math.sin(math.radians(15.0))
    assert math.isclose(center_zc, expected_zc, rel_tol=1e-3)

    # 3D coordinates in robot frame (base_link):
    # Forward distance = forward_offset + height / tan(15 deg) = 0.15 + 0.525 / tan(15 deg) ~= 2.109m
    expected_forward_x = 0.15 + 0.525 / math.tan(math.radians(15.0))
    R_opt_to_robot = extr.rotation_matrix.T
    P_cam = np.array([0.0, 0.0, center_zc], dtype=np.float32)
    t_robot = np.array([extr.forward_offset_m, 0.0, extr.height_m], dtype=np.float32)
    P_robot = R_opt_to_robot @ P_cam + t_robot

    assert math.isclose(P_robot[0], expected_forward_x, rel_tol=1e-3)
    assert math.isclose(P_robot[1], 0.0, abs_tol=1e-4)  # Lateral offset must be 0
    assert math.isclose(P_robot[2], 0.0, abs_tol=1e-4)  # Must intersect ground at Z = 0



def test_bottom_pixel_close_range():
    """Verify bottom of image corresponds to close ground in front of vehicle."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    intr = integrator.config.intrinsics
    extr = integrator.config.extrinsics
    ds = PlanarGroundDepthSource(intr, extr)
    depth = ds.get_depth_meters((480, 640))

    bottom_zc = depth[479, 320]
    assert 0.4 < bottom_zc < 1.0  # Close optical distance


def test_horizon_sky_culling():
    """Verify sky/horizon pixels above ground plane are marked as NaN (never ground)."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    intr = integrator.config.intrinsics
    extr = integrator.config.extrinsics
    ds = PlanarGroundDepthSource(intr, extr)
    depth = ds.get_depth_meters((480, 640))

    # Top pixels (sky)
    sky_zc = depth[50, 320]
    assert np.isnan(sky_zc)


def test_metric_bev_coordinate_roundtrip():
    """Verify local BEV grid conversion between metric coordinates and grid cells."""
    integrator = VisionCostmapIntegrator.create_simulation_default()

    # Query coordinate (x_forward = 3.0m, y_lateral = -1.2m)
    r, c = integrator.project_traversability.__self__._build_empty_costmap(
        0.0, True, ""
    ).meter_to_grid(3.0, -1.2)

    empty_bev = integrator._build_empty_costmap(0.0, True, "")
    x_rec, y_rec = empty_bev.grid_to_meter(r, c)

    assert math.isclose(x_rec, 3.0, abs_tol=integrator.config.resolution_m)
    assert math.isclose(y_rec, -1.2, abs_tol=integrator.config.resolution_m)


def test_precomputed_lut_shape_rescaling():
    """Verify precomputed projection adapts when input shape changes."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    assert integrator._lut_shape == (240, 320)

    # Trigger shape change to 480x640
    integrator.precompute_projection((480, 640))
    assert integrator._lut_shape == (480, 640)
    assert len(integrator._lut_rows) > 0
    assert len(integrator._lut_cols) > 0
