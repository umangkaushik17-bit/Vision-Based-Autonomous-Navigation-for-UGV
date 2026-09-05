"""Tests A through H for SE(3) coordinate transformations in Phase 9."""

import numpy as np
import pytest

from ai.spatial_fusion.transforms import (
    SpatialCameraExtrinsics,
    create_se3_matrix,
    euler_to_rotation_matrix,
    invert_se3,
    rotation_matrix_to_euler,
    transform_points_3d,
)


def test_test_a_identity_transform():
    """Test A: Identity transform preserves coordinates."""
    T_eye = np.eye(4, dtype=np.float64)
    pts = np.array([[1.0, 2.0, 3.0], [-4.5, 0.0, 10.2]], dtype=np.float64)
    transformed = transform_points_3d(T_eye, pts)
    assert np.allclose(transformed, pts, atol=1e-12)


def test_test_b_pure_translation():
    """Test B: Known local point + known translation = known world point."""
    t = np.array([2.5, -1.0, 0.5], dtype=np.float64)
    T = create_se3_matrix(t=t)
    pt = np.array([[1.0, 3.0, 0.0]], dtype=np.float64)
    expected = np.array([[3.5, 2.0, 0.5]], dtype=np.float64)
    res = transform_points_3d(T, pt)
    assert np.allclose(res, expected, atol=1e-12)


def test_test_c_pure_yaw():
    """Test C: Known point rotated by known yaw angle lands at expected location."""
    # 90 degrees CCW (yaw = +pi/2): (1, 0, 0) -> (0, 1, 0)
    R_yaw = euler_to_rotation_matrix(roll_rad=0.0, pitch_rad=0.0, yaw_rad=np.pi / 2.0)
    T = create_se3_matrix(R=R_yaw)
    pt = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    expected = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    res = transform_points_3d(T, pt)
    assert np.allclose(res, expected, atol=1e-12)


def test_test_d_combined_se3():
    """Test D: Translation + rotation matches analytical solution."""
    # Yaw 90 deg + translation [2, 3, 0]:
    # (1, 0, 0) rotated by 90 deg -> (0, 1, 0), then translated -> (2, 4, 0)
    R = euler_to_rotation_matrix(0.0, 0.0, np.pi / 2.0)
    t = np.array([2.0, 3.0, 0.0])
    T = create_se3_matrix(R, t)
    pt = np.array([[1.0, 0.0, 0.0]])
    expected = np.array([[2.0, 4.0, 0.0]])
    res = transform_points_3d(T, pt)
    assert np.allclose(res, expected, atol=1e-12)


def test_test_e_camera_to_body_transform():
    """Test E: Known camera observation maps correctly into body coordinates."""
    # Default extrinsics: height=0.40m, forward=0.15m, pitch=0.0 deg for flat test
    ext = SpatialCameraExtrinsics(forward_offset_m=0.15, lateral_offset_m=0.0, height_m=0.40, pitch_deg=0.0)
    T_bc = ext.T_body_camera

    # A landmark at 2.0m directly ahead along camera optical axis (+Z_opt):
    # Camera optical point: [0, 0, 2.0]
    # In body frame:
    # X_body = forward_offset + Z_opt = 0.15 + 2.0 = 2.15
    # Y_body = lateral_offset - X_opt = 0.0 - 0.0 = 0.0
    # Z_body = height - Y_opt = 0.40 - 0.0 = 0.40
    pt_cam = np.array([[0.0, 0.0, 2.0]])
    expected_body = np.array([[2.15, 0.0, 0.40]])
    res_body = transform_points_3d(T_bc, pt_cam)
    assert np.allclose(res_body, expected_body, atol=1e-12)


def test_test_f_body_to_world_transform():
    """Test F: Known body point maps correctly into world frame."""
    # Robot at world [10.0, 5.0, 0.0], heading East (yaw=0)
    T_wb = create_se3_matrix(t=[10.0, 5.0, 0.0])
    pt_body = np.array([[2.0, 1.0, 0.0]])
    expected_world = np.array([[12.0, 6.0, 0.0]])
    res_world = transform_points_3d(T_wb, pt_body)
    assert np.allclose(res_world, expected_world, atol=1e-12)


def test_test_g_full_camera_to_world_chain():
    """Test G: Full camera -> body -> world matches expected coordinates."""
    ext = SpatialCameraExtrinsics(forward_offset_m=0.20, lateral_offset_m=0.0, height_m=0.50, pitch_deg=0.0)
    T_bc = ext.T_body_camera

    # Robot at world [5.0, 5.0, 0.0] with yaw = 90 deg (facing North, +Y)
    R_wb = euler_to_rotation_matrix(0.0, 0.0, np.pi / 2.0)
    t_wb = np.array([5.0, 5.0, 0.0])
    T_wb = create_se3_matrix(R_wb, t_wb)

    T_wc = T_wb @ T_bc

    # Camera sees object at optical [0.0, 0.0, 3.0] (3m ahead of camera):
    # In body: [3.20, 0.0, 0.50] (facing +X_body)
    # In world (rotated 90 deg CCW):
    # X_world = 5.0 - 0.0 = 5.0
    # Y_world = 5.0 + 3.20 = 8.20
    # Z_world = 0.0 + 0.50 = 0.50
    pt_cam = np.array([[0.0, 0.0, 3.0]])
    expected_world = np.array([[5.0, 8.20, 0.50]])
    res_world = transform_points_3d(T_wc, pt_cam)
    assert np.allclose(res_world, expected_world, atol=1e-12)


def test_test_h_inverse_transform():
    """Test H: Forward followed by inverse recovers original point within numerical tolerance."""
    R = euler_to_rotation_matrix(0.1, -0.25, 0.8)
    t = np.array([12.3, -4.56, 7.89])
    T = create_se3_matrix(R, t)
    T_inv = invert_se3(T)

    # Check T @ T_inv = Eye(4)
    assert np.allclose(T @ T_inv, np.eye(4), atol=1e-12)

    pts = np.array([
        [1.2, 3.4, 5.6],
        [-10.0, 20.0, -0.5],
        [0.0, 0.0, 0.0]
    ], dtype=np.float64)

    pts_fwd = transform_points_3d(T, pts)
    pts_recovered = transform_points_3d(T_inv, pts_fwd)
    assert np.allclose(pts_recovered, pts, atol=1e-12)
