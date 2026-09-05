"""Unit tests for TF2 canonical frame tree, broadcasters, and numerical parity."""

from __future__ import annotations

import numpy as np
import pytest

from ai.spatial_fusion.transforms import (
    SpatialCameraExtrinsics,
    create_se3_matrix,
    euler_to_rotation_matrix,
    transform_points_3d,
)
from ugv_navigation.tf.tf_tree import (
    TfTreeBuffer,
    TfLookupError,
    TfExtrapolationError,
    TransformStamped,
    Transform3D,
)
from ugv_navigation.tf.tf_broadcaster import (
    StaticCameraExtrinsicsPublisher,
    DynamicOdomBroadcaster,
)
from ugv_navigation.utils.time_utils import RosHeader, RosTime


def test_tf_static_camera_extrinsics():
    """Test static camera extrinsics publication and direct lookup."""
    buf = TfTreeBuffer()
    StaticCameraExtrinsicsPublisher.configure_extrinsics(
        buf,
        forward_offset_m=0.15,
        lateral_offset_m=0.0,
        height_offset_m=0.40,
        pitch_deg=15.0,
    )

    # Lookup base_link -> camera_link
    T_bc = buf.lookup_transform_matrix("base_link", "camera_link")
    assert T_bc.shape == (4, 4)
    assert np.allclose(T_bc[:3, 3], [0.15, 0.0, 0.40])

    # Lookup camera_link -> camera_optical
    T_co = buf.lookup_transform_matrix("camera_link", "camera_optical")
    # In optical frame: X is right, Y is down, Z is forward
    # optical (0,0,1) -> camera_link (1,0,0)
    p_opt = np.array([[0.0, 0.0, 1.0]])
    p_link = transform_points_3d(T_co, p_opt)
    assert np.allclose(p_link, [[1.0, 0.0, 0.0]])


def test_tf_dynamic_odom_lookup():
    """Test dynamic odom -> base_link broadcast and interpolation."""
    buf = TfTreeBuffer(cache_time_s=5.0, max_extrapolation_s=0.10)
    broadcaster = DynamicOdomBroadcaster(buf)

    T1 = create_se3_matrix(np.eye(3), [1.0, 0.0, 0.0])
    broadcaster.broadcast_pose(T1, timestamp_s=1.0)

    T2 = create_se3_matrix(np.eye(3), [2.0, 0.0, 0.0])
    broadcaster.broadcast_pose(T2, timestamp_s=2.0)

    # Query at t=1.05s (within 0.10s of t=1.0)
    T_res = buf.lookup_transform_matrix("odom", "base_link", time_s=1.05)
    assert np.allclose(T_res[:3, 3], [1.0, 0.0, 0.0])

    # Query at t=1.5s (gap 0.5s > 0.10s) -> Extrapolation Error
    with pytest.raises(TfExtrapolationError):
        buf.lookup_transform_matrix("odom", "base_link", time_s=1.5)


def test_tf_full_chain_composition():
    """Test full composite transform odom -> camera_optical."""
    buf = TfTreeBuffer()
    StaticCameraExtrinsicsPublisher.configure_extrinsics(
        buf,
        forward_offset_m=0.15,
        height_offset_m=0.40,
        pitch_deg=15.0,
    )
    broadcaster = DynamicOdomBroadcaster(buf)

    R_body = euler_to_rotation_matrix(0, 0, np.radians(30))
    t_body = np.array([5.0, 2.0, 0.0])
    T_odom_base = create_se3_matrix(R_body, t_body)
    broadcaster.broadcast_pose(T_odom_base, timestamp_s=10.0)

    # Composite lookup: odom -> camera_optical
    T_odom_optical = buf.lookup_transform_matrix("odom", "camera_optical", time_s=10.0)
    assert T_odom_optical.shape == (4, 4)

    # Inverse lookup: camera_optical -> odom
    T_optical_odom = buf.lookup_transform_matrix("camera_optical", "odom", time_s=10.0)
    assert np.allclose(T_odom_optical @ T_optical_odom, np.eye(4), atol=1e-10)


def test_tf_numerical_parity_with_phase_9():
    """CRITICAL (Section 38): Verify exact numerical parity between TF2-composed

    transform and Phase 9 pure-Python transform engine. Tolerance: < 1e-12 m.
    """
    # 1. Phase 9 Setup
    extrinsics = SpatialCameraExtrinsics(forward_offset_m=0.15, height_m=0.40, pitch_deg=15.0)
    T_body_cam_py = extrinsics.T_body_camera

    R_wb = euler_to_rotation_matrix(0, 0, np.radians(45))
    t_wb = np.array([3.0, 1.5, 0.0])
    T_world_body_py = create_se3_matrix(R_wb, t_wb)
    T_world_cam_py = T_world_body_py @ T_body_cam_py

    # 2. Phase 10 TF2 Buffer Setup
    buf = TfTreeBuffer()
    StaticCameraExtrinsicsPublisher.configure_extrinsics(
        buf,
        forward_offset_m=0.15,
        height_offset_m=0.40,
        pitch_deg=15.0,
    )
    broadcaster = DynamicOdomBroadcaster(buf)
    broadcaster.broadcast_pose(T_world_body_py, timestamp_s=1.0)

    # Lookup composed transform odom -> camera_optical
    T_odom_optical_tf = buf.lookup_transform_matrix("odom", "camera_optical", time_s=1.0)

    # Test point transformation
    pts_cam = np.array([
        [0.0, 0.0, 2.0],
        [1.0, 0.5, 3.0],
        [-0.5, -0.2, 1.5],
    ])

    pts_world_py = transform_points_3d(T_world_cam_py, pts_cam)
    pts_world_tf = transform_points_3d(T_odom_optical_tf, pts_cam)

    max_diff_m = np.max(np.abs(pts_world_py - pts_world_tf))
    print(f"Phase 9 vs TF2 Parity Max Difference: {max_diff_m:.4e} m")
    assert max_diff_m < 1e-12, f"Parity difference {max_diff_m} exceeded 1e-12 m threshold"
