"""Unit tests for TemporalSynchronizer and motion-compensated pose interpolation."""

import numpy as np
import pytest

from ai.localization.base import TrackingState
from ai.localization.result import LocalizationEstimate
from ai.spatial_fusion.synchronization import TemporalSynchronizer
from ai.spatial_fusion.transforms import create_se3_matrix, rotation_matrix_to_euler


def create_test_estimate(
    timestamp: float,
    x: float = 0.0,
    y: float = 0.0,
    yaw_rad: float = 0.0,
    vx: float = 0.0,
    wz: float = 0.0,
    tracking_state: TrackingState = TrackingState.TRACKING_OK,
    is_pose_valid: bool = True,
    confidence: float = 0.90
) -> LocalizationEstimate:
    R = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
        [np.sin(yaw_rad),  np.cos(yaw_rad), 0.0],
        [0.0,              0.0,             1.0]
    ])
    T = create_se3_matrix(R=R, t=[x, y, 0.0])
    return LocalizationEstimate(
        timestamp=timestamp,
        pose=T,
        tracking_state=tracking_state,
        confidence=confidence,
        is_pose_valid=is_pose_valid,
        linear_velocity=np.array([vx, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, wz])
    )


def test_sync_matching_within_tolerance():
    """Verify synchronizer finds closest pose within max_sync_delta_s."""
    sync = TemporalSynchronizer(max_sync_delta_s=0.10)
    sync.add_localization_estimate(create_test_estimate(10.0, x=1.0))
    sync.add_localization_estimate(create_test_estimate(10.05, x=1.5))
    sync.add_localization_estimate(create_test_estimate(10.10, x=2.0))

    # Query at 10.06 -> Should match with 10.05
    res = sync.synchronize(obs_timestamp=10.06)
    assert res.is_valid
    assert abs(res.time_delta_s - 0.01) < 1e-6
    assert res.source_estimate is not None
    assert res.source_estimate.timestamp == 10.05


def test_sync_motion_compensation_linear_and_angular():
    """Verify velocity extrapolation correctly projects pose forward during small time delta."""
    sync = TemporalSynchronizer(max_sync_delta_s=0.10, max_extrapolation_s=0.05)
    # Vehicle moving forward at 1.0 m/s and turning CCW at 0.5 rad/s
    est = create_test_estimate(timestamp=20.0, x=5.0, y=0.0, yaw_rad=0.0, vx=1.0, wz=0.5)
    sync.add_localization_estimate(est)

    # Observation arrived 0.03s later
    res = sync.synchronize(obs_timestamp=20.03)
    assert res.is_valid
    assert res.is_extrapolated
    assert res.pose is not None

    # Expected position: x = 5.0 + 1.0 * 0.03 = 5.03, y = 0.0
    assert np.allclose(res.pose[:3, 3], [5.03, 0.0, 0.0], atol=1e-3)

    # Expected yaw: 0.5 rad/s * 0.03s = 0.015 rad
    roll, pitch, yaw = rotation_matrix_to_euler(res.pose[:3, :3])
    assert np.isclose(yaw, 0.015, atol=1e-3)


def test_sync_stale_pose_rejection():
    """Verify observations with no pose within max_sync_delta_s are rejected."""
    sync = TemporalSynchronizer(max_sync_delta_s=0.10)
    sync.add_localization_estimate(create_test_estimate(30.0, x=0.0))

    # Observation arrived 0.25s later (> 0.10s limit)
    res = sync.synchronize(obs_timestamp=30.25)
    assert not res.is_valid
    assert res.pose is None
    assert "timestamp_gap_exceeded" in res.reason


def test_sync_untrusted_pose_rejection():
    """Verify synchronizer rejects candidate poses in LOST or RELOCALIZING states."""
    sync = TemporalSynchronizer(max_sync_delta_s=0.10)
    lost_est = create_test_estimate(
        timestamp=40.0,
        tracking_state=TrackingState.LOST,
        is_pose_valid=False,
        confidence=0.10
    )
    sync.add_localization_estimate(lost_est)

    res = sync.synchronize(obs_timestamp=40.01)
    assert not res.is_valid
    assert res.pose is None
    assert "untrusted" in res.reason


def test_sync_empty_buffer():
    """Verify synchronizer handles empty buffer safely."""
    sync = TemporalSynchronizer()
    res = sync.synchronize(100.0)
    assert not res.is_valid
    assert res.reason == "empty_localization_buffer"
