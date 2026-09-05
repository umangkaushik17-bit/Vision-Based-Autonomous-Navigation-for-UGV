"""Tests verifying the frozen Phase 8 -> Phase 9 Localization Contract and zero-trust safety rules."""

import numpy as np
import pytest

from ai.localization.base import TrackingState
from ai.localization.result import LocalizationEstimate


def create_sample_pose(x: float = 1.0, y: float = 2.0, z: float = 0.5) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [x, y, z]
    return T


def test_contract_test_a_tracking_ok():
    """Test A: TRACKING_OK produces a valid, trusted localization result."""
    T = create_sample_pose(1.5, -0.2, 0.0)
    est = LocalizationEstimate(
        timestamp=10.0,
        pose=T,
        tracking_state=TrackingState.TRACKING_OK,
        confidence=0.85,
        is_pose_valid=True
    )
    assert est.is_pose_valid
    assert est.trusted_pose is not None
    assert np.allclose(est.trusted_pose, T)
    assert est.is_reliable
    assert est.allows_spatial_fusion


def test_contract_test_b_degraded():
    """Test B: DEGRADED produces documented confidence behavior and inflated covariance."""
    T = create_sample_pose(2.0, 0.1, 0.0)
    est = LocalizationEstimate(
        timestamp=10.1,
        pose=T,
        tracking_state=TrackingState.DEGRADED,
        confidence=0.50,
        is_pose_valid=True
    )
    assert est.is_pose_valid
    assert est.trusted_pose is not None
    assert not est.is_reliable  # Not fully reliable for nominal full-speed motion
    assert est.allows_spatial_fusion  # Allowed with conservative weighting
    # Covariance inflated by factor of 1/(0.5^2) = 4x
    cov = est.covariance_proxy
    assert cov[0, 0] > 0.01


def test_contract_test_c_lost():
    """Test C: LOST cannot be mistaken for a trusted current pose."""
    T = create_sample_pose(5.0, 5.0, 0.0)
    est = LocalizationEstimate(
        timestamp=10.2,
        pose=T,
        tracking_state=TrackingState.LOST,
        confidence=0.10,
        is_pose_valid=False
    )
    assert not est.is_pose_valid
    assert est.trusted_pose is None  # Strictly None under Zero-Trust
    assert not est.is_reliable
    assert not est.allows_spatial_fusion


def test_contract_test_d_relocalizing():
    """Test D: RELOCALIZING cannot be mistaken for normal tracking."""
    T = create_sample_pose(5.0, 5.0, 0.0)
    est = LocalizationEstimate(
        timestamp=10.3,
        pose=T,
        tracking_state=TrackingState.RELOCALIZING,
        confidence=0.15,
        is_pose_valid=False
    )
    assert not est.is_pose_valid
    assert est.trusted_pose is None
    assert not est.is_reliable
    assert not est.allows_spatial_fusion


def test_contract_test_e_recovered():
    """Test E: RECOVERED restores documented pose-validity behavior."""
    T = create_sample_pose(2.5, 0.3, 0.0)
    est = LocalizationEstimate(
        timestamp=10.4,
        pose=T,
        tracking_state=TrackingState.RECOVERED,
        confidence=0.90,
        is_pose_valid=True,
        relocalized=True
    )
    assert est.is_pose_valid
    assert est.trusted_pose is not None
    assert est.is_reliable
    assert est.allows_spatial_fusion
    assert est.relocalized


def test_contract_test_f_timestamps_and_frames():
    """Test F: Timestamps and frame IDs are preserved correctly."""
    T = create_sample_pose()
    est = LocalizationEstimate(
        timestamp=1725350400.123,
        pose=T,
        tracking_state=TrackingState.TRACKING_OK,
        frame_id="odom",
        child_frame_id="base_link",
        sequence_id="seq_alpha"
    )
    assert est.timestamp == 1725350400.123
    assert est.frame_id == "odom"
    assert est.child_frame_id == "base_link"
    assert est.sequence_id == "seq_alpha"


def test_contract_test_g_no_stale_pose_silent_propagation():
    """Test G: No stale pose is silently treated as current during LOST."""
    stale_T = create_sample_pose(10.0, 20.0, 1.0)
    # Even if raw pose matrix holds stale position, trusted_pose must be None
    est = LocalizationEstimate(
        timestamp=15.0,
        pose=stale_T,
        tracking_state=TrackingState.LOST,
        is_pose_valid=False,
        confidence=0.05
    )
    # Downstream spatial fusion checking trusted_pose gets None
    assert est.trusted_pose is None
    # Downstream spatial fusion checking allows_spatial_fusion gets False
    assert est.allows_spatial_fusion is False
