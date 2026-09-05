"""Unit tests for core localization contracts, tracking states, and camera models."""

import numpy as np
import pytest

from ai.localization.base import (
    CameraExtrinsics,
    CameraIntrinsics,
    IMUReading,
    TrackingState,
)
from ai.localization.result import LocalizationEstimate


def test_tracking_state_enums():
    """Verify all 5 required navigation tracking states exist."""
    assert TrackingState.TRACKING_OK.value == "TRACKING_OK"
    assert TrackingState.DEGRADED.value == "DEGRADED"
    assert TrackingState.LOST.value == "LOST"
    assert TrackingState.RELOCALIZING.value == "RELOCALIZING"
    assert TrackingState.RECOVERED.value == "RECOVERED"


def test_camera_intrinsics_properties():
    """Verify camera matrix K and inverse."""
    cam = CameraIntrinsics(fx=462.14, fy=462.14, cx=320.0, cy=240.0, width=640, height=480)
    assert cam.K.shape == (3, 3)
    assert abs(cam.K[0, 0] - 462.14) < 1e-4
    assert abs(cam.K[0, 2] - 320.0) < 1e-4

    # Invertibility check
    diff = cam.K @ cam.K_inv - np.eye(3)
    assert np.linalg.norm(diff) < 1e-10


def test_camera_extrinsics_transformation():
    """Verify base_link to camera optical frame transformation."""
    ext = CameraExtrinsics(x_forward_m=0.15, y_left_m=0.0, z_up_m=0.40, pitch_deg=15.0)
    R = ext.rotation_matrix
    t = ext.translation_vector

    # Orthonormality
    assert abs(np.linalg.det(R) - 1.0) < 1e-6
    assert np.linalg.norm(R.T @ R - np.eye(3)) < 1e-6

    # Forward direction in robot (+X) maps to optical axis forward (+Z)
    p_fwd_rob = np.array([1.0, 0.0, 0.0])
    v_optical = R @ p_fwd_rob
    # With 15 deg pitch down, optical axis points down, so straight-ahead appears upwards (-Y in camera optical frame)
    assert v_optical[2] > 0.90
    assert v_optical[1] < -0.20


def test_localization_estimate_contract():
    """Verify frozen LocalizationEstimate contract and reliability criteria."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [1.5, -0.4, 0.1]

    est = LocalizationEstimate(
        timestamp=10.5,
        pose=T,
        tracking_state=TrackingState.TRACKING_OK,
        confidence=0.85,
        tracked_features=120,
        inliers=45
    )

    assert est.is_reliable
    assert np.allclose(est.position, [1.5, -0.4, 0.1])
    assert abs(np.linalg.norm(est.quaternion) - 1.0) < 1e-6

    # Test dictionary serialization
    d = est.to_dict()
    assert d["timestamp"] == 10.5
    assert d["tracking_state"] == "TRACKING_OK"
    assert d["confidence"] == 0.85

    # Degraded/low confidence state is not reliable
    est_bad = LocalizationEstimate(
        timestamp=11.0,
        pose=T,
        tracking_state=TrackingState.LOST,
        confidence=0.10
    )
    assert not est_bad.is_reliable
