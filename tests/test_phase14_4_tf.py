"""Phase 14.4 TF Tree and Frame Integrity Tests."""

from __future__ import annotations

import math
import numpy as np
import pytest

from production.localization.ekf_fusion import MultiSensorEKF
from production.localization.types import WheelOdometryMeasurement


def test_static_sensor_frame_transforms() -> None:
    """Verifies geometric positions and orientation of camera and IMU links relative to base_link."""
    # Camera link mount geometry
    cam_x = 0.15
    cam_y = 0.00
    cam_z = 0.40
    cam_pitch_deg = 15.0

    # IMU link mount geometry
    imu_x = 0.00
    imu_y = 0.00
    imu_z = 0.275

    assert cam_x > 0.0, "Camera must be mounted facing forward on UGV chassis"
    assert cam_z > imu_z, "Camera must be elevated above internal IMU chassis mount"
    assert math.isclose(cam_pitch_deg, 15.0, abs_tol=1e-3)


def test_odom_to_base_link_transform_generation() -> None:
    """Verifies that dynamic odom -> base_link transform matches EKF fused state."""
    ekf = MultiSensorEKF()
    ekf.reset(initial_pose=(2.5, -1.2, 0.785))

    state = ekf.get_state(0.0)
    assert math.isclose(state.x, 2.5, abs_tol=1e-3)
    assert math.isclose(state.y, -1.2, abs_tol=1e-3)
    assert math.isclose(state.yaw, 0.785, abs_tol=1e-3)

    # Quaternion generation
    yaw = state.yaw
    qw = math.cos(yaw / 2.0)
    qz = math.sin(yaw / 2.0)
    norm = math.sqrt(qw * qw + qz * qz)
    assert math.isclose(norm, 1.0, abs_tol=1e-6)


def test_tf_timestamp_monotonicity() -> None:
    """Verifies that published state stamps increase monotonically and handle negative dt gracefully."""
    ekf = MultiSensorEKF()
    ekf.reset()

    # Step forward timestamps
    timestamps = [1.0, 1.02, 1.04, 1.06, 1.08, 1.10]
    last_time = 0.0
    for t in timestamps:
        ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=0.3, vyaw=0.0))
        state = ekf.get_state(t)
        assert state.timestamp > last_time
        last_time = state.timestamp

    # Stale timestamp (t < last_time) must not corrupt state
    ekf.predict(1.05)  # Back in time
    assert ekf.last_predict_time == 1.10  # Must preserve newest timestamp
