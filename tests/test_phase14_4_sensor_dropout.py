"""Phase 14.4 Sensor Dropout and Degradation Tests."""

from __future__ import annotations

import math
import numpy as np
import pytest

from production.localization.ekf_fusion import MultiSensorEKF
from production.localization.types import (
    IMUMeasurement,
    StereoVIOMeasurement,
    WheelOdometryMeasurement,
)


def test_stereo_vio_dropout_smooth_coasting() -> None:
    """Verifies that when Stereo VIO drops out, the EKF smoothly coasts on Wheel Odom + IMU.

    State transition: FUSED_TRACKING_OK -> COASTING_IMU_WHEEL.
    No position discontinuity. Covariance grows smoothly.
    """
    ekf = MultiSensorEKF()
    ekf.reset()

    # 1. Nominal multi-sensor tracking for 1.0 s
    t = 0.0
    dt = 0.05
    for _ in range(20):
        t += dt
        ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=0.4, vyaw=0.0))
        ekf.update_imu(IMUMeasurement(timestamp=t, angular_velocity=np.zeros(3), linear_acceleration=np.array([0, 0, 9.80665])))
        ekf.update_stereo_vio(StereoVIOMeasurement(
            timestamp=t, x=0.4 * t, y=0.0, z=0.0, roll=0, pitch=0, yaw=0,
            vx=0.4, vy=0, vz=0, inliers=100, tracking_state="TRACKING_OK", valid=True,
        ))

    state_before = ekf.get_state(t)
    assert state_before.tracking_state == "FUSED_TRACKING_OK"
    var_before = state_before.position_uncertainty_radius_m

    # 2. VIO drops out for 1.0 s (Wheel Odom + IMU continue)
    for _ in range(20):
        t += dt
        ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=0.4, vyaw=0.0))
        ekf.update_imu(IMUMeasurement(timestamp=t, angular_velocity=np.zeros(3), linear_acceleration=np.array([0, 0, 9.80665])))
        # Note: No VIO update!

    state_during = ekf.get_state(t)
    assert state_during.tracking_state == "COASTING_IMU_WHEEL"
    assert "stereo_vio" not in state_during.active_sensors
    assert "wheel_odom" in state_during.active_sensors
    assert "imu_6axis" in state_during.active_sensors
    # Uncertainty grew smoothly
    assert state_during.position_uncertainty_radius_m > var_before
    # Motion continued smoothly forward without jumping backward or freezing
    assert state_during.x > state_before.x


def test_wheel_odometry_dropout_vio_fallback() -> None:
    """Verifies that if wheel encoders fail, the system continues navigating using Stereo VIO + IMU."""
    ekf = MultiSensorEKF()
    ekf.reset()

    t = 0.0
    dt = 0.05
    # Wheel drops out immediately, only VIO + IMU
    for _ in range(15):
        t += dt
        ekf.update_imu(IMUMeasurement(timestamp=t, angular_velocity=np.zeros(3), linear_acceleration=np.array([0, 0, 9.80665])))
        ekf.update_stereo_vio(StereoVIOMeasurement(
            timestamp=t, x=0.5 * t, y=0.0, z=0.0, roll=0, pitch=0, yaw=0,
            vx=0.5, vy=0, vz=0, inliers=110, tracking_state="TRACKING_OK", valid=True,
        ))

    state = ekf.get_state(t)
    assert "stereo_vio" in state.active_sensors
    assert "wheel_odom" not in state.active_sensors
    assert state.confidence >= 0.70
    assert math.isclose(state.x, 0.5 * t, abs_tol=0.10)


def test_all_sensor_blackout_safe_halt() -> None:
    """Verifies that if all sensors drop out for >0.5s, state degrades to LOST and safe speed limit drops to 0.0 m/s."""
    ekf = MultiSensorEKF()
    ekf.reset()

    # Feed initially
    t = 0.1
    ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=0.3, vyaw=0.0))
    ekf.update_imu(IMUMeasurement(timestamp=t, angular_velocity=np.zeros(3), linear_acceleration=np.array([0, 0, 9.80665])))
    ekf.update_stereo_vio(StereoVIOMeasurement(
        timestamp=t, x=0.03, y=0.0, z=0.0, roll=0, pitch=0, yaw=0,
        vx=0.3, vy=0, vz=0, inliers=80, tracking_state="TRACKING_OK", valid=True,
    ))

    # Advance time 1.0 s with NO sensor updates
    t_blackout = 1.20
    state = ekf.get_state(t_blackout)
    assert state.tracking_state == "LOST"
    assert len(state.active_sensors) == 0
    assert state.confidence < 0.15

    # Safe speed limit must mandate full vehicle halt
    safe_speed = ekf.get_safe_speed_limit(nominal_max_speed=0.55)
    assert safe_speed == 0.0, "Blackout did not safely halt vehicle!"


def test_vio_reacquisition_after_dropout() -> None:
    """Verifies that when VIO recovers after a temporary dropout, covariance contracts smoothly."""
    ekf = MultiSensorEKF()
    ekf.reset()

    # Step 1: Nominal
    t = 0.0
    dt = 0.05
    for _ in range(10):
        t += dt
        ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=0.3, vyaw=0.0))
        ekf.update_stereo_vio(StereoVIOMeasurement(
            timestamp=t, x=0.3 * t, y=0.0, z=0.0, roll=0, pitch=0, yaw=0,
            vx=0.3, vy=0, vz=0, inliers=100, tracking_state="TRACKING_OK", valid=True,
        ))

    # Step 2: VIO lost for 20 frames (covariance inflates)
    for _ in range(20):
        t += dt
        ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=0.3, vyaw=0.0))

    var_inflated = ekf.get_state(t).position_uncertainty_radius_m

    # Step 3: VIO re-acquires
    for _ in range(5):
        t += dt
        ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=0.3, vyaw=0.0))
        ekf.update_stereo_vio(StereoVIOMeasurement(
            timestamp=t, x=0.3 * t, y=0.0, z=0.0, roll=0, pitch=0, yaw=0,
            vx=0.3, vy=0, vz=0, inliers=120, tracking_state="TRACKING_OK", valid=True,
        ))

    var_recovered = ekf.get_state(t).position_uncertainty_radius_m
    assert var_recovered < var_inflated
    assert ekf.get_state(t).tracking_state == "FUSED_TRACKING_OK"
