"""Phase 14.4 Localization Unit Tests: Multi-Sensor EKF Fusion Correctness and Covariance Handling."""

from __future__ import annotations

import math
import numpy as np
import pytest

from production.localization.types import (
    FusedLocalizationState,
    IMUMeasurement,
    StereoVIOMeasurement,
    WheelOdometryMeasurement,
)
from production.localization.ekf_fusion import MultiSensorEKF, normalize_angle
from production.localization.stereo_vio_adapter import StereoVIOAdapter


def test_ekf_initialization() -> None:
    """Verifies that EKF initializes with correct 15D state, symmetric positive-definite covariance, and initial LOST state."""
    ekf = MultiSensorEKF()
    assert ekf.x.shape == (15,)
    assert ekf.P.shape == (15, 15)
    # Check symmetry
    assert np.allclose(ekf.P, ekf.P.T)
    # Check positive definiteness (eigenvalues > 0)
    eigvals = np.linalg.eigvalsh(ekf.P)
    assert np.all(eigvals > 0.0)

    state = ekf.get_state(0.0)
    assert state.tracking_state == "LOST"
    assert state.confidence < 0.15
    assert len(state.active_sensors) == 0


def test_kinematic_prediction_propagation() -> None:
    """Verifies discrete 3D kinematic state prediction with constant velocity."""
    ekf = MultiSensorEKF()
    ekf.reset(initial_pose=(0.0, 0.0, 0.0))
    # Inject forward velocity in world frame
    ekf.x[6] = 1.0  # vx = 1.0 m/s
    ekf.last_predict_time = 0.0

    # Step forward 1.0 s
    ekf.predict(1.0)
    assert math.isclose(ekf.x[0], 1.0, abs_tol=1e-3)
    assert math.isclose(ekf.x[1], 0.0, abs_tol=1e-3)
    # Covariance should grow due to process noise
    assert ekf.P[0, 0] > 0.01


def test_wheel_odometry_update() -> None:
    """Verifies wheel odometry fusion updates velocity and orientation properly."""
    ekf = MultiSensorEKF()
    ekf.reset()

    # Apply wheel odom measurement: vx = 0.5 m/s, vyaw = 0.1 rad/s
    meas = WheelOdometryMeasurement(
        timestamp=0.1,
        vx=0.50,
        vyaw=0.10,
    )
    ok = ekf.update_wheel_odom(meas)
    assert ok is True

    # State velocity should be updated towards 0.5 m/s
    assert ekf.x[6] > 0.10
    assert ekf.last_wheel_time == 0.1


def test_imu_angular_and_linear_acceleration_update() -> None:
    """Verifies 6-axis IMU updates angular velocities and body accelerations."""
    ekf = MultiSensorEKF()
    ekf.reset()

    # IMU reading with yaw rate 0.2 rad/s and forward body acceleration 0.5 m/s^2
    meas = IMUMeasurement(
        timestamp=0.05,
        angular_velocity=np.array([0.0, 0.0, 0.20], dtype=np.float64),
        linear_acceleration=np.array([0.50, 0.0, 9.80665], dtype=np.float64),  # Proper accel cancelling gravity
    )
    ok = ekf.update_imu(meas)
    assert ok is True

    # State angular velocity in z should update towards 0.2 rad/s
    assert math.isclose(ekf.x[11], 0.20, abs_tol=0.05)
    assert ekf.last_imu_time == 0.05


def test_stereo_vio_metric_pose_update() -> None:
    """Verifies Stereo VIO updates 3D pose and constrains drift."""
    ekf = MultiSensorEKF()
    ekf.reset()

    v = 0.50
    dt = 0.05
    steps = 30  # 1.5 seconds at 0.5 m/s -> x reaches 0.75 m; or 60 steps -> x reaches 1.50 m
    # Track sequence of VIO frames moving at 0.5 m/s to x=1.50 m (3.0s total)
    t = 0.0
    for i in range(1, 31):
        t = i * 0.10
        meas = StereoVIOMeasurement(
            timestamp=t,
            x=v * t,
            y=0.10 * t,
            z=0.00,
            roll=0.0,
            pitch=0.0,
            yaw=0.02 * t,
            vx=v,
            vy=0.10,
            vz=0.0,
            inliers=95,
            tracking_state="TRACKING_OK",
            valid=True,
        )
        ok = ekf.update_stereo_vio(meas)
        assert ok is True

    state = ekf.get_state(t)
    assert math.isclose(state.x, 1.50, abs_tol=0.15)
    assert math.isclose(state.y, 0.30, abs_tol=0.10)
    assert state.tracking_state == "VIO_STANDALONE"


def test_wheel_slip_rejection() -> None:
    """Verifies that an unrealistic wheel speed jump (wheel slip) triggers Mahalanobis gating."""
    ekf = MultiSensorEKF()
    ekf.reset()

    # Initialize robot moving at 0.2 m/s verified by VIO
    for t in [0.05, 0.10, 0.15]:
        ekf.update_stereo_vio(StereoVIOMeasurement(
            timestamp=t,
            x=0.2 * t, y=0.0, z=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            vx=0.20, vy=0.0, vz=0.0,
            inliers=100, tracking_state="TRACKING_OK", valid=True,
        ))

    initial_slip_events = ekf.wheel_slip_events
    # Sudden wheel spin: wheel reports 3.0 m/s while vehicle is moving at 0.2 m/s
    slip_meas = WheelOdometryMeasurement(
        timestamp=0.20,
        vx=3.0,
        vyaw=0.0,
    )
    ekf.update_wheel_odom(slip_meas)

    # Wheel slip should have been detected and gated
    assert ekf.wheel_slip_events > initial_slip_events
    # Vehicle speed should NOT jump to 3.0 m/s
    state = ekf.get_state(0.20)
    assert state.vx < 1.0, f"Wheel slip contaminated state velocity: vx = {state.vx}"


def test_multi_sensor_fusion_covariance_reduction() -> None:
    """Verifies that fusing Wheel + IMU + VIO yields lower covariance than individual dead reckoning."""
    ekf = MultiSensorEKF()
    ekf.reset()

    t = 0.05
    ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=0.4, vyaw=0.0))
    ekf.update_imu(IMUMeasurement(
        timestamp=t,
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        linear_acceleration=np.array([0.0, 0.0, 9.80665]),
    ))
    ekf.update_stereo_vio(StereoVIOMeasurement(
        timestamp=t,
        x=0.02, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0,
        vx=0.4, vy=0.0, vz=0.0,
        inliers=120, tracking_state="TRACKING_OK", valid=True,
    ))

    state = ekf.get_state(t)
    assert state.tracking_state == "FUSED_TRACKING_OK"
    assert "wheel_odom" in state.active_sensors
    assert "imu_6axis" in state.active_sensors
    assert "stereo_vio" in state.active_sensors
    assert state.confidence >= 0.80
    # Variance should be tightly bounded
    assert state.position_uncertainty_radius_m < 0.20


def test_computational_safety_rule_speed_limits() -> None:
    """Verifies the core computational safety invariant:

    Higher uncertainty / lost tracking strictly REDUCES vehicle speed; NEVER accelerates.
    """
    ekf = MultiSensorEKF()
    ekf.reset()

    # 1. Nominal Tracking
    t = 0.10
    ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=0.4, vyaw=0.0))
    ekf.update_imu(IMUMeasurement(timestamp=t, angular_velocity=np.zeros(3), linear_acceleration=np.array([0, 0, 9.80665])))
    ekf.update_stereo_vio(StereoVIOMeasurement(
        timestamp=t, x=0.04, y=0.0, z=0.0, roll=0, pitch=0, yaw=0,
        vx=0.4, vy=0, vz=0, inliers=100, tracking_state="TRACKING_OK", valid=True,
    ))
    nominal_limit = ekf.get_safe_speed_limit(nominal_max_speed=0.55)
    assert nominal_limit >= 0.40

    # 2. Complete Sensor Loss -> Speed must be 0.0 (immediate halt)
    ekf_lost = MultiSensorEKF()
    ekf_lost.reset()
    lost_limit = ekf_lost.get_safe_speed_limit(nominal_max_speed=0.55)
    assert lost_limit == 0.0, f"Lost localization allowed nonzero speed: {lost_limit}"
    assert lost_limit < nominal_limit
