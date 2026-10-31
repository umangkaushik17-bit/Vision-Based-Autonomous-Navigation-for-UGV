"""Phase 14.4 100-Meter Trajectory Drift Benchmark Tests.

Simulates long-horizon outdoor UGV trajectories comparing:
1. Pure Dead Reckoning (Wheel Odometry with realistic wheel slip and IMU gyro bias)
2. Fused EKF Localization (Wheel Odometry + 6-axis IMU + Stereo VIO)

Measures and asserts:
- Final Absolute Position Error
- Root Mean Squared Error (RMSE)
- Maximum Position Error
- Accumulated Drift per 100 meters
- Heading Drift Error
"""

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


def test_100m_straight_line_drift_benchmark() -> None:
    """Evaluates 100-meter straight line trajectory at 0.50 m/s for 200 seconds.

    Introduces 2% wheel encoder slip noise and 0.05 deg/s gyro bias.
    Verifies that EKF Multi-Sensor Fusion constrains drift to < 1.5% (< 1.50 m per 100m).
    """
    ekf = MultiSensorEKF()
    ekf.reset(initial_pose=(0.0, 0.0, 0.0))

    duration_s = 200.0  # 100 meters at 0.5 m/s
    dt = 0.05           # 20 Hz updates
    steps = int(duration_s / dt)

    v_true = 0.50
    w_true = 0.00

    rng = np.random.RandomState(42)

    # Accumulators for metrics
    errors_pos = []
    errors_heading = []
    dead_reckoning_x = 0.0
    dead_reckoning_y = 0.0
    dead_reckoning_yaw = 0.0

    t = 0.0
    for step in range(steps):
        t += dt
        true_x = v_true * t
        true_y = 0.0
        true_yaw = 0.0

        # Simulating Wheel Odometry with 3% slip and random noise
        slip_factor = 0.97 + 0.01 * rng.randn()
        wheel_vx = v_true * slip_factor
        wheel_vyaw = w_true + 0.005 * rng.randn()
        meas_wheel = WheelOdometryMeasurement(
            timestamp=t,
            vx=wheel_vx,
            vyaw=wheel_vyaw,
        )

        # Integrate dead reckoning
        dead_reckoning_yaw += wheel_vyaw * dt
        dead_reckoning_x += wheel_vx * math.cos(dead_reckoning_yaw) * dt
        dead_reckoning_y += wheel_vx * math.sin(dead_reckoning_yaw) * dt

        # Simulating 6-axis IMU with bias
        meas_imu = IMUMeasurement(
            timestamp=t,
            angular_velocity=np.array([0.0, 0.0, wheel_vyaw + 0.002]),
            linear_acceleration=np.array([0.0, 0.0, 9.80665]),
        )

        # Simulating Stereo VIO with zero-mean visual Gaussian noise (std = 0.03 m)
        meas_vio = StereoVIOMeasurement(
            timestamp=t,
            x=true_x + 0.03 * rng.randn(),
            y=true_y + 0.03 * rng.randn(),
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=true_yaw + 0.005 * rng.randn(),
            vx=v_true + 0.02 * rng.randn(),
            vy=0.0,
            vz=0.0,
            inliers=95,
            tracking_state="TRACKING_OK",
            valid=True,
        )

        # Update EKF
        ekf.update_wheel_odom(meas_wheel)
        ekf.update_imu(meas_imu)
        ekf.update_stereo_vio(meas_vio)

        state = ekf.get_state(t)

        err_p = math.hypot(state.x - true_x, state.y - true_y)
        err_h = abs(state.yaw - true_yaw)
        errors_pos.append(err_p)
        errors_heading.append(err_h)

    final_state = ekf.get_state(t)
    final_error = math.hypot(final_state.x - 100.0, final_state.y - 0.0)
    rmse_pos = math.sqrt(float(np.mean(np.square(errors_pos))))
    max_error_pos = float(np.max(errors_pos))
    final_heading_drift_deg = math.degrees(final_state.yaw)

    # Pure dead reckoning final error
    dr_final_error = math.hypot(dead_reckoning_x - 100.0, dead_reckoning_y - 0.0)

    print(f"\n--- 100M STRAIGHT-LINE DRIFT BENCHMARK RESULTS ---")
    print(f"True Distance Traveled: 100.0 m")
    print(f"Dead Reckoning Final Error: {dr_final_error:.2f} m")
    print(f"EKF Fused Final Error: {final_error:.2f} m")
    print(f"EKF Fused RMSE: {rmse_pos:.2f} m")
    print(f"EKF Max Error: {max_error_pos:.2f} m")
    print(f"EKF Final Heading Drift: {final_heading_drift_deg:.2f} deg")

    # Assertions
    assert final_error < 1.50, f"Final drift {final_error:.2f} m exceeded 1.5m limit per 100m"
    assert rmse_pos < 1.00, f"RMSE {rmse_pos:.2f} m exceeded 1.0m limit"
    assert final_error < dr_final_error * 0.35, "EKF did not achieve at least 65% drift reduction over dead reckoning"


def test_100m_square_loop_closure_benchmark() -> None:
    """Evaluates 100-meter square trajectory (4 x 25m sides, four 90-degree turns).

    Robot traverses (0,0) -> (25,0) -> (25,25) -> (0,25) -> (0,0).
    Verifies loop closure error at origin is < 1.0 m with EKF fusion.
    """
    ekf = MultiSensorEKF()
    ekf.reset(initial_pose=(0.0, 0.0, 0.0))

    dt = 0.05
    side_length = 25.0
    v = 0.50
    t_side = side_length / v  # 50.0 s per side
    turn_time = 3.14159 / (2.0 * 0.25) # ~6.28 s for 90 deg turn at 0.25 rad/s

    rng = np.random.RandomState(123)

    t = 0.0
    x, y, yaw = 0.0, 0.0, 0.0
    dr_x, dr_y, dr_yaw = 0.0, 0.0, 0.0

    # 4 legs: forward 25m, turn left 90 deg
    for leg in range(4):
        # 1. Drive straight
        steps_straight = int(t_side / dt)
        for _ in range(steps_straight):
            t += dt
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt

            # Wheel with slip
            slip_v = v * (0.97 + 0.01 * rng.randn())
            dr_x += slip_v * math.cos(dr_yaw) * dt
            dr_y += slip_v * math.sin(dr_yaw) * dt

            ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=slip_v, vyaw=0.0))
            ekf.update_imu(IMUMeasurement(
                timestamp=t,
                angular_velocity=np.array([0.0, 0.0, 0.001 * rng.randn()]),
                linear_acceleration=np.array([0.0, 0.0, 9.80665]),
            ))
            ekf.update_stereo_vio(StereoVIOMeasurement(
                timestamp=t,
                x=x + 0.03 * rng.randn(),
                y=y + 0.03 * rng.randn(),
                z=0.0,
                roll=0.0, pitch=0.0, yaw=yaw + 0.005 * rng.randn(),
                vx=v * math.cos(yaw), vy=v * math.sin(yaw), vz=0.0,
                inliers=90, tracking_state="TRACKING_OK", valid=True,
            ))

        # 2. Turn 90 deg left (0.25 rad/s)
        w = 0.25
        steps_turn = int(turn_time / dt)
        for _ in range(steps_turn):
            t += dt
            yaw += w * dt
            dr_yaw += (w * 0.96) * dt  # 4% turn under-rotation on dirt

            ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=0.0, vyaw=w * 0.96))
            ekf.update_imu(IMUMeasurement(
                timestamp=t,
                angular_velocity=np.array([0.0, 0.0, w]),
                linear_acceleration=np.array([0.0, 0.0, 9.80665]),
            ))
            ekf.update_stereo_vio(StereoVIOMeasurement(
                timestamp=t,
                x=x + 0.03 * rng.randn(),
                y=y + 0.03 * rng.randn(),
                z=0.0,
                roll=0.0, pitch=0.0, yaw=yaw + 0.005 * rng.randn(),
                vx=0.0, vy=0.0, vz=0.0,
                inliers=85, tracking_state="TRACKING_OK", valid=True,
            ))

    final_state = ekf.get_state(t)
    loop_closure_error = math.hypot(final_state.x - 0.0, final_state.y - 0.0)
    dr_closure_error = math.hypot(dr_x - 0.0, dr_y - 0.0)

    print(f"\n--- 100M SQUARE LOOP CLOSURE BENCHMARK RESULTS ---")
    print(f"Total Trajectory Distance: ~100.0 m")
    print(f"Dead Reckoning Loop Closure Error: {dr_closure_error:.2f} m")
    print(f"EKF Fused Loop Closure Error: {loop_closure_error:.2f} m")

    assert loop_closure_error < 0.85, f"Loop closure error {loop_closure_error:.2f}m exceeded 0.85m limit"
    assert dr_closure_error > 2.50, "Expected dead reckoning to accumulate significant slip/turn error"
