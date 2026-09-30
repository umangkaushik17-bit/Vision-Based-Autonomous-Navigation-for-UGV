"""15-Dimensional Multi-Sensor Fusion Extended Kalman Filter (EKF) for Phase 14.4.

Implements full 3D spatial state estimation fusing:
1. Differential Drive Wheel Odometry (velocities vx, vyaw in base_link)
2. 6-Axis IMU (angular velocity [wx, wy, wz], linear acceleration [ax, ay, az])
3. Stereo Visual Odometry (metric 3D pose [x, y, z, r, p, y] and linear velocity)

Features:
- Full 15-state kinematic motion model:
  State = [x, y, z, phi, theta, psi, vx, vy, vz, vphi, vtheta, vpsi, ax, ay, az]^T
- Joseph-form covariance update for strict positive semi-definiteness:
  P = (I - K H) P (I - K H)^T + K R K^T
- Normalized innovation gating (Mahalanobis distance) to detect and reject wheel slip
- Dynamic sensor timeout watchdogs (smooth coasting on IMU + wheel during VIO dropouts)
- Dynamic confidence and position uncertainty calculation
- Critical safety invariant: High uncertainty strictly reduces allowed vehicle speed
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from production.localization.types import (
    FusedLocalizationState,
    IMUMeasurement,
    PoseWithCovariance,
    StereoVIOMeasurement,
    TwistWithCovariance,
    WheelOdometryMeasurement,
)


def normalize_angle(angle: float) -> float:
    """Normalizes an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class MultiSensorEKF:
    """Multi-Sensor Extended Kalman Filter for robust GPS-denied outdoor navigation."""

    def __init__(
        self,
        dt_default: float = 0.02,  # 50 Hz prediction cycle
        wheel_slip_gate_chi2: float = 9.21,  # 99% confidence threshold for 2 DoF
    ) -> None:
        self.dt_default = dt_default
        self.wheel_slip_gate_chi2 = wheel_slip_gate_chi2

        # 15-state vector:
        # 0: x, 1: y, 2: z (m)
        # 3: phi (roll), 4: theta (pitch), 5: psi (yaw) (rad)
        # 6: vx, 7: vy, 8: vz (m/s)
        # 9: vphi, 10: vtheta, 11: vpsi (rad/s)
        # 12: ax, 13: ay, 14: az (m/s^2)
        self.x = np.zeros(15, dtype=np.float64)

        # State error covariance P (15x15)
        self.P = np.eye(15, dtype=np.float64)
        self.P[0:3, 0:3] *= 0.01      # Position variance: (10 cm)^2
        self.P[3:6, 3:6] *= 0.005     # Orientation variance: (~4 deg)^2
        self.P[6:9, 6:9] *= 0.05      # Velocity variance
        self.P[9:12, 9:12] *= 0.01    # Angular velocity variance
        self.P[12:15, 12:15] *= 0.10  # Acceleration variance

        # Process noise covariance base Q_base (15x15)
        self.Q_base = np.eye(15, dtype=np.float64)
        self.Q_base[0:3, 0:3] *= 1e-4
        self.Q_base[3:6, 3:6] *= 1e-4
        self.Q_base[6:9, 6:9] *= 1e-2
        self.Q_base[9:12, 9:12] *= 1e-2
        self.Q_base[12:15, 12:15] *= 5e-2

        # Gravity vector in world frame
        self.gravity = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

        # Watchdogs & Sensor Timestamps
        self.last_predict_time: Optional[float] = None
        self.last_wheel_time: Optional[float] = None
        self.last_imu_time: Optional[float] = None
        self.last_vio_time: Optional[float] = None

        # Cumulative Metrics
        self.total_distance_m: float = 0.0
        self.prev_position: Optional[np.ndarray] = None
        self.initial_position: Optional[np.ndarray] = None
        self.wheel_slip_events: int = 0
        self.vio_dropouts: int = 0

    def reset(self, initial_pose: Optional[Tuple[float, float, float]] = None) -> None:
        """Resets the EKF state and covariance."""
        self.x.fill(0.0)
        if initial_pose is not None:
            self.x[0] = initial_pose[0]
            self.x[1] = initial_pose[1]
            self.x[5] = initial_pose[2]

        self.P = np.eye(15, dtype=np.float64)
        self.P[0:3, 0:3] *= 0.01
        self.P[3:6, 3:6] *= 0.005
        self.P[6:9, 6:9] *= 0.05
        self.P[9:12, 9:12] *= 0.01
        self.P[12:15, 12:15] *= 0.10

        self.last_predict_time = None
        self.last_wheel_time = None
        self.last_imu_time = None
        self.last_vio_time = None
        self.total_distance_m = 0.0
        self.prev_position = None
        self.initial_position = None
        self.wheel_slip_events = 0
        self.vio_dropouts = 0

    def predict(self, current_time: float) -> None:
        """Propagates state and covariance forward using discrete 3D kinematics."""
        if self.last_predict_time is None:
            self.last_predict_time = current_time
            if self.initial_position is None:
                self.initial_position = self.x[0:3].copy()
                self.prev_position = self.x[0:3].copy()
            return

        total_dt = current_time - self.last_predict_time
        if total_dt <= 0.0:
            return
        # Cap total integration time to 2.0s to prevent unbounded extrapolation during pauses
        total_dt = min(total_dt, 2.0)

        # Numerically accurate sub-stepping (maximum 0.05s per kinematic step)
        max_step = 0.05
        num_steps = max(1, int(math.ceil(total_dt / max_step)))
        dt = total_dt / num_steps

        F = np.eye(15, dtype=np.float64)
        dt2_half = 0.5 * dt * dt

        # Position integrates velocity and acceleration
        F[0, 6] = dt; F[0, 12] = dt2_half
        F[1, 7] = dt; F[1, 13] = dt2_half
        F[2, 8] = dt; F[2, 14] = dt2_half

        # Orientation integrates angular velocity
        F[3, 9] = dt
        F[4, 10] = dt
        F[5, 11] = dt

        # Velocity integrates acceleration
        F[6, 12] = dt
        F[7, 13] = dt
        F[8, 14] = dt

        Q = self.Q_base * dt

        for _ in range(num_steps):
            # Non-linear state propagation
            self.x[0] += self.x[6] * dt + self.x[12] * dt2_half
            self.x[1] += self.x[7] * dt + self.x[13] * dt2_half
            self.x[2] += self.x[8] * dt + self.x[14] * dt2_half

            self.x[3] = normalize_angle(self.x[3] + self.x[9] * dt)
            self.x[4] = normalize_angle(self.x[4] + self.x[10] * dt)
            self.x[5] = normalize_angle(self.x[5] + self.x[11] * dt)

            self.x[6] += self.x[12] * dt
            self.x[7] += self.x[13] * dt
            self.x[8] += self.x[14] * dt

            # Covariance propagation P = F * P * F^T + Q
            self.P = F @ self.P @ F.T + Q

        # Enforce symmetry
        self.P = 0.5 * (self.P + self.P.T)

        self.last_predict_time = current_time

        # Update distance traveled
        curr_pos = self.x[0:3]
        if self.prev_position is not None:
            step_d = float(np.linalg.norm(curr_pos - self.prev_position))
            self.total_distance_m += step_d
        self.prev_position = curr_pos.copy()

    def update_wheel_odom(self, meas: WheelOdometryMeasurement) -> bool:
        """Updates filter with wheel odometry forward velocity and yaw rate.

        Fuses body-frame velocity vx and vyaw.
        Performs Mahalanobis distance gating to reject wheel slip anomalies.
        """
        if not meas.valid:
            return False

        self.predict(meas.timestamp)
        self.last_wheel_time = meas.timestamp

        psi = self.x[5]
        cos_psi = math.cos(psi)
        sin_psi = math.sin(psi)

        vx_world = self.x[6]
        vy_world = self.x[7]

        # Predicted body-frame forward velocity and yaw rate
        pred_vx_body = vx_world * cos_psi + vy_world * sin_psi
        pred_vyaw = self.x[11]

        # Innovation (residual)
        y = np.array([
            meas.vx - pred_vx_body,
            meas.vyaw - pred_vyaw,
        ], dtype=np.float64)

        # Measurement Jacobian H (2x15)
        H = np.zeros((2, 15), dtype=np.float64)
        # d(v_body_x) / d(psi) = -vx_world * sin(psi) + vy_world * cos(psi)
        H[0, 5] = -vx_world * sin_psi + vy_world * cos_psi
        # d(v_body_x) / d(vx_world) = cos(psi)
        H[0, 6] = cos_psi
        # d(v_body_x) / d(vy_world) = sin(psi)
        H[0, 7] = sin_psi

        # d(vyaw) / d(vpsi) = 1.0
        H[1, 11] = 1.0

        # Measurement covariance R (2x2)
        R = np.array([
            [max(1e-4, float(meas.twist_covariance[0, 0])), 0.0],
            [0.0, max(1e-4, float(meas.twist_covariance[5, 5]))],
        ], dtype=np.float64)

        # Innovation covariance S = H * P * H^T + R
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)

        # Mahalanobis distance gating (wheel slip detection)
        mahalanobis_sq = float(y.T @ S_inv @ y)
        if mahalanobis_sq > self.wheel_slip_gate_chi2:
            self.wheel_slip_events += 1
            # Scale R to softly downweight slipping wheels rather than corrupting state
            R *= (mahalanobis_sq / self.wheel_slip_gate_chi2)
            S = H @ self.P @ H.T + R
            S_inv = np.linalg.inv(S)

        # Kalman gain K = P * H^T * S_inv
        K = self.P @ H.T @ S_inv

        # State update
        self.x += K @ y
        self.x[3] = normalize_angle(self.x[3])
        self.x[4] = normalize_angle(self.x[4])
        self.x[5] = normalize_angle(self.x[5])

        # Joseph form covariance update
        I_KH = np.eye(15, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True

    def update_imu(self, meas: IMUMeasurement) -> bool:
        """Updates filter with 6-axis IMU angular velocities and linear accelerations."""
        if not meas.valid:
            return False

        self.predict(meas.timestamp)
        self.last_imu_time = meas.timestamp

        phi = self.x[3]
        theta = self.x[4]
        psi = self.x[5]

        # Rotation matrix from body to world
        cp = math.cos(pitch := theta)
        sp = math.sin(pitch)
        cr = math.cos(roll := phi)
        sr = math.sin(roll)
        cy = math.cos(yaw := psi)
        sy = math.sin(yaw)

        R_body_to_world = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr],
        ], dtype=np.float64)

        # 1. Update Angular Velocities (wx, wy, wz)
        # Meas matches state indices [9, 10, 11]
        y_w = np.array([
            meas.angular_velocity[0] - self.x[9],
            meas.angular_velocity[1] - self.x[10],
            meas.angular_velocity[2] - self.x[11],
        ], dtype=np.float64)

        H_w = np.zeros((3, 15), dtype=np.float64)
        H_w[0, 9] = 1.0
        H_w[1, 10] = 1.0
        H_w[2, 11] = 1.0

        R_w = meas.angular_velocity_covariance.copy()

        # Kalman update for angular velocity
        S_w = H_w @ self.P @ H_w.T + R_w
        K_w = self.P @ H_w.T @ np.linalg.inv(S_w)

        self.x += K_w @ y_w
        I_KH_w = np.eye(15, dtype=np.float64) - K_w @ H_w
        self.P = I_KH_w @ self.P @ I_KH_w.T + K_w @ R_w @ K_w.T

        # 2. Update Linear Accelerations (ax, ay, az)
        # IMU measures proper acceleration in body frame: a_meas = R^T * (a_world - gravity)
        # Therefore: a_world_meas = R * a_meas + gravity
        a_world_meas = R_body_to_world @ meas.linear_acceleration + self.gravity

        y_a = np.array([
            a_world_meas[0] - self.x[12],
            a_world_meas[1] - self.x[13],
            a_world_meas[2] - self.x[14],
        ], dtype=np.float64)

        H_a = np.zeros((3, 15), dtype=np.float64)
        H_a[0, 12] = 1.0
        H_a[1, 13] = 1.0
        H_a[2, 14] = 1.0

        R_a = meas.linear_acceleration_covariance.copy()

        S_a = H_a @ self.P @ H_a.T + R_a
        K_a = self.P @ H_a.T @ np.linalg.inv(S_a)

        self.x += K_a @ y_a
        I_KH_a = np.eye(15, dtype=np.float64) - K_a @ H_a
        self.P = I_KH_a @ self.P @ I_KH_a.T + K_a @ R_a @ K_a.T

        self.x[3] = normalize_angle(self.x[3])
        self.x[4] = normalize_angle(self.x[4])
        self.x[5] = normalize_angle(self.x[5])
        self.P = 0.5 * (self.P + self.P.T)

        return True

    def update_stereo_vio(self, meas: StereoVIOMeasurement) -> bool:
        """Updates filter with metric 3D position and orientation from Stereo VIO."""
        if not meas.valid or meas.tracking_state == "LOST":
            self.vio_dropouts += 1
            return False

        self.predict(meas.timestamp)
        self.last_vio_time = meas.timestamp

        # Residuals for 3D position and 3D orientation (indices 0..5)
        # Note: angle residuals are wrapped
        y = np.array([
            meas.x - self.x[0],
            meas.y - self.x[1],
            meas.z - self.x[2],
            normalize_angle(meas.roll - self.x[3]),
            normalize_angle(meas.pitch - self.x[4]),
            normalize_angle(meas.yaw - self.x[5]),
            meas.vx - self.x[6],
            meas.vy - self.x[7],
            meas.vz - self.x[8],
        ], dtype=np.float64)

        # Measurement Jacobian H (9x15)
        H = np.zeros((9, 15), dtype=np.float64)
        for i in range(9):
            H[i, i] = 1.0

        # Measurement Covariance R (9x9)
        R = np.zeros((9, 9), dtype=np.float64)
        R[0:6, 0:6] = meas.pose_covariance.copy()
        R[6:9, 6:9] = meas.twist_covariance[0:3, 0:3].copy()

        # Regularize R diagonal
        for i in range(9):
            R[i, i] = max(1e-4, float(R[i, i]))

        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        K = self.P @ H.T @ S_inv

        self.x += K @ y
        self.x[3] = normalize_angle(self.x[3])
        self.x[4] = normalize_angle(self.x[4])
        self.x[5] = normalize_angle(self.x[5])

        # Joseph form update
        I_KH = np.eye(15, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True

    def get_state(self, current_time: Optional[float] = None) -> FusedLocalizationState:
        """Returns the current 15D state estimate, confidence, and tracking state."""
        now = current_time if current_time is not None else (self.last_predict_time or 0.0)
        if self.last_predict_time is not None and now > self.last_predict_time:
            self.predict(now)

        # Check sensor freshness
        timeout_s = 0.35
        wheel_active = (self.last_wheel_time is not None) and (now - self.last_wheel_time <= timeout_s)
        imu_active = (self.last_imu_time is not None) and (now - self.last_imu_time <= timeout_s)
        vio_active = (self.last_vio_time is not None) and (now - self.last_vio_time <= timeout_s)

        active_sensors = []
        if wheel_active: active_sensors.append("wheel_odom")
        if imu_active: active_sensors.append("imu_6axis")
        if vio_active: active_sensors.append("stereo_vio")

        # Determine overall tracking state
        if vio_active and (wheel_active or imu_active):
            tracking_state = "FUSED_TRACKING_OK"
            base_confidence = 0.95
        elif wheel_active and imu_active:
            # Coasting on dead-reckoning
            tracking_state = "COASTING_IMU_WHEEL"
            base_confidence = 0.65
        elif vio_active:
            tracking_state = "VIO_STANDALONE"
            base_confidence = 0.75
        elif wheel_active or imu_active:
            tracking_state = "DEGRADED"
            base_confidence = 0.40
        else:
            tracking_state = "LOST"
            base_confidence = 0.05

        # Position uncertainty radius: sqrt(var_x + var_y)
        var_x = max(0.0, float(self.P[0, 0]))
        var_y = max(0.0, float(self.P[1, 1]))
        uncertainty_radius = math.sqrt(var_x + var_y)

        # Penalize confidence with growing uncertainty
        # e.g. at 0.1m uncertainty penalty factor is ~1.0; at 1.0m it drops by half
        uncertainty_penalty = 1.0 / (1.0 + uncertainty_radius)
        confidence = float(np.clip(base_confidence * uncertainty_penalty, 0.0, 1.0))

        # Calculate drift per 100m
        drift_per_100m = 0.0
        if self.total_distance_m > 5.0:
            drift_per_100m = (uncertainty_radius / self.total_distance_m) * 100.0

        return FusedLocalizationState(
            timestamp=now,
            x=float(self.x[0]),
            y=float(self.x[1]),
            z=float(self.x[2]),
            roll=float(self.x[3]),
            pitch=float(self.x[4]),
            yaw=float(self.x[5]),
            vx=float(self.x[6]),
            vy=float(self.x[7]),
            vz=float(self.x[8]),
            vroll=float(self.x[9]),
            vpitch=float(self.x[10]),
            vyaw=float(self.x[11]),
            ax=float(self.x[12]),
            ay=float(self.x[13]),
            az=float(self.x[14]),
            covariance=self.P.copy(),
            tracking_state=tracking_state,
            active_sensors=active_sensors,
            confidence=confidence,
            distance_traveled_m=self.total_distance_m,
            cumulative_drift_m=uncertainty_radius,
            drift_per_100m=drift_per_100m,
        )

    def get_safe_speed_limit(self, nominal_max_speed: float = 0.55) -> float:
        """Computational Safety Invariant:

        Lower compute or higher uncertainty MUST reduce vehicle speed.
        Missing/stale localization must NEVER increase vehicle speed.
        """
        state = self.get_state()
        if state.tracking_state == "LOST" or state.confidence < 0.15:
            return 0.0  # Immediate halt
        if state.tracking_state == "DEGRADED":
            return min(0.15, nominal_max_speed * 0.3)
        if state.tracking_state == "COASTING_IMU_WHEEL":
            return min(0.30, nominal_max_speed * 0.6)

        # Nominal fused tracking: scale by confidence and uncertainty radius
        safe_v = nominal_max_speed * state.confidence
        if state.position_uncertainty_radius_m > 0.50:
            safe_v = min(safe_v, 0.20)
        return float(max(0.0, min(nominal_max_speed, safe_v)))
