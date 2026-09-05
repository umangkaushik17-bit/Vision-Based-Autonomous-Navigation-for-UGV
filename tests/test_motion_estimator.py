"""Unit tests for Milestone 13.4 — Motion Estimation.

Validates:
A. Stationary target detection and zero velocity clamping
B. Constant positive vx recovery
C. Constant positive vy recovery
D. Diagonal motion recovery
E. Negative velocity recovery across quadrants
F. Constant speed with turning heading (circular motion)
G. Acceleration response
H. Deceleration response
I. Irregular timestamps handling
J. Zero dt / duplicate timestamp safety
K. Frame dropout / time gap handling
L. Noise rejection: windowed OLS regression vs raw finite difference
M. Multiple simultaneous tracks processing via update()
N. Velocity heading accuracy across quadrants
O. Speed calculation accuracy
P. Deterministic repeated execution
Q. Quantitative error metrics (RMSE, MAE, bias, heading error)
R. Configuration validation
S. Ground-truth isolation audit
T. Ego-motion source audit (ensuring estimated localization frame only)
"""

import inspect
import math
import numpy as np
import pytest

from ai.planning.dynamic.data_model import TrackedObstacleArray, TrackedObstacleState
from ai.planning.dynamic.motion_estimator import (
    MotionEstimate,
    MotionEstimator,
    MotionEstimatorConfig,
)


@pytest.fixture
def estimator():
    return MotionEstimator(
        MotionEstimatorConfig(
            window_size=5,
            min_history_points=2,
            stationary_speed_threshold_m_s=0.08,
            min_dt_s=1e-4,
            default_pos_noise_std_m=0.05,
        )
    )


class TestMotionEstimatorScenarios:
    def test_A_stationary_target(self, estimator):
        # Target stationary at (5.0, 3.0) for 10 frames with tiny jitter (< 1cm)
        history = [(0.1 * i, 5.0 + 0.005 * math.sin(i), 3.0 + 0.005 * math.cos(i)) for i in range(10)]
        est = estimator.estimate_track_motion(
            track_id=1,
            current_x=history[-1][1],
            current_y=history[-1][2],
            current_timestamp=history[-1][0],
            history=history,
            fallback_yaw=0.5,
        )
        assert est.is_stationary is True
        assert est.speed == 0.0
        assert est.vx == 0.0
        assert est.vy == 0.0
        assert math.isclose(est.velocity_heading, 0.5)

    def test_B_constant_positive_vx(self, estimator):
        # True vx = 1.2 m/s, vy = 0.0 m/s
        dt = 0.05
        true_vx = 1.20
        history = [(i * dt, 10.0 + true_vx * i * dt, 2.0) for i in range(10)]
        est = estimator.estimate_track_motion(
            track_id=1,
            current_x=history[-1][1],
            current_y=history[-1][2],
            current_timestamp=history[-1][0],
            history=history,
        )
        assert est.is_stationary is False
        assert math.isclose(est.vx, true_vx, abs_tol=1e-3)
        assert math.isclose(est.vy, 0.0, abs_tol=1e-3)
        assert math.isclose(est.speed, true_vx, abs_tol=1e-3)
        assert math.isclose(est.velocity_heading, 0.0, abs_tol=1e-3)

    def test_C_constant_positive_vy(self, estimator):
        # True vx = 0.0, vy = 1.5 m/s
        dt = 0.05
        true_vy = 1.50
        history = [(i * dt, 0.0, true_vy * i * dt) for i in range(8)]
        est = estimator.estimate_track_motion(
            track_id=2,
            current_x=history[-1][1],
            current_y=history[-1][2],
            current_timestamp=history[-1][0],
            history=history,
        )
        assert math.isclose(est.vx, 0.0, abs_tol=1e-3)
        assert math.isclose(est.vy, true_vy, abs_tol=1e-3)
        assert math.isclose(est.speed, true_vy, abs_tol=1e-3)
        assert math.isclose(est.velocity_heading, math.pi / 2.0, abs_tol=1e-3)

    def test_D_diagonal_motion(self, estimator):
        # True vx = 1.0, vy = 1.0 -> speed = sqrt(2), heading = pi/4
        dt = 0.1
        history = [(i * dt, 1.0 * i * dt, 1.0 * i * dt) for i in range(6)]
        est = estimator.estimate_track_motion(
            track_id=3,
            current_x=history[-1][1],
            current_y=history[-1][2],
            current_timestamp=history[-1][0],
            history=history,
        )
        assert math.isclose(est.vx, 1.0, abs_tol=1e-3)
        assert math.isclose(est.vy, 1.0, abs_tol=1e-3)
        assert math.isclose(est.speed, math.sqrt(2.0), abs_tol=1e-3)
        assert math.isclose(est.velocity_heading, math.pi / 4.0, abs_tol=1e-3)

    def test_E_negative_velocity(self, estimator):
        # True vx = -0.8 m/s, vy = -0.6 m/s -> speed = 1.0 m/s, quadrant III
        dt = 0.05
        history = [(i * dt, 20.0 - 0.8 * i * dt, 10.0 - 0.6 * i * dt) for i in range(8)]
        est = estimator.estimate_track_motion(
            track_id=4,
            current_x=history[-1][1],
            current_y=history[-1][2],
            current_timestamp=history[-1][0],
            history=history,
        )
        assert math.isclose(est.vx, -0.8, abs_tol=1e-3)
        assert math.isclose(est.vy, -0.6, abs_tol=1e-3)
        assert math.isclose(est.speed, 1.0, abs_tol=1e-3)
        assert math.isclose(est.velocity_heading, math.atan2(-0.6, -0.8), abs_tol=1e-3)

    def test_F_constant_speed_circular_turn(self, estimator):
        # Radius R = 5m, angular velocity w = 0.2 rad/s, speed = 1.0 m/s
        dt = 0.05
        R = 5.0
        w = 0.2
        history = []
        for i in range(12):
            t = i * dt
            theta = w * t
            history.append((t, R * math.cos(theta), R * math.sin(theta)))

        est = estimator.estimate_track_motion(
            track_id=5,
            current_x=history[-1][1],
            current_y=history[-1][2],
            current_timestamp=history[-1][0],
            history=history,
        )
        # Tangential speed should be within 10% of 1.0 m/s
        assert math.isclose(est.speed, 1.0, rel_tol=0.10)

    def test_G_acceleration(self, estimator):
        # x(t) = 0.5 * a * t^2 -> v(t) = a * t, a = 1.0 m/s^2
        dt = 0.1
        a = 1.0
        history = [(i * dt, 0.5 * a * (i * dt) ** 2, 0.0) for i in range(8)]
        est = estimator.estimate_track_motion(
            track_id=6,
            current_x=history[-1][1],
            current_y=history[-1][2],
            current_timestamp=history[-1][0],
            history=history,
        )
        # At window center (t ~ 0.5s), velocity should be approx 0.5 m/s
        assert 0.40 <= est.vx <= 0.70

    def test_H_deceleration(self, estimator):
        # v(t) = 2.0 - a * t, a = 1.0 m/s^2
        dt = 0.1
        history = [(i * dt, 2.0 * (i * dt) - 0.5 * 1.0 * (i * dt) ** 2, 0.0) for i in range(8)]
        est = estimator.estimate_track_motion(
            track_id=7,
            current_x=history[-1][1],
            current_y=history[-1][2],
            current_timestamp=history[-1][0],
            history=history,
        )
        assert 1.30 <= est.vx <= 1.60

    def test_I_irregular_timestamps(self, estimator):
        # Irregular timestamps with exact constant velocity (vx=1.0, vy=2.0)
        time_offsets = [0.0, 0.03, 0.11, 0.19, 0.35, 0.42]
        history = [(t, 1.0 * t, 2.0 * t) for t in time_offsets]
        est = estimator.estimate_track_motion(
            track_id=8,
            current_x=history[-1][1],
            current_y=history[-1][2],
            current_timestamp=history[-1][0],
            history=history,
        )
        assert math.isclose(est.vx, 1.0, abs_tol=1e-3)
        assert math.isclose(est.vy, 2.0, abs_tol=1e-3)

    def test_J_zero_dt_duplicate_timestamp(self, estimator):
        # Repeated timestamp at t=0.2s should not cause ZeroDivisionError
        history = [(0.0, 0.0, 0.0), (0.1, 0.1, 0.0), (0.2, 0.2, 0.0), (0.2, 0.2, 0.0)]
        est = estimator.estimate_track_motion(
            track_id=9,
            current_x=0.2,
            current_y=0.0,
            current_timestamp=0.2,
            history=history,
        )
        assert math.isclose(est.vx, 1.0, abs_tol=1e-3)

    def test_K_frame_dropout_handling(self, estimator):
        # 0.5s dropout between frame 3 and frame 4
        history = [
            (0.0, 0.0, 0.0),
            (0.1, 0.1, 0.0),
            (0.2, 0.2, 0.0),
            (0.7, 0.7, 0.0),  # 0.5s gap, steady 1.0 m/s
        ]
        est = estimator.estimate_track_motion(
            track_id=10,
            current_x=0.7,
            current_y=0.0,
            current_timestamp=0.7,
            history=history,
        )
        assert math.isclose(est.vx, 1.0, abs_tol=1e-3)

    def test_L_noisy_position_filtering_comparison(self, estimator):
        # True velocity: vx = 1.0, vy = 0.0
        # Position corrupted by zero-mean Gaussian noise sigma = 0.03m
        rng = np.random.default_rng(42)
        true_vx = 1.0
        dt = 0.05
        steps = 25
        ts = np.array([i * dt for i in range(steps)])
        true_xs = true_vx * ts
        noisy_xs = true_xs + rng.normal(0.0, 0.03, steps)

        raw_errors = []
        filtered_errors = []

        history = []
        for i in range(steps):
            history.append((float(ts[i]), float(noisy_xs[i]), 0.0))
            if i >= 4:
                est = estimator.estimate_track_motion(
                    track_id=11,
                    current_x=float(noisy_xs[i]),
                    current_y=0.0,
                    current_timestamp=float(ts[i]),
                    history=history,
                )
                raw_errors.append(abs(est.raw_vx - true_vx))
                filtered_errors.append(abs(est.vx - true_vx))

        raw_rmse = math.sqrt(np.mean(np.array(raw_errors) ** 2))
        filtered_rmse = math.sqrt(np.mean(np.array(filtered_errors) ** 2))

        # Windowed OLS regression must significantly lower RMSE compared to raw two-point finite difference
        assert filtered_rmse < raw_rmse
        assert filtered_rmse < 0.25  # Well bounded error

    def test_M_multiple_simultaneous_tracks_update(self, estimator):
        cov = np.eye(4) * 0.05
        # Track 1: moving along X at 1.0 m/s
        t1 = TrackedObstacleState(
            track_id=1,
            timestamp=0.3,
            x=0.3,
            y=0.0,
            vx=0.0,
            vy=0.0,
            covariance=cov,
            bbox_size=(1, 1),
            bbox_yaw=0,
            history=[(0.1, 0.1, 0.0), (0.2, 0.2, 0.0), (0.3, 0.3, 0.0)],
        )
        # Track 2: moving along Y at -0.5 m/s
        t2 = TrackedObstacleState(
            track_id=2,
            timestamp=0.3,
            x=0.0,
            y=-0.15,
            vx=0.0,
            vy=0.0,
            covariance=cov,
            bbox_size=(1, 1),
            bbox_yaw=0,
            history=[(0.1, 0.0, -0.05), (0.2, 0.0, -0.10), (0.3, 0.0, -0.15)],
        )
        arr_in = TrackedObstacleArray(header_timestamp=0.3, frame_id="odom", tracks=[t1, t2])
        arr_out = estimator.update(arr_in, timestamp_s=0.3)

        assert len(arr_out) == 2
        assert math.isclose(arr_out[0].vx, 1.0, abs_tol=1e-3)
        assert math.isclose(arr_out[1].vy, -0.5, abs_tol=1e-3)

    def test_N_velocity_heading_quadrants(self, estimator):
        # Quadrant I: (1, 1) -> pi/4
        est_q1 = estimator.estimate_track_motion(1, 1, 1, 1.0, [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
        assert math.isclose(est_q1.velocity_heading, math.pi / 4.0, abs_tol=1e-3)

        # Quadrant II: (-1, 1) -> 3*pi/4
        est_q2 = estimator.estimate_track_motion(2, -1, 1, 1.0, [(0.0, 0.0, 0.0), (1.0, -1.0, 1.0)])
        assert math.isclose(est_q2.velocity_heading, 3 * math.pi / 4.0, abs_tol=1e-3)

        # Quadrant III: (-1, -1) -> -3*pi/4
        est_q3 = estimator.estimate_track_motion(3, -1, -1, 1.0, [(0.0, 0.0, 0.0), (1.0, -1.0, -1.0)])
        assert math.isclose(est_q3.velocity_heading, -3 * math.pi / 4.0, abs_tol=1e-3)

        # Quadrant IV: (1, -1) -> -pi/4
        est_q4 = estimator.estimate_track_motion(4, 1, -1, 1.0, [(0.0, 0.0, 0.0), (1.0, 1.0, -1.0)])
        assert math.isclose(est_q4.velocity_heading, -math.pi / 4.0, abs_tol=1e-3)

    def test_O_speed_calculation(self, estimator):
        # 3-4-5 right triangle: vx=3.0, vy=4.0 -> speed = 5.0
        est = estimator.estimate_track_motion(1, 3.0, 4.0, 1.0, [(0.0, 0.0, 0.0), (1.0, 3.0, 4.0)])
        assert math.isclose(est.speed, 5.0, abs_tol=1e-3)

    def test_P_deterministic_repeated_execution(self, estimator):
        history = [(0.05 * i, 2.0 + 0.4 * i, 1.0 - 0.2 * i) for i in range(8)]
        res1 = estimator.estimate_track_motion(1, history[-1][1], history[-1][2], history[-1][0], history)
        res2 = estimator.estimate_track_motion(1, history[-1][1], history[-1][2], history[-1][0], history)
        assert res1.to_dict() == res2.to_dict()


class TestMetricsAndAudits:
    def test_Q_quantitative_velocity_metrics(self, estimator):
        # Evaluate 50 steps of steady diagonal trajectory with true vx=0.8, vy=0.6
        true_vx = 0.80
        true_vy = 0.60
        true_speed = 1.00
        true_heading = math.atan2(true_vy, true_vx)

        dt = 0.05
        steps = 30
        vx_errors = []
        vy_errors = []
        speed_errors = []
        heading_errors = []

        history = []
        for i in range(steps):
            t = i * dt
            history.append((t, true_vx * t, true_vy * t))
            if i >= 3:
                est = estimator.estimate_track_motion(
                    track_id=1,
                    current_x=history[-1][1],
                    current_y=history[-1][2],
                    current_timestamp=t,
                    history=history,
                )
                vx_errors.append(est.vx - true_vx)
                vy_errors.append(est.vy - true_vy)
                speed_errors.append(est.speed - true_speed)
                heading_errors.append(est.velocity_heading - true_heading)

        vx_rmse = float(np.sqrt(np.mean(np.array(vx_errors) ** 2)))
        vy_rmse = float(np.sqrt(np.mean(np.array(vy_errors) ** 2)))
        vel_rmse = float(np.sqrt(0.5 * (vx_rmse ** 2 + vy_rmse ** 2)))
        vel_mae = float(np.mean(np.abs(vx_errors) + np.abs(vy_errors)) / 2.0)
        vel_bias = float(np.mean(vx_errors))
        heading_err = float(np.mean(np.abs(heading_errors)))

        # Quantitative verification on steady trajectory
        assert vel_rmse < 1e-3, f"Velocity RMSE {vel_rmse} exceeded tolerance"
        assert vel_mae < 1e-3, f"Velocity MAE {vel_mae} exceeded tolerance"
        assert abs(vel_bias) < 1e-3, f"Velocity bias {vel_bias} exceeded tolerance"
        assert heading_err < 1e-3, f"Heading error {heading_err} exceeded tolerance"

    def test_R_config_validation(self):
        with pytest.raises(ValueError, match="window_size"):
            MotionEstimatorConfig(window_size=1)
        with pytest.raises(ValueError, match="min_history_points"):
            MotionEstimatorConfig(min_history_points=0)
        with pytest.raises(ValueError, match="stationary_speed_threshold_m_s"):
            MotionEstimatorConfig(stationary_speed_threshold_m_s=-0.1)
        with pytest.raises(ValueError, match="velocity_smoothing_alpha"):
            MotionEstimatorConfig(velocity_smoothing_alpha=0.0)

    def test_S_ground_truth_isolation_audit(self):
        sig = inspect.signature(MotionEstimator.estimate_track_motion)
        param_names = list(sig.parameters.keys())
        assert param_names == [
            "self",
            "track_id",
            "current_x",
            "current_y",
            "current_timestamp",
            "history",
            "fallback_yaw",
        ]
        # Verify no GT parameters exist in estimate_track_motion
        for prohibited in ["gt", "ground_truth", "future_vel"]:
            assert not any(prohibited in p for p in param_names)

    def test_T_ego_motion_source_audit(self, estimator):
        # Verify estimator operates in world 'odom' frame, independent of robot ego-motion
        t = TrackedObstacleState(
            track_id=1,
            timestamp=0.2,
            x=5.0,
            y=2.0,
            vx=0.0,
            vy=0.0,
            covariance=np.eye(4) * 0.05,
            bbox_size=(1, 1),
            bbox_yaw=0,
            frame_id="odom",
            history=[(0.0, 5.0, 2.0), (0.1, 5.1, 2.0), (0.2, 5.2, 2.0)],
        )
        updated, est = estimator.update_track_state(t)
        assert updated.frame_id == "odom"
        assert math.isclose(updated.vx, 1.0, abs_tol=1e-3)
