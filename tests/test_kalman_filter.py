"""Comprehensive unit test suite for Milestone 13.5 — Kalman Filtering & State Estimation.

Validates:
A. Initialization: valid state, covariance, timestamp, and uninitialized access guard
B. Predict: constant velocity state extrapolation and process noise accumulation
C. Update: innovation residual, Kalman gain, state correction
D. Position convergence: noisy measurements converge toward true mean
E. Velocity convergence: initial zero velocity converges to true target velocity
F. Covariance propagation: Q(dt) scales with dt, P remains symmetric and positive-definite
G. Covariance reduction: observation strictly reduces state covariance trace
H. Covariance growth during dropout: coasting without updates increases uncertainty
I. Irregular dt: handling dt = [0.03, 0.05, 0.10, 0.15] s and mixed sequences
J. Duplicate timestamp: zero/near-zero dt handled gracefully as identity predict
K. Invalid timestamp: non-monotonic dt raises ValueError without corrupting filter state
L. Outlier rejection: Mahalanobis gating rejects extreme outliers while accepting noisy measurements
M. Stationary object: zero-velocity target yields near-zero velocity estimate
N. Constant velocity: linear trajectory achieves low position/velocity RMSE
O. Acceleration: maneuver response tracking with bounded lag
P. Direction change: 90-degree turning target tracking
Q. Missed observations: predict-only coasting preserves track continuity and degrades uncertainty
R. Reacquisition: returns to tracking state without creating new ID after dropout
S. Reset: filter properly resets to uninitialized state
T. NaN/Inf protection: non-finite inputs or corrupted matrices detected and prevented
U. Multiple filters/tracks: isolated state estimators without cross-talk
V. OLS vs Kalman comparison: quantitative comparison (Finite Diff vs OLS vs Kalman)
W. Deterministic reproducibility: fixed random seeds produce bitwise identical results
X. Ground-truth isolation audit: no GT attributes accessed by filter runtime
Y. Performance latency benchmarks: microsecond timing for predict, update, and batch tracks
"""

from __future__ import annotations

import inspect
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    TrackedObstacleArray,
    TrackedObstacleState,
    normalize_angle,
)
from ai.planning.dynamic.kalman_filter import (
    KalmanFilterConfig,
    KalmanTrackFilter,
)
from ai.planning.dynamic.motion_estimator import (
    MotionEstimator,
    MotionEstimatorConfig,
)
from ai.planning.dynamic.tracker import (
    MultiFrameTracker,
    TrackStatus,
    TrackerConfig,
)


@pytest.fixture
def default_kf() -> KalmanTrackFilter:
    config = KalmanFilterConfig(
        process_noise_accel=1.0,
        measurement_noise_pos=0.05,
        initial_pos_variance=0.04,
        initial_vel_variance=1.0,
        gate_chi2_threshold=9.21,
        enable_gating=True,
    )
    return KalmanTrackFilter(config)


class TestKalmanTrackFilterCore:
    """Test fundamental state-space filter mechanics."""

    def test_A_initialization(self, default_kf: KalmanTrackFilter) -> None:
        assert not default_kf.is_initialized
        with pytest.raises(RuntimeError):
            _ = default_kf.position

        default_kf.initialize(position=(5.0, -3.0), timestamp_s=1.0, initial_velocity=(0.5, -0.2))
        assert default_kf.is_initialized
        assert math.isclose(default_kf.last_timestamp, 1.0)
        assert math.isclose(default_kf.position[0], 5.0)
        assert math.isclose(default_kf.position[1], -3.0)
        assert math.isclose(default_kf.velocity[0], 0.5)
        assert math.isclose(default_kf.velocity[1], -0.2)

        P = default_kf.get_covariance()
        assert P.shape == (4, 4)
        assert np.allclose(P, P.T)
        assert np.all(np.linalg.eigvalsh(P) > 0)

        # Re-initialization with negative timestamp raises ValueError
        with pytest.raises(ValueError):
            default_kf.initialize(position=(0, 0), timestamp_s=-0.5)

    def test_B_predict(self, default_kf: KalmanTrackFilter) -> None:
        default_kf.initialize(position=(1.0, 2.0), timestamp_s=0.0, initial_velocity=(2.0, -1.0))
        # Predict 0.5s ahead: px = 1 + 2*0.5 = 2.0, py = 2 - 1*0.5 = 1.5
        state_pred = default_kf.predict(0.5)

        assert math.isclose(state_pred[0], 2.0, abs_tol=1e-6)
        assert math.isclose(state_pred[1], 1.5, abs_tol=1e-6)
        assert math.isclose(state_pred[2], 2.0, abs_tol=1e-6)
        assert math.isclose(state_pred[3], -1.0, abs_tol=1e-6)
        assert math.isclose(default_kf.last_timestamp, 0.5)

    def test_C_update(self, default_kf: KalmanTrackFilter) -> None:
        default_kf.initialize(position=(0.0, 0.0), timestamp_s=0.0, initial_velocity=(0.0, 0.0))
        # Initial covariance position variance is 0.04 (std 0.2m)
        P_prior = default_kf.get_covariance()
        prior_trace = np.trace(P_prior)

        # Measurement at (0.2, 0.1)
        accepted, d_m = default_kf.update(measurement=(0.2, 0.1), timestamp_s=0.1)
        assert accepted is True
        assert d_m > 0.0

        pos = default_kf.position
        # Updated position should move toward measurement
        assert 0.0 < pos[0] < 0.2
        assert 0.0 < pos[1] < 0.1

        P_post = default_kf.get_covariance()
        # Covariance trace must be reduced by the measurement update
        assert np.trace(P_post) < prior_trace

    def test_D_position_convergence(self, default_kf: KalmanTrackFilter) -> None:
        rng = np.random.RandomState(42)
        true_pos = np.array([10.0, -5.0])
        # Initialize filter near true position (e.g. initial observation with noise)
        init_obs = (float(true_pos[0] + 0.03), float(true_pos[1] - 0.02))
        default_kf.initialize(position=init_obs, timestamp_s=0.0)

        # Stream 50 noisy observations around true_pos
        pos_errors = []
        for step in range(1, 51):
            t = step * 0.1
            noise = rng.normal(0.0, 0.05, size=2)
            obs = (float(true_pos[0] + noise[0]), float(true_pos[1] + noise[1]))
            default_kf.update(obs, timestamp_s=t)
            if step >= 20:
                pos = default_kf.position
                pos_errors.append(math.hypot(pos[0] - true_pos[0], pos[1] - true_pos[1]))

        mean_error = float(np.mean(pos_errors))
        assert mean_error < 0.05, f"Position error {mean_error:.4f}m exceeds 0.05m tolerance"

    def test_E_velocity_convergence(self, default_kf: KalmanTrackFilter) -> None:
        rng = np.random.RandomState(42)
        true_v = np.array([1.5, -0.8])
        pos = np.array([0.0, 0.0])
        default_kf.initialize(position=tuple(pos), timestamp_s=0.0, initial_velocity=(0.0, 0.0))

        vels = []
        # Target moving at constant velocity for 40 steps (4.0s)
        for step in range(1, 41):
            t = step * 0.1
            pos = pos + true_v * 0.1
            noise = rng.normal(0.0, 0.03, size=2)
            obs = (float(pos[0] + noise[0]), float(pos[1] + noise[1]))
            default_kf.update(obs, timestamp_s=t)
            if step >= 15:
                vels.append(default_kf.velocity)

        mean_v = np.mean(vels, axis=0)
        mean_vel_error = float(np.linalg.norm(mean_v - true_v))
        assert mean_vel_error < 0.05, f"Mean velocity error {mean_vel_error:.4f}m/s exceeds 0.05m/s tolerance"

    def test_F_covariance_propagation(self, default_kf: KalmanTrackFilter) -> None:
        default_kf.initialize(position=(0.0, 0.0), timestamp_s=0.0)
        P_init = default_kf.get_covariance()

        default_kf.predict(0.2)
        P_dt2 = default_kf.get_covariance()

        default_kf.predict(0.5)
        P_dt5 = default_kf.get_covariance()

        # Uncertainty must grow with prediction interval
        assert np.trace(P_dt5) > np.trace(P_dt2) > np.trace(P_init)
        # Symmetry check
        assert np.allclose(P_dt5, P_dt5.T)
        # PSD check: all eigenvalues >= 0
        assert np.all(np.linalg.eigvalsh(P_dt5) > 0)

    def test_G_covariance_reduction_after_measurement(self, default_kf: KalmanTrackFilter) -> None:
        default_kf.initialize(position=(0.0, 0.0), timestamp_s=0.0)
        default_kf.predict(0.5)
        P_pred = default_kf.get_covariance()

        default_kf.update((0.1, 0.1))
        P_upd = default_kf.get_covariance()

        # Measurement update must contract uncertainty
        assert np.trace(P_upd) < np.trace(P_pred)
        assert P_upd[0, 0] < P_pred[0, 0]
        assert P_upd[1, 1] < P_pred[1, 1]

    def test_H_covariance_growth_during_dropout(self, default_kf: KalmanTrackFilter) -> None:
        default_kf.initialize(position=(0.0, 0.0), timestamp_s=0.0)
        # 10 missed frames (predict only)
        traces = []
        for step in range(1, 11):
            t = step * 0.1
            default_kf.predict(t)
            traces.append(np.trace(default_kf.get_covariance()))

        # Monotonically increasing uncertainty
        for i in range(len(traces) - 1):
            assert traces[i + 1] > traces[i]

    def test_I_irregular_dt(self, default_kf: KalmanTrackFilter) -> None:
        default_kf.initialize(position=(0.0, 0.0), timestamp_s=0.0, initial_velocity=(1.0, 0.0))
        dt_sequence = [0.03, 0.05, 0.10, 0.15, 0.04, 0.12]
        current_t = 0.0

        for dt in dt_sequence:
            current_t += dt
            pred = default_kf.predict(current_t)
            assert np.all(np.isfinite(pred))
            P = default_kf.get_covariance()
            assert np.all(np.isfinite(P))
            assert np.allclose(P, P.T)
            assert np.all(np.linalg.eigvalsh(P) > 0)

    def test_J_duplicate_timestamp(self, default_kf: KalmanTrackFilter) -> None:
        default_kf.initialize(position=(1.0, 2.0), timestamp_s=1.0, initial_velocity=(1.0, 0.0))
        P1 = default_kf.get_covariance()

        # Zero dt: exactly same timestamp
        pred = default_kf.predict(1.0)
        P2 = default_kf.get_covariance()

        assert math.isclose(pred[0], 1.0)
        assert math.isclose(pred[1], 2.0)
        assert np.allclose(P1, P2)

        # Update at identical timestamp
        accepted, _ = default_kf.update((1.01, 2.01), timestamp_s=1.0)
        assert accepted is True

    def test_K_invalid_timestamp(self, default_kf: KalmanTrackFilter) -> None:
        default_kf.initialize(position=(1.0, 2.0), timestamp_s=1.0)

        # Backward time jump raises ValueError
        with pytest.raises(ValueError, match="Non-monotonic timestamp"):
            default_kf.predict(0.8)

        with pytest.raises(ValueError, match="Non-monotonic timestamp"):
            default_kf.update((1.0, 2.0), timestamp_s=0.5)

        # Filter state must remain valid and uncorrupted
        assert math.isclose(default_kf.last_timestamp, 1.0)
        assert np.all(np.isfinite(default_kf.get_state()))

    def test_L_outlier_rejection(self, default_kf: KalmanTrackFilter) -> None:
        default_kf.initialize(position=(0.0, 0.0), timestamp_s=0.0, initial_velocity=(1.0, 0.0))
        # Establish steady state tracking over 10 frames
        for step in range(1, 11):
            t = step * 0.1
            default_kf.update((t * 1.0, 0.0), timestamp_s=t)

        pred_before = default_kf.predict(1.1)
        # Moderate measurement (slightly noisy: 0.08m off) -> ACCEPTED
        accepted, d_m = default_kf.update((1.15, 0.06), timestamp_s=1.1)
        assert accepted is True

        # Extreme outlier (10 meters away!)
        pred_before_outlier = default_kf.predict(1.2)
        accepted_outlier, d_m_outlier = default_kf.update((11.2, 5.0), timestamp_s=1.2)

        assert accepted_outlier is False
        assert d_m_outlier > default_kf.config.gate_chi2_threshold

        # Filter state must reject the outlier and keep the predicted state
        curr_pos = default_kf.position
        assert math.isclose(curr_pos[0], pred_before_outlier[0], abs_tol=1e-5)
        assert math.isclose(curr_pos[1], pred_before_outlier[1], abs_tol=1e-5)

        rej_count, tot_count = default_kf.rejection_stats
        assert rej_count == 1
        assert tot_count == 12

    def test_S_reset(self, default_kf: KalmanTrackFilter) -> None:
        default_kf.initialize((1, 2), 0.5)
        assert default_kf.is_initialized

        default_kf.reset()
        assert not default_kf.is_initialized
        assert default_kf.last_timestamp == -1.0
        with pytest.raises(RuntimeError):
            default_kf.get_state()

    def test_T_nan_inf_protection(self, default_kf: KalmanTrackFilter) -> None:
        with pytest.raises(ValueError):
            default_kf.initialize((float("nan"), 0.0), 0.0)

        with pytest.raises(ValueError):
            default_kf.initialize((0.0, float("inf")), 0.0)

        default_kf.initialize((0.0, 0.0), 0.0)
        with pytest.raises(ValueError):
            default_kf.update((float("nan"), 0.0))

        with pytest.raises(ValueError):
            default_kf.update((0.0, float("inf")))


class TestDynamicTrajectoriesAndEstimators:
    """Comprehensive synthetic trajectory validation and 3-way estimator comparison."""

    @staticmethod
    def generate_trajectory(
        traj_type: str,
        n_steps: int = 50,
        dt: float = 0.1,
        pos_noise_std: float = 0.04,
        seed: int = 12345,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate deterministic ground-truth trajectory and noisy measurements."""
        rng = np.random.RandomState(seed)
        times = np.array([step * dt for step in range(n_steps)], dtype=np.float64)
        gt_pos = np.zeros((n_steps, 2), dtype=np.float64)
        gt_vel = np.zeros((n_steps, 2), dtype=np.float64)

        if traj_type == "stationary":
            gt_pos[:, 0] = 3.0
            gt_pos[:, 1] = -2.0
            gt_vel[:, :] = 0.0

        elif traj_type == "constant_velocity":
            v = np.array([1.2, 0.0])
            for i, t in enumerate(times):
                gt_pos[i] = np.array([1.0, 2.0]) + v * t
                gt_vel[i] = v

        elif traj_type == "diagonal_cv":
            v = np.array([0.8, -0.6])  # Speed = 1.0 m/s
            for i, t in enumerate(times):
                gt_pos[i] = np.array([-2.0, 1.0]) + v * t
                gt_vel[i] = v

        elif traj_type == "positive_accel":
            a = np.array([0.4, 0.2])
            v0 = np.array([0.2, 0.1])
            for i, t in enumerate(times):
                gt_pos[i] = v0 * t + 0.5 * a * (t ** 2)
                gt_vel[i] = v0 + a * t

        elif traj_type == "negative_accel":
            a = np.array([-0.3, 0.0])
            v0 = np.array([1.5, 0.5])
            for i, t in enumerate(times):
                gt_pos[i] = v0 * t + 0.5 * a * (t ** 2)
                gt_vel[i] = v0 + a * t

        elif traj_type == "direction_change":
            # Turns 90 degrees around step 25
            for i, t in enumerate(times):
                if i < 25:
                    gt_pos[i] = np.array([t * 1.0, 0.0])
                    gt_vel[i] = np.array([1.0, 0.0])
                else:
                    t_turn = t - 2.5
                    gt_pos[i] = np.array([2.5, t_turn * 1.0])
                    gt_vel[i] = np.array([0.0, 1.0])

        noisy_obs = gt_pos + rng.normal(0.0, pos_noise_std, size=gt_pos.shape)
        return times, gt_pos, gt_vel, noisy_obs

    def test_M_stationary_trajectory(self) -> None:
        times, gt_pos, gt_vel, noisy_obs = self.generate_trajectory("stationary", n_steps=40)
        kf = KalmanTrackFilter()
        kf.initialize(position=tuple(noisy_obs[0]), timestamp_s=times[0])

        vels = []
        for t, obs in zip(times[1:], noisy_obs[1:]):
            kf.update(tuple(obs), timestamp_s=t)
            vels.append(kf.velocity)

        # Average steady-state speed must be near zero (< 0.04 m/s)
        mean_v = np.mean(vels[15:], axis=0)
        speed = float(np.linalg.norm(mean_v))
        assert speed < 0.04, f"Stationary steady-state speed {speed:.4f}m/s must remain near zero"

    def test_N_constant_velocity_accuracy(self) -> None:
        times, gt_pos, gt_vel, noisy_obs = self.generate_trajectory("diagonal_cv", n_steps=50)
        kf = KalmanTrackFilter(KalmanFilterConfig(process_noise_accel=0.2, measurement_noise_pos=0.04))
        kf.initialize(position=tuple(noisy_obs[0]), timestamp_s=times[0])

        pos_errors = []
        vel_errors = []

        for i in range(1, len(times)):
            t = times[i]
            kf.update(tuple(noisy_obs[i]), timestamp_s=t)
            # After transient (first 10 steps)
            if i >= 10:
                pos_errors.append(math.hypot(kf.position[0] - gt_pos[i, 0], kf.position[1] - gt_pos[i, 1]))
                vel_errors.append(math.hypot(kf.velocity[0] - gt_vel[i, 0], kf.velocity[1] - gt_vel[i, 1]))

        pos_rmse = float(np.sqrt(np.mean(np.square(pos_errors))))
        vel_rmse = float(np.sqrt(np.mean(np.square(vel_errors))))

        assert pos_rmse < 0.050, f"Position RMSE {pos_rmse:.4f}m exceeds 0.050m"
        assert vel_rmse < 0.180, f"Velocity RMSE {vel_rmse:.4f}m/s exceeds 0.180m/s"

    def test_O_acceleration_tracking(self) -> None:
        times, gt_pos, gt_vel, noisy_obs = self.generate_trajectory("positive_accel", n_steps=50)
        kf = KalmanTrackFilter(KalmanFilterConfig(process_noise_accel=2.0))
        kf.initialize(position=tuple(noisy_obs[0]), timestamp_s=times[0])

        for i in range(1, len(times)):
            kf.update(tuple(noisy_obs[i]), timestamp_s=times[i])

        # State should track acceleration within bounded error
        final_vel = kf.velocity
        true_final_vel = gt_vel[-1]
        vel_err = math.hypot(final_vel[0] - true_final_vel[0], final_vel[1] - true_final_vel[1])
        assert vel_err < 0.25, f"Accelerating velocity error {vel_err:.4f} exceeds 0.25m/s"

    def test_P_direction_change_tracking(self) -> None:
        times, gt_pos, gt_vel, noisy_obs = self.generate_trajectory("direction_change", n_steps=50)
        kf = KalmanTrackFilter(KalmanFilterConfig(process_noise_accel=3.0))
        kf.initialize(position=tuple(noisy_obs[0]), timestamp_s=times[0])

        for i in range(1, len(times)):
            kf.update(tuple(noisy_obs[i]), timestamp_s=times[i])

        # At the end of the turn (step 49), heading should align with positive Y (pi/2 rad)
        heading = kf.heading
        assert math.isclose(heading, math.pi / 2, abs_tol=0.20)

    def test_V_estimator_comparison(self) -> None:
        """Explicit benchmark: Raw Finite-Difference vs Windowed OLS vs Kalman Filter."""
        times, gt_pos, gt_vel, noisy_obs = self.generate_trajectory("diagonal_cv", n_steps=60, pos_noise_std=0.04)

        # 1. Kalman Filter
        kf = KalmanTrackFilter(KalmanFilterConfig(process_noise_accel=0.2, measurement_noise_pos=0.04))
        kf.initialize(tuple(noisy_obs[0]), times[0])

        # 2. Windowed OLS MotionEstimator
        ols_estimator = MotionEstimator(MotionEstimatorConfig(window_size=5, min_history_points=2))

        # 3. Raw finite difference history
        raw_fd_vels = []
        ols_vels = []
        kf_vels = []

        history_records = [(float(times[0]), float(noisy_obs[0, 0]), float(noisy_obs[0, 1]))]

        for i in range(1, len(times)):
            t = times[i]
            dt = t - times[i - 1]

            # Raw Finite Difference
            raw_vx = (noisy_obs[i, 0] - noisy_obs[i - 1, 0]) / dt
            raw_vy = (noisy_obs[i, 1] - noisy_obs[i - 1, 1]) / dt
            raw_fd_vels.append([raw_vx, raw_vy])

            # OLS
            history_records.append((float(t), float(noisy_obs[i, 0]), float(noisy_obs[i, 1])))
            ols_est = ols_estimator.estimate_track_motion(
                track_id=0,
                current_x=float(noisy_obs[i, 0]),
                current_y=float(noisy_obs[i, 1]),
                current_timestamp=float(t),
                history=list(history_records),
            )
            ols_vels.append([ols_est.vx, ols_est.vy])

            # Kalman Filter
            kf.update(tuple(noisy_obs[i]), timestamp_s=t)
            kf_vels.append(list(kf.velocity))

        # Evaluate over steady-state (step 15 onward)
        start_idx = 15
        gt_eval_vel = gt_vel[start_idx + 1:]
        raw_eval = np.array(raw_fd_vels[start_idx:])
        ols_eval = np.array(ols_vels[start_idx:])
        kf_eval = np.array(kf_vels[start_idx:])

        # Compute Metrics
        def calc_metrics(est: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
            diff = est - gt
            vx_rmse = float(np.sqrt(np.mean(diff[:, 0] ** 2)))
            vy_rmse = float(np.sqrt(np.mean(diff[:, 1] ** 2)))
            vec_rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
            mae = float(np.mean(np.linalg.norm(diff, axis=1)))
            bias = float(np.mean(np.linalg.norm(diff, axis=1) * np.sign(np.mean(diff))))
            return {
                "vx_rmse": vx_rmse,
                "vy_rmse": vy_rmse,
                "vec_rmse": vec_rmse,
                "mae": mae,
                "bias": bias,
            }

        raw_metrics = calc_metrics(raw_eval, gt_eval_vel)
        ols_metrics = calc_metrics(ols_eval, gt_eval_vel)
        kf_metrics = calc_metrics(kf_eval, gt_eval_vel)

        # Assertions
        # 1. Raw finite difference has high variance due to differentiation of noise
        assert raw_metrics["vec_rmse"] > 0.40, f"Raw FD RMSE should be noisy: {raw_metrics['vec_rmse']}"
        # 2. Windowed OLS significantly filters noise
        assert ols_metrics["vec_rmse"] < 0.20, f"OLS RMSE should be < 0.20: {ols_metrics['vec_rmse']}"
        # 3. Kalman Filter provides smooth recursive estimation achieving lowest RMSE and MAE
        assert kf_metrics["vec_rmse"] < 0.18, f"Kalman RMSE should be < 0.18: {kf_metrics['vec_rmse']}"
        assert kf_metrics["vec_rmse"] < ols_metrics["vec_rmse"]
        assert ols_metrics["vec_rmse"] < raw_metrics["vec_rmse"]


class TestTrackerKalmanIntegration:
    """Test full integration of Kalman filter with MultiFrameTracker."""

    def test_Q_missed_observations_and_reacquisition(self) -> None:
        """Verify dropout coasting and seamless reacquisition under persistent track ID."""
        tracker = MultiFrameTracker(
            TrackerConfig(
                hits_to_confirm=3,
                misses_to_degrade=1,
                misses_to_lost=3,
                misses_to_expire=6,
                use_kalman_filter=True,
            )
        )

        # 3 frames to confirm track
        for step in range(3):
            t = 0.1 * (step + 1)
            det = DynamicObstacleDetection(f"d_{step}", t, "odom", (1.0 + 0.1 * step, 2.0), (1, 1), 0, 0.9)
            arr = tracker.update([det], timestamp_s=t)

        assert arr[0].status == TrackStatus.TRACKING.value
        initial_track_id = arr[0].track_id
        cov_tracking = arr[0].covariance.copy()

        # 2 missed frames: status degrades to DEGRADED, uncertainty expands
        arr_m1 = tracker.update([], timestamp_s=0.4)
        arr_m2 = tracker.update([], timestamp_s=0.5)
        assert arr_m2[0].status == TrackStatus.DEGRADED.value
        cov_degraded = arr_m2[0].covariance.copy()
        assert np.trace(cov_degraded) > np.trace(cov_tracking)

        # 3rd missed frame: status degrades to LOST
        arr_m3 = tracker.update([], timestamp_s=0.6)
        assert arr_m3[0].status == TrackStatus.LOST.value

        # Reacquisition frame at t = 0.7s (observation returns at predicted position)
        det_reacq = DynamicObstacleDetection("d_reacq", 0.7, "odom", (1.7, 2.0), (1, 1), 0, 0.9)
        arr_reacq = tracker.update([det_reacq], timestamp_s=0.7)

        # Reacquired! Track ID MUST be identical, status recovered to TRACKING
        assert len(arr_reacq) == 1
        assert arr_reacq[0].track_id == initial_track_id
        assert arr_reacq[0].status == TrackStatus.TRACKING.value
        cov_reacq = arr_reacq[0].covariance.copy()
        assert np.trace(cov_reacq) < np.trace(cov_degraded)

    def test_U_multiple_tracks_isolation(self) -> None:
        """Test simultaneous tracking of multiple independent obstacles."""
        tracker = MultiFrameTracker(TrackerConfig(hits_to_confirm=1, use_kalman_filter=True))

        for step in range(5):
            t = 0.1 * (step + 1)
            # Track 0 moving along +X, Track 1 moving along +Y, Track 2 moving diagonally
            d0 = DynamicObstacleDetection("d0", t, "odom", (1.0 + step * 0.1, 0.0), (1, 1), 0, 0.9)
            d1 = DynamicObstacleDetection("d1", t, "odom", (0.0, 5.0 + step * 0.1), (1, 1), 0, 0.9)
            d2 = DynamicObstacleDetection("d2", t, "odom", (-3.0 + step * 0.1, -3.0 + step * 0.1), (1, 1), 0, 0.9)

            arr = tracker.update([d0, d1, d2], timestamp_s=t)
            assert len(arr) == 3

        ids = [trk.track_id for trk in arr]
        assert set(ids) == {0, 1, 2}

        t0 = arr[0]
        assert t0.vx > 0.5
        assert math.isclose(t0.vy, 0.0, abs_tol=0.2)

        t1 = arr[1]
        assert math.isclose(t1.vx, 0.0, abs_tol=0.2)
        assert t1.vy > 0.5

    def test_W_deterministic_reproducibility(self) -> None:
        """Verify identical inputs produce bitwise identical states across runs."""
        def run_simulation():
            kf = KalmanTrackFilter(KalmanFilterConfig(process_noise_accel=1.5))
            kf.initialize((1.234, 5.678), 0.0, initial_velocity=(0.12, -0.34))
            for i in range(1, 20):
                t = i * 0.05
                kf.update((1.234 + i * 0.01, 5.678 - i * 0.02), timestamp_s=t)
            return kf.get_state(), kf.get_covariance()

        state1, cov1 = run_simulation()
        state2, cov2 = run_simulation()

        assert np.array_equal(state1, state2)
        assert np.array_equal(cov1, cov2)


class TestGroundTruthIsolationAndPerformance:
    """Validate zero ground-truth leakage and measure execution latency."""

    def test_X_ground_truth_isolation(self) -> None:
        """Audit source code to guarantee no ground truth imports or calls exist in runtime code."""
        import ai.planning.dynamic.kalman_filter as kf_mod
        import ai.planning.dynamic.tracker as trk_mod

        forbidden_tokens = [
            "ground_truth",
            "gt_registry",
            "ground_truth_pose",
            "simulator",
            "Gazebo",
            "ai.simulation",
        ]

        for mod in [kf_mod, trk_mod]:
            src = inspect.getsource(mod)
            for token in forbidden_tokens:
                assert token not in src, f"Forbidden ground-truth token '{token}' found in {mod.__name__}"

    def test_Y_performance_benchmarks(self) -> None:
        """Measure predict latency, update latency, and batch processing times."""
        kf = KalmanTrackFilter()
        kf.initialize((0.0, 0.0), 0.0)

        # Single predict latency benchmark (1000 iterations)
        t0 = time.perf_counter()
        for i in range(1000):
            kf.predict(0.1 + i * 0.001)
        t_predict_total = time.perf_counter() - t0
        predict_us = (t_predict_total / 1000) * 1e6

        # Single update latency benchmark (1000 iterations)
        t0 = time.perf_counter()
        for i in range(1000):
            kf.update((1.0, 1.0))
        t_update_total = time.perf_counter() - t0
        update_us = (t_update_total / 1000) * 1e6

        # Batch MultiFrameTracker benchmarks for 1, 2, 4, 10 tracks
        batch_times_us = {}
        for num_tracks in [1, 2, 4, 10]:
            tracker = MultiFrameTracker(TrackerConfig(hits_to_confirm=1, use_kalman_filter=True))
            dets = [
                DynamicObstacleDetection(f"d_{j}", 0.0, "odom", (float(j * 3.0), 0.0), (1, 1), 0, 0.9)
                for j in range(num_tracks)
            ]
            tracker.update(dets, timestamp_s=0.0)

            t0 = time.perf_counter()
            for step in range(100):
                t = (step + 1) * 0.1
                tracker.update(dets, timestamp_s=t)
            batch_dt = (time.perf_counter() - t0) / 100 * 1e6
            batch_times_us[num_tracks] = batch_dt

        # Ensure reasonable computation bounds (< 150us per filter update)
        assert predict_us < 150.0, f"Predict latency {predict_us:.2f}us exceeds 150us limit"
        assert update_us < 200.0, f"Update latency {update_us:.2f}us exceeds 200us limit"
        assert batch_times_us[10] < 5000.0, f"10-track update {batch_times_us[10]:.2f}us exceeds 5ms"
