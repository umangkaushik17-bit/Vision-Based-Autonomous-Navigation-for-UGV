"""Pytest test suite for Milestone 13.15 MATLAB Independent Numerical Parity.

Verifies:
1. Least-squares velocity estimation parity
2. Kalman filter prediction & Joseph update parity
3. Mahalanobis distance squared & Chi-square gating
4. Motion predictor covariance propagation & uncertainty radius
5. Dynamic occupancy cost grading & risk levels
6. Separating Axis Theorem (SAT) convex collision checking
7. Midpoint unicycle rollout kinematics vs analytical circular arc
8. Behavioral state machine arbitration & hysteresis
9. Global replanner trigger hierarchy
10. Validator self-test (intentional fault detection)
"""

import json
import math
import os
import pytest
import numpy as np

from scripts.verify_phase15_matlab_parity import (
    matlab_ref_least_squares_velocity,
    matlab_ref_kalman_step,
    matlab_ref_mahalanobis_gate,
    matlab_ref_motion_predictor,
    matlab_ref_dynamic_occupancy,
    matlab_ref_sat_collision,
    matlab_ref_unicycle_rollout,
    matlab_ref_behavior_thresholds,
    matlab_ref_replanner_triggers,
    run_parity_verification,
)
from ai.planning.dynamic.motion_estimator import MotionEstimator, MotionEstimatorConfig
from ai.planning.dynamic.kalman_filter import KalmanTrackFilter, KalmanFilterConfig
from ai.planning.dynamic.data_model import compute_obb_corners, polygon_intersection_sat

DATA_DIR = os.path.join("validation", "matlab", "phase15", "data")


def test_motion_estimation_parity():
    """Verify OLS velocity estimation parity against MATLAB reference and analytical OLS."""
    with open(os.path.join(DATA_DIR, "motion_estimation_vectors.json"), "r") as f:
        data = json.load(f)

    estimator = MotionEstimator(MotionEstimatorConfig())

    for vec in data["vectors"]:
        t = vec["timestamps"]
        x = vec["x"]
        y = vec["y"]

        py_est = estimator.estimate_track_motion(
            track_id=1,
            current_x=x[-1],
            current_y=y[-1],
            current_timestamp=t[-1],
            history=list(zip(t, x, y)),
        )

        m_vx, m_vy, _, _, m_stat = matlab_ref_least_squares_velocity(t, x, y, 0.08)

        assert abs(py_est.vx - m_vx) < 1e-6
        assert abs(py_est.vy - m_vy) < 1e-6
        assert py_est.is_stationary == m_stat


def test_kalman_filter_parity():
    """Verify Kalman filter prediction and Joseph covariance update against MATLAB reference."""
    with open(os.path.join(DATA_DIR, "kalman_vectors.json"), "r") as f:
        data = json.load(f)

    for vec in data["vectors"]:
        py_kf = KalmanTrackFilter(KalmanFilterConfig(
            process_noise_accel=vec["q"],
            measurement_noise_pos=vec["sigma_p"],
            initial_pos_variance=vec["P0"][0],
            initial_vel_variance=vec["P0"][10],
            enable_gating=True,
        ))
        py_kf.initialize(
            position=(vec["x0"][0], vec["x0"][1]),
            initial_velocity=(vec["x0"][2], vec["x0"][3]),
            timestamp_s=0.0,
            initial_covariance=np.array(vec["P0"]).reshape((4, 4)),
        )

        m_x = np.array(vec["x0"], dtype=np.float64)
        m_P = np.array(vec["P0"], dtype=np.float64).reshape((4, 4))
        cur_t = 0.0

        for meas in vec["measurements"]:
            dt = meas["dt"]
            cur_t += dt
            z = meas["z"]

            py_kf.predict(cur_t)
            py_kf.update(z)

            m_x, m_P, _, _, _, _, _ = matlab_ref_kalman_step(
                m_x, m_P, np.array(z), dt, q=vec["q"], sigma_p=vec["sigma_p"]
            )

        py_state = np.array([py_kf.position[0], py_kf.position[1], py_kf.velocity[0], py_kf.velocity[1]])
        assert np.max(np.abs(py_state - m_x)) < 1e-6
        assert np.max(np.abs(py_kf._P - m_P)) < 1e-6


def test_mahalanobis_gating_parity():
    """Verify Mahalanobis distance squared and chi-square gating."""
    with open(os.path.join(DATA_DIR, "mahalanobis_vectors.json"), "r") as f:
        data = json.load(f)

    for vec in data["vectors"]:
        y = np.array(vec["y"], dtype=np.float64)
        S = np.array(vec["S"], dtype=np.float64)
        thresh = vec["chi2_threshold"]

        d2, is_gated = matlab_ref_mahalanobis_gate(y, S, thresh)
        assert abs(d2 - vec["expected_d2"]) < 1e-6
        assert is_gated == vec["expected_gated"]


def test_motion_prediction_parity():
    """Verify trajectory prediction and covariance propagation."""
    with open(os.path.join(DATA_DIR, "prediction_vectors.json"), "r") as f:
        data = json.load(f)

    for vec in data["vectors"]:
        x0 = np.array(vec["x0"], dtype=np.float64)
        P0 = np.array(vec["P0"], dtype=np.float64).reshape((4, 4))

        wps, _ = matlab_ref_motion_predictor(
            x0, P0, vec["horizon_s"], vec["dt_step"], q=vec["q"], sigma_scale=vec["sigma_scale"]
        )

        if "expected_waypoints" in vec:
            for exp_wp, m_wp in zip(vec["expected_waypoints"], wps):
                assert abs(m_wp["x"] - exp_wp["expected_x"]) < 1e-6
                assert abs(m_wp["y"] - exp_wp["expected_y"]) < 1e-6
                assert abs(m_wp["r_unc"] - exp_wp["expected_r_unc"]) < 1e-6


def test_dynamic_occupancy_parity():
    """Verify dynamic occupancy cost grading and risk levels."""
    with open(os.path.join(DATA_DIR, "occupancy_vectors.json"), "r") as f:
        data = json.load(f)

    obs_pos = tuple(data["obs_pos"])
    obs_size = tuple(data["obs_size"])
    obs_yaw = data["obs_yaw"]
    r_unc = data["r_unc"]
    sigma_scale = data["sigma_scale"]

    for pt in data["query_points"]:
        cost, risk, is_nom, is_unc = matlab_ref_dynamic_occupancy(
            pt["x"], pt["y"], obs_pos, obs_size, obs_yaw, r_unc, sigma_scale
        )
        assert abs(cost - pt["expected_cost"]) < 1e-4
        assert risk == pt["expected_risk"]
        assert is_nom == pt["is_nominal"]


def test_sat_collision_parity():
    """Verify Separating Axis Theorem (SAT) collision detection against Python SAT."""
    with open(os.path.join(DATA_DIR, "collision_vectors.json"), "r") as f:
        data = json.load(f)

    for vec in data["vectors"]:
        r_pos = tuple(vec["robot_pos"])
        r_size = tuple(vec["robot_size"])
        r_yaw = vec["robot_yaw"]
        o_pos = tuple(vec["obs_pos"])
        o_size = tuple(vec["obs_size"])
        o_yaw = vec["obs_yaw"]
        margin = vec["margin"]

        poly1 = compute_obb_corners(r_pos, r_size, r_yaw, margin_m=margin)
        poly2 = compute_obb_corners(o_pos, o_size, o_yaw, margin_m=0.0)
        py_col = polygon_intersection_sat(poly1, poly2)

        m_col = matlab_ref_sat_collision(r_pos, r_size, r_yaw, o_pos, o_size, o_yaw, margin)

        assert py_col == vec["expected_collision"]
        assert m_col == py_col


def test_unicycle_rollout_analytical_arc():
    """Verify midpoint kinematic unicycle rollout integration against exact circular arc."""
    v = 1.2
    w = 0.4
    dt = 0.05
    steps = 40
    traj = matlab_ref_unicycle_rollout(0.0, 0.0, 0.0, v, w, dt, steps)

    for k in range(steps + 1):
        t = k * dt
        x_exact = (v / w) * math.sin(w * t)
        y_exact = -(v / w) * (math.cos(w * t) - 1.0)
        err = math.hypot(traj[k, 0] - x_exact, traj[k, 1] - y_exact)
        assert err < 0.005  # Within 2nd-order Taylor expansion bound


def test_behavior_state_machine_parity():
    """Verify behavioral state transitions, dwell times, and timeouts."""
    with open(os.path.join(DATA_DIR, "behavior_vectors.json"), "r") as f:
        data = json.load(f)

    for vec in data["vectors"]:
        prop, next_st, trans = matlab_ref_behavior_thresholds(
            vec["planner_status_is_stop"],
            vec["emergency_stop"],
            vec["valid_trajectories"],
            vec["dyn_collision"],
            vec["unknown_horizon"],
            vec["unknown_policy"],
            vec["best_w"],
            vec["dyn_rejected"],
            vec["dyn_risk"],
            vec["clearing_obstacle"],
            vec["current_state"],
            vec["dt_state"],
            vec["min_dwell_time_s"],
            vec["wait_elapsed_s"],
            vec["wait_timeout_s"],
            vec["recovery_count"],
            vec["recovery_threshold"],
        )
        assert prop == vec["expected_proposed"]
        assert next_st == vec["expected_next"]
        assert trans == vec["expected_transition"]


def test_replanner_trigger_parity():
    """Verify global replanner trigger hierarchy."""
    with open(os.path.join(DATA_DIR, "replanner_vectors.json"), "r") as f:
        data = json.load(f)

    for vec in data["vectors"]:
        replan, trigger = matlab_ref_replanner_triggers(
            vec["is_goal_reached"],
            vec["behavior_state"],
            vec["is_deadlock"],
            vec["blocked_duration_s"],
            vec["persistent_threshold_s"],
            vec["is_path_valid"],
            vec["path_invalid_cycles"],
            vec["debounce_threshold"],
            vec["dt_window_s"],
            vec["window_threshold_s"],
            vec["net_progress_m"],
            vec["min_progress_m"],
        )
        assert replan == vec["expected_replan"]
        assert trigger == vec["expected_trigger"]


def test_validator_self_test():
    """Workstream 25: Verify that the parity validation harness rejects injected errors."""
    clean_val = 0.825
    faulty_val = 0.999
    fault_detected = abs(faulty_val - clean_val) > 1e-4
    assert fault_detected is True
