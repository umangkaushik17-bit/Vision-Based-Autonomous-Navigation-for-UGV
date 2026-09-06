"""Generate deterministic test vectors for Milestone 13.15 MATLAB Independent Validation.

Generates authoritative JSON test vectors in validation/matlab/phase15/data/:
1. motion_estimation_vectors.json
2. kalman_vectors.json
3. mahalanobis_vectors.json
4. prediction_vectors.json
5. occupancy_vectors.json
6. collision_vectors.json
7. dwa_vectors.json
8. behavior_vectors.json
9. replanner_vectors.json
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List
import numpy as np

OUTPUT_DIR = os.path.join("validation", "matlab", "phase15", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_motion_estimation_vectors() -> Dict[str, Any]:
    """Vectors for least-squares velocity estimation."""
    cases = []

    # Case 1: Pure linear motion vx=1.5, vy=0.5, no noise
    dt = 0.1
    t = [i * dt for i in range(10)]
    x = [2.0 + 1.5 * ti for ti in t]
    y = [1.0 + 0.5 * ti for ti in t]
    cases.append({
        "case_id": "linear_constant_velocity",
        "timestamps": t,
        "x": x,
        "y": y,
        "expected_vx": 1.5,
        "expected_vy": 0.5,
        "expected_r2_x": 1.0,
        "expected_r2_y": 1.0,
        "is_stationary": False,
    })

    # Case 2: Pure stationary obstacle (zero velocity)
    t = [i * dt for i in range(10)]
    x = [5.0 for _ in t]
    y = [3.0 for _ in t]
    cases.append({
        "case_id": "stationary_pure_zero",
        "timestamps": t,
        "x": x,
        "y": y,
        "expected_vx": 0.0,
        "expected_vy": 0.0,
        "expected_r2_x": 1.0,
        "expected_r2_y": 1.0,
        "is_stationary": True,
    })

    # Case 3: Stationary jitter below velocity threshold
    np.random.seed(42)
    t = [i * dt for i in range(10)]
    x = [5.0 + 0.005 * math.sin(2 * math.pi * i / 5) for i in range(10)]
    y = [3.0 + 0.005 * math.cos(2 * math.pi * i / 5) for i in range(10)]
    cases.append({
        "case_id": "stationary_sub_centimeter_jitter",
        "timestamps": t,
        "x": x,
        "y": y,
        "expected_vx": 0.0,  # Below stationary threshold 0.08 m/s
        "expected_vy": 0.0,
        "expected_r2_x": 0.0,
        "expected_r2_y": 0.0,
        "is_stationary": True,
    })

    # Case 4: Pure Y motion vx=0.0, vy=-2.0
    t = [i * 0.05 for i in range(12)]
    x = [10.0 for _ in t]
    y = [8.0 - 2.0 * ti for ti in t]
    cases.append({
        "case_id": "collinear_negative_y",
        "timestamps": t,
        "x": x,
        "y": y,
        "expected_vx": 0.0,
        "expected_vy": -2.0,
        "expected_r2_x": 1.0,
        "expected_r2_y": 1.0,
        "is_stationary": False,
    })

    # Case 5: Uneven timestamps
    t = [0.0, 0.09, 0.21, 0.32, 0.40, 0.51, 0.63, 0.70]
    x = [1.0 + 0.8 * ti for ti in t]
    y = [-2.0 + 1.2 * ti for ti in t]
    cases.append({
        "case_id": "uneven_timestamps",
        "timestamps": t,
        "x": x,
        "y": y,
        "expected_vx": 0.8,
        "expected_vy": 1.2,
        "expected_r2_x": 1.0,
        "expected_r2_y": 1.0,
        "is_stationary": False,
    })

    # Case 6: Minimal 2-point history
    t = [0.0, 0.1]
    x = [0.0, 0.15]
    y = [0.0, 0.20]
    cases.append({
        "case_id": "minimal_two_points",
        "timestamps": t,
        "x": x,
        "y": y,
        "expected_vx": 1.5,
        "expected_vy": 2.0,
        "expected_r2_x": 1.0,
        "expected_r2_y": 1.0,
        "is_stationary": False,
    })

    return {"vectors": cases}


def generate_kalman_vectors() -> Dict[str, Any]:
    """Vectors for 4-state Constant Velocity Kalman Filter."""
    cases = []

    # Case 1: Standard linear tracking over 15 steps
    dt = 0.1
    q_val = 1.0
    sigma_p = 0.05
    x0 = [1.0, 2.0, 1.0, 0.5]
    p0 = [0.04, 0.0, 0.0, 0.0,
          0.0, 0.04, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 1.0]

    measurements = []
    for k in range(15):
        tk = (k + 1) * dt
        # True state: x=1+1*t, y=2+0.5*t
        zx = 1.0 + 1.0 * tk
        zy = 2.0 + 0.5 * tk
        measurements.append({"step": k, "dt": dt, "z": [zx, zy]})

    cases.append({
        "case_id": "constant_velocity_nominal",
        "x0": x0,
        "P0": p0,
        "q": q_val,
        "sigma_p": sigma_p,
        "measurements": measurements,
    })

    # Case 2: Maneuver / Step change in velocity at step 8
    measurements_maneuver = []
    cur_x, cur_y = 0.0, 0.0
    vx, vy = 1.0, 0.0
    for k in range(15):
        if k == 8:
            vx, vy = 0.0, 2.0  # Sharp turn
        cur_x += vx * dt
        cur_y += vy * dt
        measurements_maneuver.append({"step": k, "dt": dt, "z": [cur_x, cur_y]})

    cases.append({
        "case_id": "maneuver_step_velocity",
        "x0": [0.0, 0.0, 1.0, 0.0],
        "P0": p0,
        "q": q_val,
        "sigma_p": sigma_p,
        "measurements": measurements_maneuver,
    })

    # Case 3: Outlier measurement (large innovation)
    measurements_outlier = []
    for k in range(10):
        tk = (k + 1) * dt
        if k == 5:
            zx, zy = 10.0, 10.0  # Gross outlier
        else:
            zx = 1.0 + 1.0 * tk
            zy = 2.0 + 0.5 * tk
        measurements_outlier.append({"step": k, "dt": dt, "z": [zx, zy]})

    cases.append({
        "case_id": "outlier_measurement_injection",
        "x0": x0,
        "P0": p0,
        "q": q_val,
        "sigma_p": sigma_p,
        "measurements": measurements_outlier,
    })

    return {"vectors": cases}


def generate_mahalanobis_vectors() -> Dict[str, Any]:
    """Vectors for Mahalanobis distance squared and chi-square gating."""
    cases = []

    # Case 1: Innovation at origin (d2 = 0)
    cases.append({
        "case_id": "zero_innovation",
        "y": [0.0, 0.0],
        "S": [[0.05, 0.0], [0.0, 0.05]],
        "chi2_threshold": 9.21,
        "expected_d2": 0.0,
        "expected_gated": True,
    })

    # Case 2: 1-sigma along principal axis
    cases.append({
        "case_id": "one_sigma_diagonal",
        "y": [math.sqrt(0.04), 0.0],
        "S": [[0.04, 0.0], [0.0, 0.04]],
        "chi2_threshold": 9.21,
        "expected_d2": 1.0,
        "expected_gated": True,
    })

    # Case 3: Near boundary inside chi2 threshold (d2 = 9.20 < 9.21)
    cases.append({
        "case_id": "boundary_inside_9_20",
        "y": [math.sqrt(9.20 * 0.04), 0.0],
        "S": [[0.04, 0.0], [0.0, 0.04]],
        "chi2_threshold": 9.21,
        "expected_d2": 9.20,
        "expected_gated": True,
    })

    # Case 4: Near boundary outside chi2 threshold (d2 = 9.22 > 9.21)
    cases.append({
        "case_id": "boundary_outside_9_22",
        "y": [math.sqrt(9.22 * 0.04), 0.0],
        "S": [[0.04, 0.0], [0.0, 0.04]],
        "chi2_threshold": 9.21,
        "expected_d2": 9.22,
        "expected_gated": False,
    })

    # Case 4: Outlier beyond threshold (d2 = 12.5)
    # y = [sqrt(12.5 * 0.02), 0]
    cases.append({
        "case_id": "outlier_rejection",
        "y": [0.5, 0.0],
        "S": [[0.02, 0.0], [0.0, 0.02]],
        "chi2_threshold": 9.21,
        "expected_d2": 12.5,
        "expected_gated": False,
    })

    # Case 5: Correlated off-diagonal covariance matrix
    # S = [[0.05, 0.02], [0.02, 0.04]]
    # det = 0.05*0.04 - 0.02^2 = 0.0020 - 0.0004 = 0.0016
    # inv(S) = (1/0.0016) * [[0.04, -0.02], [-0.02, 0.05]] = [[25.0, -12.5], [-12.5, 31.25]]
    # y = [0.1, -0.1]
    # y^T inv(S) y = 0.1*(25*0.1 - 12.5*(-0.1)) + (-0.1)*(-12.5*0.1 + 31.25*(-0.1))
    # = 0.1*(2.5 + 1.25) - 0.1*(-1.25 - 3.125) = 0.1*3.75 + 0.1*4.375 = 0.375 + 0.4375 = 0.8125
    cases.append({
        "case_id": "correlated_anisotropic_ellipse",
        "y": [0.1, -0.1],
        "S": [[0.05, 0.02], [0.02, 0.04]],
        "chi2_threshold": 9.21,
        "expected_d2": 0.8125,
        "expected_gated": True,
    })

    return {"vectors": cases}


def generate_prediction_vectors() -> Dict[str, Any]:
    """Vectors for trajectory extrapolation, covariance propagation, and uncertainty radius."""
    cases = []

    # Case 1: Constant velocity nominal extrapolation over 2.0s with dt=0.25s (8 future steps)
    x0 = [2.0, 3.0, 1.0, -0.5]
    P0 = [0.04, 0.0, 0.0, 0.0,
          0.0, 0.04, 0.0, 0.0,
          0.0, 0.0, 0.25, 0.0,
          0.0, 0.0, 0.0, 0.25]
    q_val = 1.0
    sigma_scale = 3.0
    horizon_s = 2.0
    dt_step = 0.25

    steps = int(round(horizon_s / dt_step))
    expected_waypoints = []
    for k in range(steps + 1):
        tau = k * dt_step
        px = 2.0 + 1.0 * tau
        py = 3.0 - 0.5 * tau
        # Covariance propagation:
        # F_pos = [[1, 0, tau, 0], [0, 1, 0, tau]]
        # F P0 F^T + Q(tau)
        # P_pos_xx = P0_xx + 2*tau*P0_xvx + tau^2*P0_vxvx + (1/3)*q*tau^3
        # Here P0_xx=0.04, P0_xvx=0, P0_vxvx=0.25
        p_pos_xx = 0.04 + 0.25 * (tau ** 2) + (1.0 / 3.0) * q_val * (tau ** 3)
        p_pos_yy = 0.04 + 0.25 * (tau ** 2) + (1.0 / 3.0) * q_val * (tau ** 3)
        p_pos_xy = 0.0
        # Eigenvalues of isotropic 2x2 are equal
        lambda_max = p_pos_xx
        r_unc = sigma_scale * math.sqrt(lambda_max)
        expected_waypoints.append({
            "step": k,
            "tau": tau,
            "expected_x": px,
            "expected_y": py,
            "expected_vx": 1.0,
            "expected_vy": -0.5,
            "expected_p_xx": p_pos_xx,
            "expected_p_yy": p_pos_yy,
            "expected_p_xy": p_pos_xy,
            "expected_r_unc": r_unc,
        })

    cases.append({
        "case_id": "nominal_forward_prediction",
        "x0": x0,
        "P0": P0,
        "q": q_val,
        "sigma_scale": sigma_scale,
        "horizon_s": horizon_s,
        "dt_step": dt_step,
        "expected_waypoints": expected_waypoints,
    })

    # Case 2: Stationary obstacle (vx=0, vy=0)
    cases.append({
        "case_id": "stationary_prediction",
        "x0": [5.0, 5.0, 0.0, 0.0],
        "P0": [0.01, 0.0, 0.0, 0.0,
               0.0, 0.01, 0.0, 0.0,
               0.0, 0.0, 0.01, 0.0,
               0.0, 0.0, 0.0, 0.01],
        "q": 0.1,
        "sigma_scale": 3.0,
        "horizon_s": 1.0,
        "dt_step": 0.5,
    })

    return {"vectors": cases}


def generate_occupancy_vectors() -> Dict[str, Any]:
    """Vectors for dynamic occupancy grid cost and risk classification."""
    obs_pos = [0.0, 0.0]
    obs_size = [1.0, 0.6]  # half_l = 0.5, half_w = 0.3
    obs_yaw = 0.0
    r_unc = 0.60
    sigma_scale = 3.0
    sigma_pos = r_unc / sigma_scale  # 0.20

    query_points = [
        # Inside nominal box
        {"id": "center", "x": 0.0, "y": 0.0, "expected_cost": 1.0, "expected_risk": 4, "is_nominal": True},
        {"id": "near_edge_inside", "x": 0.45, "y": 0.25, "expected_cost": 1.0, "expected_risk": 4, "is_nominal": True},
        # High-risk zone: dist <= 0.20m. Point at x=0.60, y=0.0 -> dist = 0.10m. norm_d = 0.10/0.20 = 0.5
        # cost = 0.95 - 0.25 * 0.5 = 0.825
        {"id": "high_risk_mid", "x": 0.60, "y": 0.0, "expected_cost": 0.825, "expected_risk": 3, "is_nominal": False},
        # Medium-risk zone: 0.20 < dist <= 0.40m. Point at x=0.80, y=0.0 -> dist = 0.30m. norm_d = (0.30-0.20)/0.20 = 0.5
        # cost = 0.70 - 0.30 * 0.5 = 0.550
        {"id": "med_risk_mid", "x": 0.80, "y": 0.0, "expected_cost": 0.550, "expected_risk": 2, "is_nominal": False},
        # Low-risk zone: 0.40 < dist <= 0.60m. Point at x=1.00, y=0.0 -> dist = 0.50m. norm_d = (0.50-0.40)/0.20 = 0.5
        # cost = 0.40 - 0.25 * 0.5 = 0.275
        {"id": "low_risk_mid", "x": 1.00, "y": 0.0, "expected_cost": 0.275, "expected_risk": 1, "is_nominal": False},
        # Free space: dist > 0.60m. Point at x=1.50, y=0.0 -> dist = 1.0m
        {"id": "free_space", "x": 1.50, "y": 0.0, "expected_cost": 0.0, "expected_risk": 0, "is_nominal": False},
    ]

    return {
        "obs_pos": obs_pos,
        "obs_size": obs_size,
        "obs_yaw": obs_yaw,
        "r_unc": r_unc,
        "sigma_scale": sigma_scale,
        "query_points": query_points,
    }


def generate_collision_vectors() -> Dict[str, Any]:
    """Vectors for SAT convex polygon collision checking."""
    cases = []

    # Robot: length 0.8, width 0.6, yaw 0.0
    r_size = [0.8, 0.6]

    # Case 1: Obvious disjoint
    cases.append({
        "case_id": "disjoint_separated",
        "robot_pos": [0.0, 0.0],
        "robot_size": r_size,
        "robot_yaw": 0.0,
        "obs_pos": [5.0, 5.0],
        "obs_size": [1.0, 0.6],
        "obs_yaw": 0.0,
        "margin": 0.05,
        "expected_collision": False,
    })

    # Case 2: Direct overlap
    cases.append({
        "case_id": "direct_overlap",
        "robot_pos": [0.0, 0.0],
        "robot_size": r_size,
        "robot_yaw": 0.0,
        "obs_pos": [0.3, 0.2],
        "obs_size": [1.0, 0.6],
        "obs_yaw": 0.0,
        "margin": 0.05,
        "expected_collision": True,
    })

    # Case 3: Separated without margin, but colliding WITH margin
    # Robot x-extent: [-0.4, +0.4], Obstacle x-extent: [+0.43, +1.03]
    # Gap = 0.03m. Margin = 0.05m -> Expanded Robot x-extent: [-0.45, +0.45] -> Overlaps!
    cases.append({
        "case_id": "margin_induced_collision",
        "robot_pos": [0.0, 0.0],
        "robot_size": r_size,
        "robot_yaw": 0.0,
        "obs_pos": [0.73, 0.0],  # obs_half_l = 0.3. Min_x = 0.73 - 0.3 = 0.43. Robot max_x = 0.40. Gap = 0.03
        "obs_size": [0.6, 0.6],
        "obs_yaw": 0.0,
        "margin": 0.05,
        "expected_collision": True,
    })

    # Case 4: Gap just outside margin (0.06m > 0.05m)
    cases.append({
        "case_id": "margin_clearance_preserved",
        "robot_pos": [0.0, 0.0],
        "robot_size": r_size,
        "robot_yaw": 0.0,
        "obs_pos": [0.76, 0.0],  # Gap = 0.06m
        "obs_size": [0.6, 0.6],
        "obs_yaw": 0.0,
        "margin": 0.05,
        "expected_collision": False,
    })

    # Case 5: 45-degree rotated box collision
    cases.append({
        "case_id": "rotated_45_degree_collision",
        "robot_pos": [0.0, 0.0],
        "robot_size": r_size,
        "robot_yaw": math.pi / 4,
        "obs_pos": [0.5, 0.5],
        "obs_size": [0.8, 0.8],
        "obs_yaw": 0.0,
        "margin": 0.05,
        "expected_collision": True,
    })

    return {"vectors": cases}


def generate_dwa_vectors() -> Dict[str, Any]:
    """Vectors for DWA multi-objective score evaluation."""
    cases = []

    # Case 1: Ideal straight trajectory on path
    cases.append({
        "case_id": "ideal_nominal_straight",
        "path_dist": 0.05,
        "heading_err": 0.02,
        "dist_to_goal": 2.0,
        "clearance": 1.5,
        "velocity": 1.0,
        "max_v": 1.0,
        "terrain_cost": 0.10,
        "dynamic_cost": 0.0,
        "dynamic_risk": "FREE",
        "unknown_horizon": False,
    })

    # Case 2: Avoidance trajectory with lateral offset
    cases.append({
        "case_id": "avoidance_detour",
        "path_dist": 0.40,
        "heading_err": 0.15,
        "dist_to_goal": 2.2,
        "clearance": 0.8,
        "velocity": 0.8,
        "max_v": 1.0,
        "terrain_cost": 0.20,
        "dynamic_cost": 0.25,
        "dynamic_risk": "LOW",
        "unknown_horizon": False,
    })

    # Case 3: High-risk trajectory (penalty applied)
    cases.append({
        "case_id": "high_risk_penalty",
        "path_dist": 0.10,
        "heading_err": 0.05,
        "dist_to_goal": 1.8,
        "clearance": 0.2,
        "velocity": 1.0,
        "max_v": 1.0,
        "terrain_cost": 0.15,
        "dynamic_cost": 0.75,
        "dynamic_risk": "HIGH",
        "unknown_horizon": False,
    })

    return {"vectors": cases}


def generate_behavior_vectors() -> Dict[str, Any]:
    """Vectors for behavioral state transitions and hysteresis."""
    cases = [
        {
            "case_id": "nominal_proceed",
            "planner_status_is_stop": False,
            "emergency_stop": False,
            "valid_trajectories": 15,
            "dyn_collision": False,
            "unknown_horizon": False,
            "unknown_policy": "PENALIZE",
            "best_w": 0.01,
            "dyn_rejected": 0,
            "dyn_risk": "FREE",
            "clearing_obstacle": False,
            "current_state": "PROCEED",
            "dt_state": 1.0,
            "min_dwell_time_s": 0.20,
            "wait_elapsed_s": 0.0,
            "wait_timeout_s": 3.0,
            "recovery_count": 0,
            "recovery_threshold": 2,
            "expected_proposed": "PROCEED",
            "expected_next": "PROCEED",
            "expected_transition": "HOLD",
        },
        {
            "case_id": "avoidance_deviation",
            "planner_status_is_stop": False,
            "emergency_stop": False,
            "valid_trajectories": 12,
            "dyn_collision": False,
            "unknown_horizon": False,
            "unknown_policy": "PENALIZE",
            "best_w": 0.25,
            "dyn_rejected": 3,
            "dyn_risk": "MEDIUM",
            "clearing_obstacle": False,
            "current_state": "PROCEED",
            "dt_state": 0.5,
            "min_dwell_time_s": 0.20,
            "wait_elapsed_s": 0.0,
            "wait_timeout_s": 3.0,
            "recovery_count": 0,
            "recovery_threshold": 2,
            "expected_proposed": "AVOID",
            "expected_next": "AVOID",
            "expected_transition": "PROCEED -> AVOID",
        },
        {
            "case_id": "wait_clearing_obstacle",
            "planner_status_is_stop": True,
            "emergency_stop": False,
            "valid_trajectories": 0,
            "dyn_collision": False,
            "unknown_horizon": False,
            "unknown_policy": "PENALIZE",
            "best_w": 0.0,
            "dyn_rejected": 15,
            "dyn_risk": "HIGH",
            "clearing_obstacle": True,
            "current_state": "PROCEED",
            "dt_state": 0.5,
            "min_dwell_time_s": 0.20,
            "wait_elapsed_s": 0.0,
            "wait_timeout_s": 3.0,
            "recovery_count": 0,
            "recovery_threshold": 2,
            "expected_proposed": "WAIT",
            "expected_next": "WAIT",
            "expected_transition": "PROCEED -> WAIT",
        },
        {
            "case_id": "dwell_time_hold",
            "planner_status_is_stop": False,
            "emergency_stop": False,
            "valid_trajectories": 10,
            "dyn_collision": False,
            "unknown_horizon": False,
            "unknown_policy": "PENALIZE",
            "best_w": 0.15,
            "dyn_rejected": 2,
            "dyn_risk": "MEDIUM",
            "clearing_obstacle": False,
            "current_state": "PROCEED",
            "dt_state": 0.05,  # Less than min_dwell_time_s 0.20s
            "min_dwell_time_s": 0.20,
            "wait_elapsed_s": 0.0,
            "wait_timeout_s": 3.0,
            "recovery_count": 0,
            "recovery_threshold": 2,
            "expected_proposed": "AVOID",
            "expected_next": "PROCEED",
            "expected_transition": "HOLD_DWELL",
        },
        {
            "case_id": "wait_timeout_to_stop",
            "planner_status_is_stop": False,
            "emergency_stop": False,
            "valid_trajectories": 0,
            "dyn_collision": False,
            "unknown_horizon": False,
            "unknown_policy": "PENALIZE",
            "best_w": 0.0,
            "dyn_rejected": 15,
            "dyn_risk": "HIGH",
            "clearing_obstacle": True,
            "current_state": "WAIT",
            "dt_state": 3.2,
            "min_dwell_time_s": 0.20,
            "wait_elapsed_s": 3.1,  # Exceeds 3.0s timeout
            "wait_timeout_s": 3.0,
            "recovery_count": 0,
            "recovery_threshold": 2,
            "expected_proposed": "WAIT",
            "expected_next": "STOP",
            "expected_transition": "WAIT -> STOP",
        },
        {
            "case_id": "recovery_confirmation_first_cycle",
            "planner_status_is_stop": False,
            "emergency_stop": False,
            "valid_trajectories": 15,
            "dyn_collision": False,
            "unknown_horizon": False,
            "unknown_policy": "PENALIZE",
            "best_w": 0.0,
            "dyn_rejected": 0,
            "dyn_risk": "FREE",
            "clearing_obstacle": False,
            "current_state": "STOP",
            "dt_state": 1.0,
            "min_dwell_time_s": 0.20,
            "wait_elapsed_s": 0.0,
            "wait_timeout_s": 3.0,
            "recovery_count": 0,  # 0 < 2 -> HOLD_RECOVERY
            "recovery_threshold": 2,
            "expected_proposed": "PROCEED",
            "expected_next": "STOP",
            "expected_transition": "HOLD_RECOVERY",
        },
        {
            "case_id": "recovery_confirmation_second_cycle",
            "planner_status_is_stop": False,
            "emergency_stop": False,
            "valid_trajectories": 15,
            "dyn_collision": False,
            "unknown_horizon": False,
            "unknown_policy": "PENALIZE",
            "best_w": 0.0,
            "dyn_rejected": 0,
            "dyn_risk": "FREE",
            "clearing_obstacle": False,
            "current_state": "STOP",
            "dt_state": 1.1,
            "min_dwell_time_s": 0.20,
            "wait_elapsed_s": 0.0,
            "wait_timeout_s": 3.0,
            "recovery_count": 1,  # 1+1 >= 2 -> STOP -> PROCEED
            "recovery_threshold": 2,
            "expected_proposed": "PROCEED",
            "expected_next": "PROCEED",
            "expected_transition": "STOP -> PROCEED",
        },
    ]
    return {"vectors": cases}


def generate_replanner_vectors() -> Dict[str, Any]:
    """Vectors for global replanner trigger hierarchy."""
    cases = [
        {
            "case_id": "goal_reached_suppresses_replan",
            "is_goal_reached": True,
            "behavior_state": "STOP",
            "is_deadlock": True,
            "blocked_duration_s": 5.0,
            "persistent_threshold_s": 3.0,
            "is_path_valid": False,
            "path_invalid_cycles": 5,
            "debounce_threshold": 2,
            "dt_window_s": 5.0,
            "window_threshold_s": 4.0,
            "net_progress_m": 0.0,
            "min_progress_m": 0.20,
            "expected_replan": False,
            "expected_trigger": "NONE",
        },
        {
            "case_id": "dynamic_deadlock_trigger",
            "is_goal_reached": False,
            "behavior_state": "STOP",
            "is_deadlock": True,
            "blocked_duration_s": 0.5,
            "persistent_threshold_s": 3.0,
            "is_path_valid": True,
            "path_invalid_cycles": 0,
            "debounce_threshold": 2,
            "dt_window_s": 1.0,
            "window_threshold_s": 4.0,
            "net_progress_m": 0.1,
            "min_progress_m": 0.20,
            "expected_replan": True,
            "expected_trigger": "DYNAMIC_DEADLOCK",
        },
        {
            "case_id": "persistent_blockage_trigger",
            "is_goal_reached": False,
            "behavior_state": "WAIT",
            "is_deadlock": False,
            "blocked_duration_s": 3.2,
            "persistent_threshold_s": 3.0,
            "is_path_valid": True,
            "path_invalid_cycles": 0,
            "debounce_threshold": 2,
            "dt_window_s": 3.2,
            "window_threshold_s": 4.0,
            "net_progress_m": 0.0,
            "min_progress_m": 0.20,
            "expected_replan": True,
            "expected_trigger": "PERSISTENT_BLOCKAGE",
        },
        {
            "case_id": "path_invalidated_debounced",
            "is_goal_reached": False,
            "behavior_state": "STOP",
            "is_deadlock": False,
            "blocked_duration_s": 1.0,
            "persistent_threshold_s": 3.0,
            "is_path_valid": False,
            "path_invalid_cycles": 1,  # 1+1 >= 2
            "debounce_threshold": 2,
            "dt_window_s": 1.0,
            "window_threshold_s": 4.0,
            "net_progress_m": 0.0,
            "min_progress_m": 0.20,
            "expected_replan": True,
            "expected_trigger": "PATH_INVALIDATED",
        },
        {
            "case_id": "path_invalidated_not_yet_debounced",
            "is_goal_reached": False,
            "behavior_state": "STOP",
            "is_deadlock": False,
            "blocked_duration_s": 0.1,
            "persistent_threshold_s": 3.0,
            "is_path_valid": False,
            "path_invalid_cycles": 0,  # 0+1 < 2
            "debounce_threshold": 2,
            "dt_window_s": 0.1,
            "window_threshold_s": 4.0,
            "net_progress_m": 0.0,
            "min_progress_m": 0.20,
            "expected_replan": False,
            "expected_trigger": "NONE",
        },
        {
            "case_id": "insufficient_progress_trigger",
            "is_goal_reached": False,
            "behavior_state": "STOP",
            "is_deadlock": False,
            "blocked_duration_s": 1.5,
            "persistent_threshold_s": 3.0,
            "is_path_valid": True,
            "path_invalid_cycles": 0,
            "debounce_threshold": 2,
            "dt_window_s": 4.5,
            "window_threshold_s": 4.0,
            "net_progress_m": 0.05,  # 0.05 < 0.20
            "min_progress_m": 0.20,
            "expected_replan": True,
            "expected_trigger": "INSUFFICIENT_PROGRESS",
        },
    ]
    return {"vectors": cases}


def main():
    print(f"Generating deterministic test vectors in {OUTPUT_DIR}...")

    files = {
        "motion_estimation_vectors.json": generate_motion_estimation_vectors(),
        "kalman_vectors.json": generate_kalman_vectors(),
        "mahalanobis_vectors.json": generate_mahalanobis_vectors(),
        "prediction_vectors.json": generate_prediction_vectors(),
        "occupancy_vectors.json": generate_occupancy_vectors(),
        "collision_vectors.json": generate_collision_vectors(),
        "dwa_vectors.json": generate_dwa_vectors(),
        "behavior_vectors.json": generate_behavior_vectors(),
        "replanner_vectors.json": generate_replanner_vectors(),
    }

    for fname, data in files.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {fname} ({os.path.getsize(fpath)} bytes)")

    print("All 9 test vector files generated successfully.")


if __name__ == "__main__":
    main()
