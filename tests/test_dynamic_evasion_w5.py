"""Unit tests verifying proactive continuous dynamic obstacle evasion across W1 to W5,

specifically focusing on W5 corridor clearance and stationary obstacle avoidance.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.path import Path


def compute_evasion_bypass(
    path: Path,
    ugv_pose: tuple[float, float, float],
    ugv_v: float,
    obstacles: list[dict],
    static_boulders: list[dict],
    latches: dict,
) -> tuple[Path, dict]:
    """Pure mathematical simulation of the dynamic obstacle evasion algorithm

    in interactive_ugv_navigation_node.py.
    """
    ugv_x, ugv_y, ugv_yaw = ugv_pose
    cos_yaw = math.cos(ugv_yaw)
    sin_yaw = math.sin(ugv_yaw)

    path_pts = path.waypoints_xy
    diffs = np.diff(path_pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_s = np.concatenate(([0.0], np.cumsum(seg_lens)))

    dists_ugv = np.hypot(path_pts[:, 0] - ugv_x, path_pts[:, 1] - ugv_y)
    ugv_idx = int(np.argmin(dists_ugv))
    s_ugv = float(cum_s[ugv_idx])

    effective_path = path
    evasion_info = {}

    sorted_tracks = sorted(obstacles, key=lambda t: math.hypot(t["x"] - ugv_x, t["y"] - ugv_y))

    for tr in sorted_tracks:
        dx = tr["x"] - ugv_x
        dy = tr["y"] - ugv_y
        x_rel = dx * cos_yaw + dy * sin_yaw
        y_rel = -dx * sin_yaw + dy * cos_yaw

        vx = tr.get("vx", 0.0)
        vy = tr.get("vy", 0.0)
        vx_rel = vx * cos_yaw + vy * sin_yaw
        closing_speed = ugv_v - vx_rel

        dists_obs = np.hypot(path_pts[:, 0] - tr["x"], path_pts[:, 1] - tr["y"])
        obs_idx = int(np.argmin(dists_obs))
        s_obs = float(cum_s[obs_idx])

        idx_next = min(len(path_pts) - 1, obs_idx + 1)
        idx_prev = max(0, obs_idx - 1)
        tx = float(path_pts[idx_next, 0] - path_pts[idx_prev, 0])
        ty = float(path_pts[idx_next, 1] - path_pts[idx_prev, 1])
        t_norm = math.hypot(tx, ty)
        if t_norm > 1e-3:
            tx /= t_norm
            ty /= t_norm
        else:
            tx, ty = 1.0, 0.0
        nx, ny = -ty, tx

        dx_obs = tr["x"] - float(path_pts[obs_idx, 0])
        dy_obs = tr["y"] - float(path_pts[obs_idx, 1])
        d_lat_obs = dx_obs * nx + dy_obs * ny

        s_dist_ahead = s_obs - s_ugv
        is_ahead = (-0.5 < s_dist_ahead < 20.0) and (x_rel > 0.5)
        in_corridor = (abs(d_lat_obs) < 2.5) or (abs(y_rel) < 2.2)
        is_threatening = (closing_speed > -0.15) or (s_dist_ahead < 12.0) or (x_rel < 12.0)

        if is_ahead and in_corridor and is_threatening:
            tr_id = str(tr.get("id", "obs"))

            p_left = (float(path_pts[obs_idx, 0]) + 1.40 * nx, float(path_pts[obs_idx, 1]) + 1.40 * ny)
            p_right = (float(path_pts[obs_idx, 0]) - 1.40 * nx, float(path_pts[obs_idx, 1]) - 1.40 * ny)

            obs_radius = max(tr.get("bbox_size", (0.8, 0.8))) / 2.0
            d_obs_self_left = math.hypot(p_left[0] - tr["x"], p_left[1] - tr["y"]) - obs_radius
            d_obs_self_right = math.hypot(p_right[0] - tr["x"], p_right[1] - tr["y"]) - obs_radius

            min_boulder_left = 999.0
            min_boulder_right = 999.0
            for b in static_boulders:
                bpos = b.get("position", [0.0, 0.0])
                bradius = float(b.get("bounding_radius_m", 0.70))
                min_boulder_left = min(min_boulder_left, math.hypot(p_left[0] - bpos[0], p_left[1] - bpos[1]) - bradius)
                min_boulder_right = min(min_boulder_right, math.hypot(p_right[0] - bpos[0], p_right[1] - bpos[1]) - bradius)

            d_other_left = 999.0
            d_other_right = 999.0
            for other_tr in obstacles:
                if str(other_tr.get("id", "obs")) == tr_id:
                    continue
                other_r = max(other_tr.get("bbox_size", (0.8, 0.8))) / 2.0
                d_other_left = min(d_other_left, math.hypot(p_left[0] - other_tr["x"], p_left[1] - other_tr["y"]) - other_r)
                d_other_right = min(d_other_right, math.hypot(p_right[0] - other_tr["x"], p_right[1] - other_tr["y"]) - other_r)

            clearance_left = min(d_obs_self_left, min_boulder_left, d_other_left)
            clearance_right = min(d_obs_self_right, min_boulder_right, d_other_right)

            latch = latches.get(tr_id)
            if latch is not None and s_ugv < latch["expire_s"]:
                latched_dir = latch["dir"]
                latched_clr = clearance_left if latched_dir > 0 else clearance_right
                if latched_clr < 0.60:
                    latch = None

            if latch is not None and s_ugv < latch["expire_s"]:
                evade_offset_m = latch["offset"]
                evade_dir = latch["dir"]
            else:
                if clearance_left < 0.90 and clearance_right >= 0.90:
                    evade_dir = -1.0
                elif clearance_right < 0.90 and clearance_left >= 0.90:
                    evade_dir = 1.0
                else:
                    if abs(d_lat_obs) > 0.20:
                        evade_dir = -1.0 if d_lat_obs >= 0.0 else 1.0
                    else:
                        evade_dir = 1.0 if clearance_left > clearance_right else -1.0

                evade_offset_m = evade_dir * 1.40
                latches[tr_id] = {
                    "offset": evade_offset_m,
                    "dir": evade_dir,
                    "expire_s": s_obs + 3.0,
                }

            xy_mod = path_pts.copy()
            sigma = 2.8
            two_sigma_sq = 2.0 * sigma * sigma

            for i in range(len(xy_mod)):
                ds = float(cum_s[i] - s_obs)
                w = math.exp(-(ds * ds) / two_sigma_sq)
                dist_to_end = float(cum_s[-1] - cum_s[i])
                if dist_to_end < 3.0:
                    w *= max(0.0, dist_to_end / 3.0)

                if w > 0.01:
                    i_next = min(len(path_pts) - 1, i + 1)
                    i_prev = max(0, i - 1)
                    itx = float(path_pts[i_next, 0] - path_pts[i_prev, 0])
                    ity = float(path_pts[i_next, 1] - path_pts[i_prev, 1])
                    it_norm = math.hypot(itx, ity)
                    if it_norm > 1e-3:
                        inx = -ity / it_norm
                        iny = itx / it_norm
                    else:
                        inx, iny = 0.0, 1.0
                    xy_mod[i, 0] += evade_offset_m * w * inx
                    xy_mod[i, 1] += evade_offset_m * w * iny

            effective_path = Path.from_xy_array(xy_mod)
            evasion_info = {
                "tr_id": tr_id,
                "evade_dir": evade_dir,
                "evade_offset_m": evade_offset_m,
                "side": "RIGHT" if evade_dir < 0 else "LEFT",
                "d_lat_obs": d_lat_obs,
                "clearance": clearance_right if evade_dir < 0 else clearance_left,
                "p_chosen": p_right if evade_dir < 0 else p_left,
            }
            break

    return effective_path, evasion_info


def test_w5_stationary_obstacle_evasion():
    """Verify that in segment W5 (80, 3) -> (100, 0), with stationary obstacle at (94, 2.5),

    evasion chooses RIGHT detour with clearance > 2.0m, strictly avoiding the obstacle.
    """
    w4 = np.array([80.0, 3.0])
    w5 = np.array([100.0, 0.0])
    nominal_path = Path.from_xy_array(np.linspace(w4, w5, 43))

    # Dynamic obstacle 5 stationary at standby pose
    obs_w5 = [{"id": "w5", "x": 94.0, "y": 2.5, "vx": 0.0, "vy": 0.0, "bbox_size": (0.8, 0.8)}]
    boulders = [
        {"position": [97.5, -3.2], "bounding_radius_m": 0.60},
        {"position": [97.5, 3.2], "bounding_radius_m": 0.60},
    ]

    # UGV entering W5 corridor from W4 (pose similar to log)
    ugv_pose = (80.26, 3.33, 0.04)
    ugv_v = 1.0
    latches = {}

    bypassed_path, info = compute_evasion_bypass(nominal_path, ugv_pose, ugv_v, obs_w5, boulders, latches)

    assert info["side"] == "RIGHT", f"Expected RIGHT bypass in W5, got {info['side']}"
    assert info["evade_offset_m"] == -1.40
    assert info["clearance"] >= 2.0, f"Clearance too low: {info['clearance']}"

    # Minimum distance from any waypoint on the bypassed path to the obstacle center
    pts = bypassed_path.waypoints_xy
    min_dist_to_obs = float(np.min(np.hypot(pts[:, 0] - 94.0, pts[:, 1] - 2.5)))
    assert min_dist_to_obs >= 2.40, f"Path passes too close to obstacle: {min_dist_to_obs:.2f}m"

    # Verify that the path converges exactly to the final destination (100, 0)
    assert pts[-1, 0] == pytest.approx(100.0, abs=1e-3)
    assert pts[-1, 1] == pytest.approx(0.0, abs=1e-3)


def test_w1_to_w4_evasion_preservation():
    """Verify that W1, W2, W3, and W4 maintain their successful bypass directions."""
    scenarios = [
        ("W1", (0.0, 0.0), (20.0, 0.0), (1.12, 0.02, 0.07), {"id": "w1", "x": 16.0, "y": 2.15}, "RIGHT"),
        ("W2", (20.0, 0.0), (40.0, 6.0), (25.8, 1.9, 0.21), {"id": "w2", "x": 32.0, "y": 3.9}, "RIGHT"),
        ("W3", (40.0, 6.0), (60.0, -4.0), (47.3, 2.4, -0.40), {"id": "w3", "x": 55.0, "y": -3.0}, "LEFT"),
        ("W4", (60.0, -4.0), (80.0, 3.0), (61.7, -4.0, 0.52), {"id": "w4", "x": 73.0, "y": 4.8}, "RIGHT"),
    ]

    for name, start_pt, goal_pt, ugv_pose, obs_data, expected_side in scenarios:
        path = Path.from_xy_array(np.linspace(np.array(start_pt), np.array(goal_pt), 43))
        obs = [obs_data]
        latches = {}
        _, info = compute_evasion_bypass(path, ugv_pose, 1.0, obs, [], latches)
        assert bool(info), f"{name} should trigger evasion"
        assert info["side"] == expected_side, f"{name} expected {expected_side}, got {info['side']}"
        assert info["clearance"] >= 1.0, f"{name} clearance too low: {info['clearance']}"


def test_w5_moving_obstacle_adaptation():
    """Verify that if W5 dynamic obstacle moves to the right side of corridor,

    evasion adapts and safely detours to the LEFT.
    """
    w4 = np.array([80.0, 3.0])
    w5 = np.array([100.0, 0.0])
    nominal_path = Path.from_xy_array(np.linspace(w4, w5, 43))

    # Obstacle on the right side of the path (at y = -1.5)
    obs_w5_right = [{"id": "w5", "x": 94.0, "y": -1.5, "vx": 0.0, "vy": 0.0, "bbox_size": (0.8, 0.8)}]
    boulders = [
        {"position": [97.5, -3.2], "bounding_radius_m": 0.60},
        {"position": [97.5, 3.2], "bounding_radius_m": 0.60},
    ]

    ugv_pose = (80.26, 3.33, 0.04)
    latches = {}
    bypassed_path, info = compute_evasion_bypass(nominal_path, ugv_pose, 1.0, obs_w5_right, boulders, latches)

    assert info["side"] == "LEFT", f"Expected LEFT bypass when obstacle is on right, got {info['side']}"
    assert info["evade_offset_m"] == 1.40
    assert info["clearance"] >= 2.0
