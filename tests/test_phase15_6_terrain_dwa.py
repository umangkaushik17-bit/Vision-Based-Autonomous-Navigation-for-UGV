"""Unit and integration tests for Subphase 15.6: Terrain-Aware Native C++17 AVX2 DWA & Quasi-Static Rollover Margin.

Verifies:
1. Quasi-Static Rollover Stability Margin model:
   - Allows low-speed aligned traversal on 22-24 deg slopes (margin >= 0.20).
   - Rejects high-speed turning arcs on 22-24 deg slopes (margin < 0.20).
   - Rejects all trajectories on >= 28 deg critical slopes.
2. Native C++ DWA (SIMD & Scalar) and Python DWA reject tip-over arcs on 22-24 deg slopes
   while selecting safe low-speed aligned trajectories.
3. Slope-adaptive velocity derating caps linear velocity to v <= v_max * max(0.25, cos(alpha)).
4. Cross-slope valley selection: DWA steers away from a 24 deg cross-slope bank toward a flat valley floor.
5. Lethal >= 28 deg embankment or >= 0.12 m step barrier ahead triggers safe stop (v = 0.0).
6. Native C++ AVX2 DWA P95 execution latency < 5.0 ms (mean < 3.5 ms) across 375 candidates.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.backends import CPPDWABackend, CPUDWABackend
from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path
from ai.spatial_fusion.world_map import WorldTerrainMap


def _make_dwa_config() -> PlanningConfig:
    cfg = PlanningConfig()
    cfg.vehicle.length_m = 0.65
    cfg.vehicle.width_m = 0.45
    cfg.vehicle.collision_margin_m = 0.05
    cfg.vehicle.circumscribed_radius_m = 0.40
    cfg.vehicle.max_linear_velocity_mps = 0.80
    cfg.vehicle.min_linear_velocity_mps = 0.0
    cfg.vehicle.max_angular_velocity_radps = 1.20
    cfg.vehicle.max_linear_acceleration_mps2 = 1.00
    cfg.vehicle.max_angular_acceleration_radps2 = 2.50
    cfg.dwa["prediction_horizon_s"] = 1.50
    cfg.dwa["simulation_dt_s"] = 0.10
    cfg.dwa["v_samples"] = 15
    cfg.dwa["omega_samples"] = 25
    return cfg


def test_quasi_static_rollover_margin_decision_model():
    """Verify mathematical boundaries of the Quasi-Static Rollover Stability Margin equation."""
    # 1. Flat ground (0 deg): even high speed + sharp turn is well above 0.20 margin
    m_flat = DWAPlanner.compute_rollover_margin(slope_deg=0.0, v=0.80, omega=1.00)
    assert m_flat >= 0.85

    # 2. 23 deg slope with low-speed aligned motion (v=0.25, omega=0.05): safe (margin >= 0.20)
    m_aligned_23 = DWAPlanner.compute_rollover_margin(slope_deg=23.0, v=0.25, omega=0.05)
    assert m_aligned_23 >= 0.20, f"Expected aligned 23 deg margin >= 0.20, got {m_aligned_23:.3f}"

    # 3. 23 deg slope with high-speed sharp turn (v=0.65, omega=1.00): rollover hazard (margin < 0.20)
    m_turn_23 = DWAPlanner.compute_rollover_margin(slope_deg=23.0, v=0.65, omega=1.00)
    assert m_turn_23 < 0.20, f"Expected high-speed turn on 23 deg to have margin < 0.20, got {m_turn_23:.3f}"

    # 4. 24 deg slope with low-speed aligned motion vs high-speed turn
    m_aligned_24 = DWAPlanner.compute_rollover_margin(slope_deg=24.0, v=0.25, omega=0.0)
    m_turn_24 = DWAPlanner.compute_rollover_margin(slope_deg=24.0, v=0.55, omega=0.85)
    assert m_aligned_24 >= 0.20
    assert m_turn_24 < 0.20

    # 5. Critical slope (>= 28 deg) is always rejected (< 0.20) even at v=0, omega=0
    m_crit = DWAPlanner.compute_rollover_margin(slope_deg=28.0, v=0.0, omega=0.0)
    assert m_crit < 0.20


@pytest.mark.parametrize("slope_deg", [22.0, 23.0, 24.0])
def test_cpp_and_py_dwa_reject_high_speed_turns_on_steep_slopes(slope_deg: float):
    """Verify C++ DWA and Python DWA reject high-yaw-rate turning arcs on 22-24 deg slopes
    while allowing low-speed aligned traversal.
    """
    cfg = _make_dwa_config()
    cpp_dwa = CPPDWABackend(cfg, variant="simd")
    sc_dwa = CPPDWABackend(cfg, variant="scalar")
    py_dwa = CPUDWABackend(cfg)

    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 10.0), y_bounds_m=(-5.0, 5.0))
    wmap.unknown.fill(False)
    wmap.cost.fill(0.25)
    wmap.obstacle.fill(False)
    wmap.traversability.fill(0.85)
    wmap.slope.fill(slope_deg)

    # Robot is cruising at v=0.55 m/s on a 22-24 deg slope along a straight path
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.55, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]], dtype=np.float32))

    traj_cpp, v_cpp, w_cpp, st_cpp, diag_cpp = cpp_dwa.compute_trajectory(state, path, wmap)
    traj_sc, v_sc, w_sc, st_sc, diag_sc = sc_dwa.compute_trajectory(state, path, wmap)
    traj_py, v_py, w_py, st_py, diag_py = py_dwa.compute_trajectory(state, path, wmap)

    assert st_cpp == PlannerStatus.SUCCESS
    assert st_sc == PlannerStatus.SUCCESS
    assert st_py == PlannerStatus.SUCCESS

    # High-turn-rate candidates must have been rejected by the rollover margin gate
    assert diag_cpp["valid_trajectories"] < diag_cpp["trajectories_evaluated"]
    assert diag_cpp["valid_trajectories"] == diag_sc["valid_trajectories"]

    # Selected trajectory must be aligned (|w| < 0.45 rad/s) and satisfy Rollover Margin >= 0.20
    assert abs(w_cpp) < 0.45
    assert DWAPlanner.compute_rollover_margin(slope_deg, v_cpp, w_cpp) >= 0.20
    assert DWAPlanner.compute_rollover_margin(slope_deg, v_py, w_py) >= 0.20


@pytest.mark.parametrize("slope_deg", [10.0, 16.0, 22.0, 24.0])
def test_slope_adaptive_velocity_derating(slope_deg: float):
    """Verify candidate velocity is automatically derated to v <= v_max * max(0.25, cos(alpha))."""
    cfg = _make_dwa_config()
    cpp_dwa = CPPDWABackend(cfg)
    py_dwa = CPUDWABackend(cfg)

    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 10.0), y_bounds_m=(-5.0, 5.0))
    wmap.unknown.fill(False)
    wmap.cost.fill(0.15)
    wmap.obstacle.fill(False)
    wmap.slope.fill(slope_deg)

    # Start near v_max (0.80 m/s) so dynamic window would reach 0.80 m/s without slope derating
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.78, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    _, v_cpp, _, st_cpp, _ = cpp_dwa.compute_trajectory(state, path, wmap)
    _, v_py, _, st_py, _ = py_dwa.compute_trajectory(state, path, wmap)

    assert st_cpp == PlannerStatus.SUCCESS
    assert st_py == PlannerStatus.SUCCESS

    v_expected_cap = cfg.vehicle.max_linear_velocity_mps * max(0.25, math.cos(math.radians(slope_deg)))
    assert v_cpp <= v_expected_cap + 1e-4, f"C++ v={v_cpp:.4f} exceeded slope cap {v_expected_cap:.4f} at {slope_deg} deg"
    assert v_py <= v_expected_cap + 1e-4, f"Python v={v_py:.4f} exceeded slope cap {v_expected_cap:.4f} at {slope_deg} deg"
    assert abs(v_cpp - v_py) < 0.02


def test_dwa_avoids_severe_cross_slope_for_valley_floor():
    """Verify C++ DWA steers away from a 25 deg lateral cross-slope bank toward the flat valley floor."""
    cfg = _make_dwa_config()
    cpp_dwa = CPPDWABackend(cfg)

    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-2.0, 8.0), y_bounds_m=(-4.0, 4.0))
    wmap.unknown.fill(False)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.traversability.fill(0.90)

    # Left half of corridor (y >= 0.05) is a severe 25 deg cross-slope bank; right side (y < 0.05) is flat 0 deg valley
    for r in range(wmap.n_rows):
        for c in range(wmap.n_cols):
            _, yw = wmap.grid_to_world(r, c)
            if yw >= 0.05:
                wmap.slope[r, c] = 25.0
                wmap.cost[r, c] = 0.55

    state = RobotState(x=0.0, y=-0.25, yaw=0.20, v=0.35, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, -0.6], [3.0, -0.6], [6.0, -0.6]], dtype=np.float32))

    traj, v_cmd, w_cmd, status, _ = cpp_dwa.compute_trajectory(state, path, wmap)
    assert status == PlannerStatus.SUCCESS
    assert traj is not None
    # Vehicle must steer right (omega < 0, end_y < 0) to stay on the flat valley floor away from the 25 deg bank
    assert w_cmd < 0.0
    assert traj.points[-1].y < -0.10


def test_dwa_halts_before_lethal_28deg_slope_or_15cm_step():
    """Verify C++ DWA triggers a safe stop when all forward trajectories hit >= 28 deg slope or >= 0.12m step."""
    cfg = _make_dwa_config()
    cpp_dwa = CPPDWABackend(cfg)

    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-2.0, 6.0), y_bounds_m=(-4.0, 4.0))
    wmap.unknown.fill(False)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    # Place a 30 deg slope wall across x >= 0.25m (without setting obstacle=True, testing pure slope gate)
    for r in range(wmap.n_rows):
        xw, _ = wmap.grid_to_world(r, 0)
        if xw >= 0.20:
            wmap.slope[r, :] = 30.0

    # Set v=0.60 so v_low = max(0, 0.60 - 1.0*0.1) = 0.50 m/s -> all candidates move forward into the 30 deg wall
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.60, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32))

    _, v_cmd, _, status, diag = cpp_dwa.compute_trajectory(state, path, wmap)
    assert status == PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY
    assert v_cmd == 0.0
    assert diag["emergency_stop_triggered"] is True


def test_terrain_dwa_cpp_latency_budget():
    """Verify C++17 AVX2 DWA executes in < 3.5 ms mean and < 5.0 ms P95 on 2.5D uneven terrain."""
    cfg = _make_dwa_config()
    cpp_dwa = CPPDWABackend(cfg, variant="simd")

    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    wmap.unknown.fill(False)
    wmap.cost.fill(0.15)
    wmap.obstacle.fill(False)
    wmap.slope.fill(18.0)
    wmap.roughness.fill(0.03)
    wmap.height_var.fill(0.04)

    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.45, omega=0.05)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]], dtype=np.float32))

    # Warmup
    _ = cpp_dwa.compute_trajectory(state, path, wmap)

    core_latencies = []
    for _ in range(25):
        _, _, _, status, diag = cpp_dwa.compute_trajectory(state, path, wmap)
        assert status == PlannerStatus.SUCCESS
        core_latencies.append(float(diag["cpp_core_latency_ms"]))

    mean_ms = float(np.mean(core_latencies))
    p95_ms = float(np.percentile(core_latencies, 95))
    assert mean_ms < 3.5, f"C++ DWA mean latency {mean_ms:.2f}ms exceeds 3.5ms budget"
    assert p95_ms < 5.0, f"C++ DWA P95 latency {p95_ms:.2f}ms exceeds 5.0ms budget"
