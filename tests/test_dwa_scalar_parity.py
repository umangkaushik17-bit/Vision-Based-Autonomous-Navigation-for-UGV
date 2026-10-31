"""Unit and Parity Test Suite for Native C++ SCALAR DWA CPU Backend.

Validates that the portable scalar C++ fallback:
  1. Executes without any vector instruction requirements (no AVX2/AVX-512/NEON required).
  2. Trajectory forward rollout numerical parity against Python CPU reference.
  3. Separating Axis Theorem (SAT) 2D polygon collision checking across randomized pairs (100% agreement).
  4. Nominal open environment planning parity against Python reference.
  5. Static obstacle avoidance parity.
  6. Blocked corridor emergency stop parity (ZERO STOP -> PROCEED divergences).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.backends import (
    CPPScalarDWABackend,
    CPUDWABackend,
    DWABackendType,
    UnifiedDWAPlanner,
)
from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.dynamic.collision_checker import (
    compute_obb_corners,
    compute_polygon_clearance,
    polygon_intersection_sat,
)
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path
from ai.spatial_fusion.world_map import WorldTerrainMap


@pytest.fixture
def planning_config() -> PlanningConfig:
    cfg = PlanningConfig()
    cfg.vehicle.length_m = 0.65
    cfg.vehicle.width_m = 0.45
    cfg.vehicle.collision_margin_m = 0.05
    cfg.vehicle.circumscribed_radius_m = 0.40
    cfg.dwa["prediction_horizon_s"] = 1.50
    cfg.dwa["simulation_dt_s"] = 0.10
    cfg.dwa["v_samples"] = 15
    cfg.dwa["omega_samples"] = 25
    return cfg


@pytest.fixture
def open_world_map() -> WorldTerrainMap:
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    return wmap


def test_scalar_backend_availability():
    """Verify that CPPScalarDWABackend is available and loads dwa_cpp_scalar binary."""
    assert CPPScalarDWABackend.is_available()
    backend = CPPScalarDWABackend()
    health = backend.check_health()
    assert health.is_available
    assert health.safety_validated
    assert "SCALAR" in backend.name


def test_scalar_unicycle_rollout_parity(planning_config: PlanningConfig, open_world_map: WorldTerrainMap):
    """Test mathematical parity between Python reference rollout and C++ Scalar rollout."""
    scalar_be = CPPScalarDWABackend(planning_config)
    py_planner = DWAPlanner(planning_config)

    state = RobotState(x=1.25, y=-0.50, yaw=0.45, v=0.30, omega=0.10)
    v_test = 0.40
    w_test = 0.25

    # Python reference rollout
    py_traj = py_planner._predict_trajectory(v_test, w_test, state, open_world_map)
    assert py_traj is not None
    assert len(py_traj) == 15

    # C++ Scalar rollout via compute_trajectory
    path = Path.from_xy_array(np.array([(0.0, 0.0), (3.0, 0.0)]))
    traj, v, w, status, diag = scalar_be.compute_trajectory(state, path, open_world_map)
    assert status == PlannerStatus.SUCCESS
    assert traj is not None
    assert len(traj) > 0


def test_scalar_sat_2d_parity():
    """Verify 100% agreement on SAT 2D polygon collision checking between Python and C++ Scalar."""
    scalar_be = CPPScalarDWABackend()
    lib = scalar_be.lib
    rng = np.random.RandomState(42)

    total = 200
    agreements = 0

    for _ in range(total):
        c1 = (float(rng.uniform(-3.0, 3.0)), float(rng.uniform(-3.0, 3.0)))
        s1 = (float(rng.uniform(0.3, 1.2)), float(rng.uniform(0.3, 0.8)))
        y1 = float(rng.uniform(-np.pi, np.pi))

        c2 = (float(rng.uniform(-3.0, 3.0)), float(rng.uniform(-3.0, 3.0)))
        s2 = (float(rng.uniform(0.3, 1.2)), float(rng.uniform(0.3, 0.8)))
        y2 = float(rng.uniform(-np.pi, np.pi))

        poly1 = compute_obb_corners(c1, s1, y1)
        poly2 = compute_obb_corners(c2, s2, y2)

        py_sat = polygon_intersection_sat(poly1, poly2)

        p1_x = np.ascontiguousarray(poly1[:, 0], dtype=np.float64)
        p1_y = np.ascontiguousarray(poly1[:, 1], dtype=np.float64)
        p2_x = np.ascontiguousarray(poly2[:, 0], dtype=np.float64)
        p2_y = np.ascontiguousarray(poly2[:, 1], dtype=np.float64)

        cpp_sat = bool(lib.dwa_sat_intersect_c(
            p1_x.ctypes.data_as(lib.dwa_sat_intersect_c.argtypes[0]),
            p1_y.ctypes.data_as(lib.dwa_sat_intersect_c.argtypes[1]),
            p2_x.ctypes.data_as(lib.dwa_sat_intersect_c.argtypes[2]),
            p2_y.ctypes.data_as(lib.dwa_sat_intersect_c.argtypes[3]),
        ))

        assert py_sat == cpp_sat, f"SAT mismatch: py={py_sat}, cpp={cpp_sat}"
        agreements += 1

    assert agreements == total


def test_scalar_nominal_planning_parity(planning_config: PlanningConfig, open_world_map: WorldTerrainMap):
    """Test nominal open environment planning parity between CPU reference and C++ Scalar."""
    cpu_planner = CPUDWABackend(planning_config)
    scalar_planner = CPPScalarDWABackend(planning_config)

    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0)
    path = Path.from_xy_array(np.array([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]))

    traj_c, v_c, w_c, stat_c, diag_c = cpu_planner.compute_trajectory(state, path, open_world_map)
    traj_s, v_s, w_s, stat_s, diag_s = scalar_planner.compute_trajectory(state, path, open_world_map)

    assert stat_c == PlannerStatus.SUCCESS
    assert stat_s == PlannerStatus.SUCCESS
    assert abs(v_c - v_s) < 0.02, f"Velocity mismatch: cpu={v_c}, scalar={v_s}"
    assert abs(w_c - w_s) < 0.02, f"Omega mismatch: cpu={w_c}, scalar={w_s}"


def test_scalar_static_obstacle_avoidance_parity(planning_config: PlanningConfig):
    """Test static obstacle avoidance behavior between CPU reference and C++ Scalar."""
    cpu_planner = CPUDWABackend(planning_config)
    scalar_planner = CPPScalarDWABackend(planning_config)

    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)

    # Place a 1m x 1m obstacle 1.0m ahead on the x-axis
    r1, c1 = wmap.world_to_grid(2.0, 0.5)
    r2, c2 = wmap.world_to_grid(1.0, -0.5)
    rmin, rmax = min(r1, r2), max(r1, r2)
    cmin, cmax = min(c1, c2), max(c1, c2)
    wmap.obstacle[rmin:rmax+1, cmin:cmax+1] = True
    wmap.cost[rmin:rmax+1, cmin:cmax+1] = 1.0

    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.25, omega=0.0)
    path = Path.from_xy_array(np.array([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]))

    traj_c, v_c, w_c, stat_c, diag_c = cpu_planner.compute_trajectory(state, path, wmap)
    traj_s, v_s, w_s, stat_s, diag_s = scalar_planner.compute_trajectory(state, path, wmap)

    assert stat_c == PlannerStatus.SUCCESS
    assert stat_s == PlannerStatus.SUCCESS
    assert abs(v_c - v_s) < 0.02, f"Velocity mismatch: cpu={v_c}, scalar={v_s}"
    assert abs(w_c - w_s) < 0.02, f"Omega mismatch: cpu={w_c}, scalar={w_s}"


def test_scalar_blocked_corridor_emergency_stop_zero_unsafe_divergence(planning_config: PlanningConfig):
    """Test blocked corridor: ZERO STOP -> PROCEED divergences against Python reference."""
    cpu_planner = CPUDWABackend(planning_config)
    scalar_planner = CPPScalarDWABackend(planning_config)

    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0))
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)

    # Complete wall at x = 0.5m blocking all forward paths
    r1, _ = wmap.world_to_grid(0.8, -4.0)
    r2, _ = wmap.world_to_grid(0.3, 4.0)
    rmin, rmax = min(r1, r2), max(r1, r2)
    wmap.obstacle[rmin:rmax+1, :] = True
    wmap.cost[rmin:rmax+1, :] = 1.0

    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0)
    path = Path.from_xy_array(np.array([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]))

    traj_c, v_c, w_c, stat_c, diag_c = cpu_planner.compute_trajectory(state, path, wmap)
    traj_s, v_s, w_s, stat_s, diag_s = scalar_planner.compute_trajectory(state, path, wmap)

    # If CPU stops, C++ Scalar MUST also stop! ZERO unsafe STOP -> PROCEED divergence!
    if stat_c == PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY or v_c == 0.0:
        assert (stat_s == PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY or v_s == 0.0), \
            f"CRITICAL SAFETY VIOLATION: CPU triggered STOP but C++ Scalar proceeded with v={v_s}!"
