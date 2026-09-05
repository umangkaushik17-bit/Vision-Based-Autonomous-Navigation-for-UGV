"""Unit tests for Phase 12 DWA Local Planner across WorldTerrainMap."""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.base import GoalPose, PlanningConfig, PlannerStatus, RobotState
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path
from ai.spatial_fusion.world_map import WorldTerrainMap


def create_empty_world_map(resolution: float = 0.10, bounds: tuple[float, float] = (-5.0, 5.0)) -> WorldTerrainMap:
    wmap = WorldTerrainMap(resolution_m=resolution, x_bounds_m=bounds, y_bounds_m=bounds)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    return wmap


def test_dwa_dynamic_window_bounds():
    """Verify velocity sampling stays within acceleration bounds and velocity limits."""
    dwa = DWAPlanner()
    st = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.5, omega=0.0)

    v_low, v_high, w_low, w_high = dwa._compute_dynamic_window(st, dt=0.1)
    # v_high <= 0.5 + 1.5 * 0.1 = 0.65
    assert v_high <= 0.65 + 1e-4
    assert v_low >= 0.5 - 1.5 * 0.1 - 1e-4
    assert v_high <= dwa.max_v
    assert v_low >= dwa.min_v


def test_dwa_degraded_speed_cap():
    """Verify in degraded mode, max velocity is strictly capped at 0.35 m/s."""
    dwa = DWAPlanner()
    st = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.5, omega=0.0)

    v_low, v_high, w_low, w_high = dwa._compute_dynamic_window(st, dt=0.1, max_v_limit=0.35)
    assert v_high <= 0.35 + 1e-4


def test_dwa_follows_straight_path():
    """Verify DWA selects forward linear velocity to follow straight global path."""
    wmap = create_empty_world_map()
    dwa = DWAPlanner()

    st = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32))

    traj, v, w, status, diag = dwa.compute_trajectory(st, path, wmap)
    assert status == PlannerStatus.SUCCESS
    assert v > 0.0  # Moves forward
    assert abs(w) < 0.20  # Minimal turning required on straight line


def test_dwa_slow_safe_vs_fast_unsafe():
    """Section 57: Obstacle 1.0m ahead. High speed would collide, low speed or turn is safe."""
    wmap = create_empty_world_map()
    dwa = DWAPlanner()

    # Place obstacle 1.0m directly ahead
    for dy in np.arange(-0.4, 0.5, 0.1):
        r, c = wmap.world_to_grid(1.0, float(dy))
        wmap.obstacle[r, c] = True

    st = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0)
    # Global path points to the right around obstacle
    path = Path.from_xy_array(np.array([[0.0, 0.0], [0.5, -0.6], [1.5, -0.6]], dtype=np.float32))

    traj, v, w, status, diag = dwa.compute_trajectory(st, path, wmap)
    assert status == PlannerStatus.SUCCESS
    assert traj is not None
    # Must steer away from obstacle
    assert w != 0.0
    # Clearance along selected trajectory must be positive
    assert traj.min_clearance_m > 0.0


def test_dwa_no_safe_trajectory_stop():
    """Section 56: When boxed in by obstacles on all sides, DWA emits zero velocity."""
    wmap = create_empty_world_map()
    dwa = DWAPlanner()

    # Wall immediately surrounding robot at distance 0.45m
    for angle in np.linspace(0, 2 * math.pi, 36):
        ox = 0.45 * math.cos(angle)
        oy = 0.45 * math.sin(angle)
        r, c = wmap.world_to_grid(ox, oy)
        wmap.obstacle[r, c] = True

    st = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))

    traj, v, w, status, diag = dwa.compute_trajectory(st, path, wmap)
    assert status == PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY
    assert v == 0.0
    assert w == 0.0
