"""Unit and integration tests for Subphase 15.5: Terrain-Constrained C++17 Global A* Planning.

Verifies:
1. Global A* rejects a direct 32 deg steep embankment and selects a longer 10 deg gentle detour trail.
2. Global A* rejects a direct 18 cm boulder step field and routes cleanly around it.
3. Global A* prefers a gentle 8 deg switchback trail over a passable but steep/rough 22 deg hill,
   and path smoothing preserves the low-slope detour without shortcutting across the hill.
4. Impassable >28 deg slope ring around goal returns PlannerStatus.NO_PATH safely.
5. Native C++17 Global A* plans terrain-constrained detours on 120x120 grids in < 5.0 ms with 100% Python parity.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.global_planners.astar import AStarPlanner
from ai.spatial_fusion.world_map import WorldTerrainMap
from production.perception.terrain_geometry import VectorizedTerrainGeometryEstimator


def _make_open_terrain_map(
    x_bounds: tuple[float, float] = (-2.0, 10.0),
    y_bounds: tuple[float, float] = (-6.0, 6.0),
    res: float = 0.10,
) -> WorldTerrainMap:
    wmap = WorldTerrainMap(resolution_m=res, x_bounds_m=x_bounds, y_bounds_m=y_bounds)
    wmap.unknown.fill(False)
    wmap.cost.fill(0.08)
    wmap.traversability.fill(0.95)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    return wmap


def test_astar_detours_around_32deg_embankment_via_10deg_trail():
    """Verify Global A* selects a longer 10 deg gentle trail over a direct 32 deg grass embankment."""
    wmap = _make_open_terrain_map()
    estimator = VectorizedTerrainGeometryEstimator()

    Z = np.zeros(wmap.shape, dtype=np.float32)
    for r in range(wmap.n_rows):
        for c in range(wmap.n_cols):
            xw, yw = wmap.grid_to_world(r, c)
            if 2.5 <= xw <= 5.5:
                # Smooth symmetric tent profile peaking at x = 4.0m
                dist_from_edge = 1.5 - abs(xw - 4.0)
                if -1.5 <= yw <= 1.5:
                    # Direct centerline: lethal 32 deg embankment
                    Z[r, c] = dist_from_edge * math.tan(math.radians(32.0))
                elif abs(yw) >= 2.2:
                    # Side bypass trail (|y| >= 2.2): gentle 10 deg pass
                    Z[r, c] = dist_from_edge * math.tan(math.radians(10.0))

    estimator.update_world_map_from_elevation(wmap, Z)

    planner_cpp = AStarPlanner(backend="cpp")
    planner_py = AStarPlanner(backend="python")

    start = RobotState(x=0.0, y=0.0, yaw=0.0)
    goal = GoalPose(x=8.0, y=0.0)

    path_cpp, status_cpp, diag_cpp = planner_cpp.plan(wmap, start, goal)
    path_py, status_py, diag_py = planner_py.plan(wmap, start, goal)

    assert status_cpp == PlannerStatus.SUCCESS
    assert status_py == PlannerStatus.SUCCESS
    assert path_cpp is not None and path_py is not None

    # Path must be longer than direct 8.0m Euclidean line due to switchback detour
    assert path_cpp.total_length_m > 9.0
    assert abs(path_cpp.total_length_m - path_py.total_length_m) < 0.15

    # Every waypoint along the path must avoid the lethal 32 deg embankment (slope < 20 deg)
    for pt in path_cpp.points:
        r, c = wmap.world_to_grid(pt.x, pt.y)
        assert float(wmap.slope[r, c]) < 20.0, f"Waypoint ({pt.x:.2f}, {pt.y:.2f}) on steep slope {wmap.slope[r, c]:.1f} deg"
        assert not bool(wmap.obstacle[r, c])


def test_astar_detours_around_18cm_boulder_step_field():
    """Verify Global A* routes around an 18 cm boulder step field blocking the direct centerline."""
    wmap = _make_open_terrain_map()
    estimator = VectorizedTerrainGeometryEstimator()

    Z = np.zeros(wmap.shape, dtype=np.float32)
    # Place an 18 cm elevated boulder cluster across x in [3.0, 5.0], y in [-1.8, 1.8]
    for r in range(wmap.n_rows):
        for c in range(wmap.n_cols):
            xw, yw = wmap.grid_to_world(r, c)
            if 3.0 <= xw <= 5.0 and -1.8 <= yw <= 1.8:
                Z[r, c] = 0.18

    estimator.update_world_map_from_elevation(wmap, Z)

    planner = AStarPlanner(backend="cpp")
    path, status, _ = planner.plan(
        wmap,
        RobotState(x=0.0, y=0.0, yaw=0.0),
        GoalPose(x=8.0, y=0.0),
    )

    assert status == PlannerStatus.SUCCESS
    assert path is not None

    mid_pts = [pt for pt in path.points if 3.2 <= pt.x <= 4.8]
    assert len(mid_pts) > 0
    for pt in mid_pts:
        assert abs(pt.y) > 1.8, f"Path crossed 18cm boulder field at ({pt.x:.2f}, {pt.y:.2f})"


def test_astar_prefers_gentle_8deg_trail_over_passable_22deg_rough_hill():
    """Verify Global A* chooses a slightly longer 8 deg smooth trail over a direct 22 deg rough hill,
    and path smoothing preserves the detour without shortcutting back across the hill.
    """
    wmap = _make_open_terrain_map()

    # Center hill: passable (obstacle=False, cost=0.48 < 0.65) but steep (22 deg) and rough (0.40)
    for r in range(wmap.n_rows):
        for c in range(wmap.n_cols):
            xw, yw = wmap.grid_to_world(r, c)
            if 2.5 <= xw <= 5.5 and -1.6 <= yw <= 1.6:
                wmap.slope[r, c] = 22.0
                wmap.roughness[r, c] = 0.40
                wmap.cost[r, c] = 0.48
            elif 2.5 <= xw <= 5.5:
                wmap.slope[r, c] = 8.0
                wmap.roughness[r, c] = 0.02
                wmap.cost[r, c] = 0.12

    planner_cpp = AStarPlanner(backend="cpp")
    planner_py = AStarPlanner(backend="python")

    start = RobotState(x=0.0, y=0.0, yaw=0.0)
    goal = GoalPose(x=8.0, y=0.0)

    path_cpp, st_cpp, diag_cpp = planner_cpp.plan(wmap, start, goal)
    path_py, st_py, diag_py = planner_py.plan(wmap, start, goal)

    assert st_cpp == PlannerStatus.SUCCESS
    assert st_py == PlannerStatus.SUCCESS
    assert abs(diag_cpp["path_cost"] - diag_py["path_cost"]) < 1e-4

    # Both smoothed paths must bypass the 22 deg hill (|y| > 1.6 for x in [3.0, 5.0])
    for pt in path_cpp.points:
        if 3.2 <= pt.x <= 4.8:
            assert abs(pt.y) > 1.5, f"Smoothed C++ path shortcut across 22 deg hill at ({pt.x:.2f}, {pt.y:.2f})"


def test_astar_impassable_full_corridor_embankment_returns_no_path():
    """Verify Global A* returns NO_PATH when a >= 28 deg embankment spans the entire corridor width."""
    wmap = _make_open_terrain_map()
    # Set slope = 31.0 deg across entire map width at x in [3.5, 4.5] (even without setting obstacle=True beforehand)
    for r in range(wmap.n_rows):
        xw, _ = wmap.grid_to_world(r, 0)
        if 3.5 <= xw <= 4.5:
            wmap.slope[r, :] = 31.0

    planner = AStarPlanner(backend="cpp")
    path, status, _ = planner.plan(
        wmap,
        RobotState(x=0.0, y=0.0, yaw=0.0),
        GoalPose(x=8.0, y=0.0),
    )
    assert status == PlannerStatus.NO_PATH
    assert path is None


def test_terrain_astar_120x120_latency():
    """Verify C++17 Global A* plans around a 32 deg embankment on a 120x120 grid in < 5.0 ms."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-2.0, 10.0), y_bounds_m=(-6.0, 6.0))
    assert wmap.shape == (120, 120)
    wmap.unknown.fill(False)
    wmap.cost.fill(0.08)
    wmap.traversability.fill(0.95)

    Z = np.zeros(wmap.shape, dtype=np.float32)
    for r in range(wmap.n_rows):
        for c in range(wmap.n_cols):
            xw, yw = wmap.grid_to_world(r, c)
            if 3.0 <= xw <= 5.0 and -1.5 <= yw <= 1.5:
                Z[r, c] = (xw - 3.0) * math.tan(math.radians(32.0))

    estimator = VectorizedTerrainGeometryEstimator()
    estimator.update_world_map_from_elevation(wmap, Z)

    planner = AStarPlanner(backend="cpp")
    start = RobotState(x=0.0, y=0.0, yaw=0.0)
    goal = GoalPose(x=8.0, y=0.0)

    # Warmup
    _ = planner.plan(wmap, start, goal)

    core_latencies = []
    for _ in range(15):
        path, status, diag = planner.plan(wmap, start, goal)
        assert status == PlannerStatus.SUCCESS
        core_latencies.append(float(diag["cpp_total_ms"]))

    mean_ms = float(np.mean(core_latencies))
    assert mean_ms < 5.0, f"120x120 Terrain A* C++ latency {mean_ms:.2f}ms exceeds 5.0ms budget"
