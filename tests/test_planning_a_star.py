"""Unit tests for Phase 12 A* Global Planner across WorldTerrainMap."""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.base import GoalPose, PlanningConfig, PlannerStatus, RobotState
from ai.planning.global_planners.astar import AStarPlanner
from ai.spatial_fusion.world_map import WorldTerrainMap


def create_empty_world_map(resolution: float = 0.10, bounds: tuple[float, float] = (-5.0, 5.0)) -> WorldTerrainMap:
    wmap = WorldTerrainMap(resolution_m=resolution, x_bounds_m=bounds, y_bounds_m=bounds)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    return wmap


def test_astar_open_terrain():
    """Scenario A: Open flat terrain, straight collision-free route found."""
    wmap = create_empty_world_map()
    planner = AStarPlanner()

    start = RobotState(x=-2.0, y=0.0, yaw=0.0)
    goal = GoalPose(x=2.0, y=0.0)

    path, status, diag = planner._search(wmap, start, goal)
    assert status == PlannerStatus.SUCCESS
    assert path is not None
    assert len(path) >= 2
    # Straight path should be close to 4.0 meters
    assert abs(path.total_length_m - 4.0) < 0.25


def test_astar_obstacle_detour():
    """Scenario B: Rock obstacle blocks direct path, A* detours around it."""
    wmap = create_empty_world_map()
    planner = AStarPlanner()

    # Block direct route at x=0.0, y in [-0.5, 0.5]
    for y_obs in np.arange(-0.5, 0.6, 0.1):
        r, c = wmap.world_to_grid(0.0, float(y_obs))
        wmap.obstacle[r, c] = True

    start = RobotState(x=-2.0, y=0.0, yaw=0.0)
    goal = GoalPose(x=2.0, y=0.0)

    path, status, diag = planner._search(wmap, start, goal)
    assert status == PlannerStatus.SUCCESS
    assert path is not None
    # Path length must be strictly greater than direct 4.0m line
    assert path.total_length_m > 4.0
    # Clearance check: no waypoint in obstacle
    for pt in path.points:
        r, c = wmap.world_to_grid(pt.x, pt.y)
        assert not wmap.obstacle[r, c]


def test_astar_hazard_avoidance():
    """Scenario C: Critical hazard (hazard >= 0.70) is avoided by A*."""
    wmap = create_empty_world_map()
    planner = AStarPlanner()

    # Place hazard patch across direct path
    for y_haz in np.arange(-0.6, 0.7, 0.1):
        r, c = wmap.world_to_grid(0.0, float(y_haz))
        wmap.hazard[r, c] = 0.90  # Critical hazard

    start = RobotState(x=-2.0, y=0.0, yaw=0.0)
    goal = GoalPose(x=2.0, y=0.0)

    path, status, diag = planner._search(wmap, start, goal)
    assert status == PlannerStatus.SUCCESS
    assert path is not None
    for pt in path.points:
        r, c = wmap.world_to_grid(pt.x, pt.y)
        assert wmap.hazard[r, c] < 0.70


def test_astar_no_path():
    """Scenario E: Completely blocked goal returns NO_PATH without crashing."""
    wmap = create_empty_world_map()
    planner = AStarPlanner()

    # Solid perimeter wall around goal at (2.0, 0.0)
    r_goal, c_goal = wmap.world_to_grid(2.0, 0.0)
    for dr in range(-6, 7):
        for dc in range(-6, 7):
            if abs(dr) in (5, 6) or abs(dc) in (5, 6):
                wmap.obstacle[r_goal + dr, c_goal + dc] = True

    start = RobotState(x=-2.0, y=0.0, yaw=0.0)
    goal = GoalPose(x=2.0, y=0.0)


    path, status, diag = planner._search(wmap, start, goal)
    assert status in (PlannerStatus.NO_PATH, PlannerStatus.GOAL_BLOCKED)
    assert path is None


def test_astar_invalid_bounds():
    """Goal outside map bounds returns INVALID_GOAL."""
    wmap = create_empty_world_map(bounds=(-5.0, 5.0))
    planner = AStarPlanner()

    start = RobotState(x=0.0, y=0.0, yaw=0.0)
    goal = GoalPose(x=10.0, y=0.0)  # Beyond bounds

    path, status, diag = planner._search(wmap, start, goal)
    assert status == PlannerStatus.INVALID_GOAL
    assert path is None


def test_astar_determinism():
    """Multiple planning runs on identical inputs yield identical paths and costs."""
    wmap = create_empty_world_map()
    # Place obstacle
    r, c = wmap.world_to_grid(0.0, 0.0)
    wmap.obstacle[r, c] = True

    planner = AStarPlanner()
    start = RobotState(x=-1.5, y=-0.5, yaw=0.0)
    goal = GoalPose(x=1.5, y=0.5)

    path1, status1, diag1 = planner._search(wmap, start, goal)
    path2, status2, diag2 = planner._search(wmap, start, goal)

    assert status1 == status2 == PlannerStatus.SUCCESS
    assert path1 is not None and path2 is not None
    assert math.isclose(path1.total_length_m, path2.total_length_m, rel_tol=1e-5)
    assert math.isclose(diag1["path_cost"], diag2["path_cost"], rel_tol=1e-5)


def test_octile_heuristic_admissibility():
    """Verify Octile heuristic never overestimates 8-connected grid movement."""
    planner = AStarPlanner()
    res = 0.10
    sqrt2 = math.sqrt(2.0)

    for dr in range(1, 10):
        for dc in range(1, 10):
            h = planner._heuristic(0, 0, dr, dc, res)
            # True 8-connected minimum distance
            true_dist = (max(dr, dc) - min(dr, dc)) * res + min(dr, dc) * sqrt2 * res
            assert h <= true_dist + 1e-6
