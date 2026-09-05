"""Unit tests for 8-connected grid A* path planner."""

import pytest
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import GoalState, PlannerStatus, RobotState
from ai.planning.global_planners.astar import AStarPlanner


@pytest.fixture
def empty_costmap():
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0),
        resolution_m=0.10, default_cost=0.10
    )
    costmap.unknown.fill(False)
    return costmap


def test_astar_flat_ground_straight_line(empty_costmap):
    """On flat ground with no obstacles, A* returns a near-straight shortest path."""
    start = RobotState(x=1.0, y=0.0, yaw=0.0)
    goal = GoalState(x=4.0, y=0.0)

    planner = AStarPlanner()
    path, status, diag = planner.plan(empty_costmap, start, goal)

    assert status == PlannerStatus.SUCCESS
    assert path is not None
    assert abs(path.length - 3.0) < 0.15
    assert diag["nodes_explored"] > 0
    # Smoothed path has minimal waypoints
    assert len(path) <= 4


def test_astar_obstacle_avoidance(empty_costmap):
    """A* navigates around an obstacle placed directly on the centerline."""
    r_mid, c_mid = empty_costmap.meter_to_grid(2.5, 0.0)
    empty_costmap.obstacle[r_mid-3:r_mid+4, c_mid-3:c_mid+4] = True
    empty_costmap.cost[r_mid-3:r_mid+4, c_mid-3:c_mid+4] = 1.00

    start = RobotState(x=1.0, y=0.0, yaw=0.0)
    goal = GoalState(x=4.0, y=0.0)

    planner = AStarPlanner()
    path, status, diag = planner.plan(empty_costmap, start, goal)

    assert status == PlannerStatus.SUCCESS
    assert path is not None
    assert path.length > 3.0  # Must be longer due to detour

    # Verify no path point intersects obstacle
    for pt in path.points:
        r, c = empty_costmap.meter_to_grid(pt.x, pt.y)
        assert not empty_costmap.obstacle[r, c]


def test_astar_blocked_corridor_no_path(empty_costmap):
    """When a wall blocks the entire map, A* returns NO_PATH gracefully without crashing."""
    r_mid, _ = empty_costmap.meter_to_grid(2.5, 0.0)
    empty_costmap.obstacle[r_mid-2:r_mid+3, :] = True
    empty_costmap.cost[r_mid-2:r_mid+3, :] = 1.00

    start = RobotState(x=1.0, y=0.0, yaw=0.0)
    goal = GoalState(x=4.0, y=0.0)

    planner = AStarPlanner()
    path, status, diag = planner.plan(empty_costmap, start, goal)

    assert status == PlannerStatus.NO_PATH
    assert path is None
