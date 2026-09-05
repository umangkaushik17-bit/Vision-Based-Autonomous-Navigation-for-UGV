"""Unit tests for integrated global/local planning handoff and pure pursuit tracking."""

import pytest
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import GoalState, PlannerStatus, RobotState
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.local_planners.path_follower import PurePursuitTracker
from ai.planning.path import Path


@pytest.fixture
def empty_costmap():
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0),
        resolution_m=0.10, default_cost=0.10
    )
    costmap.unknown.fill(False)
    return costmap


def test_pure_pursuit_steering(empty_costmap):
    """Pure pursuit computes valid steering commands along curved path."""
    tracker = PurePursuitTracker()
    state = RobotState(x=1.0, y=0.0, yaw=0.0, v=0.50, omega=0.0)

    # Path curving to the left
    pts = np.array([[1.0, 0.0], [2.0, 0.5], [3.0, 1.2], [4.0, 1.5]], dtype=np.float32)
    path = Path.from_xy_array(pts)

    traj, v, w, status, diag = tracker.compute_trajectory(state, path, empty_costmap)

    assert status == PlannerStatus.SUCCESS
    assert v > 0.0
    assert w > 0.0  # Steers left (+omega)
    assert not diag["emergency_stop_triggered"]


def test_pure_pursuit_emergency_stop(empty_costmap):
    """Pure pursuit stops if lookahead path is blocked by a solid barrier."""
    tracker = PurePursuitTracker()
    state = RobotState(x=1.0, y=0.0, yaw=0.0, v=0.50, omega=0.0)
    path = Path.from_xy_array(np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32))

    # Place wall across lookahead path
    r_wall, _ = empty_costmap.meter_to_grid(1.3, 0.0)
    empty_costmap.obstacle[r_wall-2:r_wall+3, :] = True
    empty_costmap.cost[r_wall-2:r_wall+3, :] = 1.00

    _, v, w, status, diag = tracker.compute_trajectory(state, path, empty_costmap)

    assert status == PlannerStatus.NO_SAFE_TRAJECTORY
    assert v == 0.0
    assert w == 0.0
    assert diag["emergency_stop_triggered"]


def test_global_to_local_pipeline_handoff(empty_costmap):
    """Verify seamless handoff from Global Planner -> Global Path -> Local Planner -> Trajectory."""
    start = RobotState(x=1.0, y=0.0, yaw=0.0)
    goal = GoalState(x=4.5, y=0.5)

    # 1. Global Planning
    global_planner = AStarPlanner()
    path, g_status, g_diag = global_planner.plan(empty_costmap, start, goal)

    assert g_status == PlannerStatus.SUCCESS
    assert path is not None

    # 2. Local Planning using Global Path
    local_planner = DWAPlanner()
    traj, v, w, l_status, l_diag = local_planner.compute_trajectory(start, path, empty_costmap)

    assert l_status == PlannerStatus.SUCCESS
    assert traj is not None
    assert v > 0.0
    assert not l_diag["emergency_stop_triggered"]
