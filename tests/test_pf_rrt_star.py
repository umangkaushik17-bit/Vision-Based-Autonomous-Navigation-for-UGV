"""Unit tests for MoveBase3D-inspired PF-RRT* path planner."""

import pytest
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import GoalState, PlanningConfig, PlannerStatus, RobotState
from ai.planning.global_planners.pf_rrt_star import PFRRTStarPlanner


@pytest.fixture
def empty_costmap():
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0),
        resolution_m=0.10, default_cost=0.10
    )
    costmap.unknown.fill(False)
    return costmap


def test_pf_rrt_star_basic(empty_costmap):
    """PF-RRT* finds a valid path from start to goal on open ground."""
    start = RobotState(x=1.0, y=0.0, yaw=0.0)
    goal = GoalState(x=3.5, y=0.5, xy_tolerance_m=0.25)

    planner = PFRRTStarPlanner()
    path, status, diag = planner.plan(empty_costmap, start, goal)

    assert status == PlannerStatus.SUCCESS
    assert path is not None
    assert path.length > 2.0
    assert diag["tree_size"] > 1


def test_pf_rrt_star_seed_reproducibility(empty_costmap):
    """Fixed random seed yields reproducible planning results."""
    start = RobotState(x=1.0, y=0.0, yaw=0.0)
    goal = GoalState(x=3.5, y=0.5, xy_tolerance_m=0.25)

    cfg1 = PlanningConfig()
    cfg1.pf_rrt_star["random_seed"] = 123
    planner1 = PFRRTStarPlanner(config=cfg1)
    path1, _, _ = planner1.plan(empty_costmap, start, goal)

    cfg2 = PlanningConfig()
    cfg2.pf_rrt_star["random_seed"] = 123
    planner2 = PFRRTStarPlanner(config=cfg2)
    path2, _, _ = planner2.plan(empty_costmap, start, goal)

    assert path1 is not None and path2 is not None
    assert abs(path1.length - path2.length) < 1e-4


def test_pf_rrt_star_blocked_path(empty_costmap):
    """When a complete barrier blocks the goal, PF-RRT* returns NO_PATH."""
    r_mid, _ = empty_costmap.meter_to_grid(2.5, 0.0)
    empty_costmap.obstacle[r_mid-2:r_mid+3, :] = True
    empty_costmap.cost[r_mid-2:r_mid+3, :] = 1.00

    start = RobotState(x=1.0, y=0.0, yaw=0.0)
    goal = GoalState(x=4.0, y=0.0)

    cfg = PlanningConfig()
    cfg.pf_rrt_star["max_iterations"] = 150
    planner = PFRRTStarPlanner(config=cfg)
    path, status, _ = planner.plan(empty_costmap, start, goal)

    assert status == PlannerStatus.NO_PATH
    assert path is None
