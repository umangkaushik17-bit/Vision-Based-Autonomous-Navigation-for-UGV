"""Unit tests for path and start/goal validation."""

import pytest
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import GoalState, PlannerStatus, RobotState
from ai.planning.path import Path
from ai.planning.validation import PathValidator


@pytest.fixture
def test_costmap():
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0),
        resolution_m=0.10, default_cost=0.10
    )
    costmap.unknown.fill(False)
    # Place obstacle at (2.0, 0.0)
    r, c = costmap.meter_to_grid(2.0, 0.0)
    costmap.obstacle[r, c] = True
    costmap.cost[r, c] = 1.00
    return costmap


def test_start_goal_validation_checks(test_costmap):
    """PathValidator correctly identifies invalid start and goal states."""
    validator = PathValidator()

    # Valid start and goal
    s_valid = RobotState(x=1.0, y=0.0)
    g_valid = GoalState(x=4.0, y=0.0)
    ok, status, _ = validator.validate_start_goal(s_valid, g_valid, test_costmap)
    assert ok and status == PlannerStatus.SUCCESS

    # Start out of bounds
    s_oob = RobotState(x=10.0, y=0.0)
    ok, status, _ = validator.validate_start_goal(s_oob, g_valid, test_costmap)
    assert not ok and status == PlannerStatus.INVALID_START

    # Start on obstacle
    s_obs = RobotState(x=2.0, y=0.0)
    ok, status, _ = validator.validate_start_goal(s_obs, g_valid, test_costmap)
    assert not ok and status == PlannerStatus.INVALID_START

    # Goal out of bounds
    g_oob = GoalState(x=1.0, y=-6.0)
    ok, status, _ = validator.validate_start_goal(s_valid, g_oob, test_costmap)
    assert not ok and status == PlannerStatus.INVALID_GOAL

    # Goal on obstacle
    g_obs = GoalState(x=2.0, y=0.0)
    ok, status, _ = validator.validate_start_goal(s_valid, g_obs, test_costmap)
    assert not ok and status == PlannerStatus.INVALID_GOAL


def test_path_crossing_lethal_hazard_rejection(test_costmap):
    """Validator rejects paths that cross lethal obstacles or rollover slopes."""
    validator = PathValidator(critical_slope_deg=28.0)

    # Path crossing the obstacle at (2.0, 0.0)
    bad_pts = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    bad_path = Path.from_xy_array(bad_pts)

    ok, reason, _ = validator.validate_path(bad_path, test_costmap)
    assert not ok
    assert "intersects hard obstacle" in reason or "unsafe cell" in reason
