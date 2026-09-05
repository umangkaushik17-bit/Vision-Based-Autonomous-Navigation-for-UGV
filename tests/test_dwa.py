"""Unit tests for Dynamic Window Approach (DWA) local planner."""

import pytest
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import PlannerStatus, RobotState
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path


@pytest.fixture
def empty_costmap():
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0),
        resolution_m=0.10, default_cost=0.10
    )
    costmap.unknown.fill(False)
    return costmap


def test_dwa_nominal_velocity_command(empty_costmap):
    """In an open corridor, DWA commands positive linear velocity toward the goal."""
    dwa = DWAPlanner()
    state = RobotState(x=1.0, y=0.0, yaw=0.0, v=0.20, omega=0.0)
    path = Path.from_xy_array(np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]], dtype=np.float32))

    traj, v, w, status, diag = dwa.compute_trajectory(state, path, empty_costmap)

    assert status == PlannerStatus.SUCCESS
    assert v > 0.0
    assert traj is not None
    assert not diag["emergency_stop_triggered"]


def test_dwa_emergency_stop_when_blocked(empty_costmap):
    """When a solid wall blocks all forward trajectories, DWA halts with (v=0, w=0)."""
    dwa = DWAPlanner()
    state = RobotState(x=1.0, y=0.0, yaw=0.0, v=0.40, omega=0.0)
    path = Path.from_xy_array(np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32))

    # Place wall 0.4m ahead of the robot
    r_wall, _ = empty_costmap.meter_to_grid(1.4, 0.0)
    empty_costmap.obstacle[r_wall-2:r_wall+3, :] = True
    empty_costmap.cost[r_wall-2:r_wall+3, :] = 1.00

    traj, v, w, status, diag = dwa.compute_trajectory(state, path, empty_costmap)

    assert status == PlannerStatus.NO_SAFE_TRAJECTORY
    assert v == 0.0
    assert w == 0.0
    assert diag["emergency_stop_triggered"]
