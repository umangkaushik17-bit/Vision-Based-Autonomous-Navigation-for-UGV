"""Unit tests for MotionCompensator SE(2) map alignment."""

import pytest
import numpy as np

from ai.costmap.base import CostmapMetadata
from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.costmap.motion_compensation import MotionCompensator


@pytest.fixture
def test_setup():
    metadata = CostmapMetadata(
        resolution_m=0.10,
        shape=(30, 30),
        x_range_m=(0.0, 3.0),
        y_range_m=(-1.5, 1.5)
    )
    compensator = MotionCompensator(metadata)
    return metadata, compensator


def test_identity_warp(test_setup):
    """Zero pose delta leaves the grid identical."""
    metadata, compensator = test_setup
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.0, 3.0),
        y_range_m=(-1.5, 1.5),
        resolution_m=0.10
    )
    costmap.cost[15, 15] = 0.99

    warped = compensator.warp_costmap(costmap, 0.0, 0.0, 0.0)
    assert warped.cost[15, 15] == 0.99
    assert np.allclose(warped.cost, costmap.cost)


def test_forward_motion_shift(test_setup):
    """Vehicle moving forward shifts static obstacles backward in robot frame."""
    metadata, compensator = test_setup
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.0, 3.0),
        y_range_m=(-1.5, 1.5),
        resolution_m=0.10
    )
    costmap.unknown.fill(False)
    costmap.cost.fill(0.10)

    # Place obstacle 2.0m ahead (row = (3.0 - 2.0)/0.1 = 10, col = 15)
    r_orig, c_orig = costmap.meter_to_grid(2.0, 0.0)
    costmap.cost[r_orig, c_orig] = 1.00
    costmap.obstacle[r_orig, c_orig] = True

    # Robot moves forward 0.5m (dx = +0.5m)
    # The obstacle is now at x = 1.5m relative to robot
    warped = compensator.warp_costmap(costmap, dx_m=0.5, dy_m=0.0, dyaw_rad=0.0)

    r_expected, c_expected = costmap.meter_to_grid(1.5, 0.0)
    # In row indices: row increases as x decreases (x_max is at top row 0)
    assert r_expected > r_orig
    assert warped.cost[r_expected, c_expected] > 0.80


def test_unobserved_boundary_padding(test_setup):
    """Territory outside the prior map bounds is strictly marked UNKNOWN."""
    metadata, compensator = test_setup
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.0, 3.0),
        y_range_m=(-1.5, 1.5),
        resolution_m=0.10
    )
    costmap.unknown.fill(False)

    # Move forward 1.0m
    warped = compensator.warp_costmap(costmap, dx_m=1.0, dy_m=0.0, dyaw_rad=0.0)

    # Top rows (far ahead, newly entering sensor range) must be unknown
    assert bool(warped.unknown[0, 15]) is True
