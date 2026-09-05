"""Unit tests for footprint geometry and collision detection."""

import math
import pytest
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import VehicleKinematics
from ai.planning.collision import FootprintCollisionChecker


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


def test_footprint_corners_geometry():
    """Verify corner coordinates of oriented rectangular footprint."""
    vehicle = VehicleKinematics(length_m=1.0, width_m=0.6)
    checker = FootprintCollisionChecker(vehicle)

    # Robot at (0, 0) facing +X (yaw = 0)
    corners_0 = checker.get_footprint_corners(0.0, 0.0, 0.0)
    assert corners_0.shape == (4, 2)
    # Front-left is (0.5, 0.3)
    assert abs(corners_0[0, 0] - 0.5) < 1e-4
    assert abs(corners_0[0, 1] - 0.3) < 1e-4

    # Robot rotated 90 degrees (+Y)
    corners_90 = checker.get_footprint_corners(0.0, 0.0, math.pi / 2.0)
    # Front-left is (-0.3, 0.5)
    assert abs(corners_90[0, 0] - (-0.3)) < 1e-4
    assert abs(corners_90[0, 1] - 0.5) < 1e-4


def test_inscribed_circle_collision(test_costmap):
    """Inscribed circle shortcut accurately detects collision."""
    checker = FootprintCollisionChecker()

    # Robot center 0.2m away from obstacle (less than inscribed radius 0.35m)
    assert checker.is_circle_collision(2.2, 0.0, 0.35, test_costmap)

    # Robot center 1.0m away from obstacle
    assert not checker.is_circle_collision(3.5, 0.0, 0.35, test_costmap)


def test_clearance_metric(test_costmap):
    """Distance to closest obstacle matches Euclidean geometry."""
    checker = FootprintCollisionChecker()

    # Obstacle is at (2.0, 0.0)
    # Test point at (3.0, 0.0) -> distance is approx 1.0m
    clr = checker.get_clearance(3.0, 0.0, test_costmap)
    assert abs(clr - 1.0) < 0.15
