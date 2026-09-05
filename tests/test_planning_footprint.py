"""Unit tests for UGV Footprint Collision Checker across WorldTerrainMap and BEV costmaps."""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.base import VehicleKinematics
from ai.planning.collision import FootprintCollisionChecker
from ai.spatial_fusion.world_map import WorldTerrainMap


def create_empty_world_map(resolution: float = 0.10, bounds: tuple[float, float] = (-5.0, 5.0)) -> WorldTerrainMap:
    """Helper creating an empty, observed safe WorldTerrainMap."""
    wmap = WorldTerrainMap(resolution_m=resolution, x_bounds_m=bounds, y_bounds_m=bounds)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    return wmap


def test_footprint_corner_geometry():
    """Verify oriented bounding box corners match vehicle length and width."""
    veh = VehicleKinematics(length_m=0.65, width_m=0.45)
    checker = FootprintCollisionChecker(veh)

    corners_0 = checker.get_footprint_corners(x_m=0.0, y_m=0.0, yaw_rad=0.0)
    assert corners_0.shape == (4, 2)
    # At yaw=0, front-left corner should be (+L/2, +W/2)
    np.testing.assert_allclose(corners_0[0], [0.325, 0.225], atol=1e-4)
    np.testing.assert_allclose(corners_0[1], [0.325, -0.225], atol=1e-4)
    np.testing.assert_allclose(corners_0[2], [-0.325, -0.225], atol=1e-4)
    np.testing.assert_allclose(corners_0[3], [-0.325, 0.225], atol=1e-4)

    # Rotated 90 degrees (+pi/2: East -> North)
    corners_90 = checker.get_footprint_corners(x_m=1.0, y_m=2.0, yaw_rad=math.pi / 2.0)
    np.testing.assert_allclose(corners_90[0], [1.0 - 0.225, 2.0 + 0.325], atol=1e-4)


def test_obstacle_footprint_collision():
    """Verify obstacle detection under rectangular footprint."""
    wmap = create_empty_world_map()
    veh = VehicleKinematics(length_m=0.65, width_m=0.45)
    checker = FootprintCollisionChecker(veh)

    # Place an obstacle at (0.25, 0.15), inside front-left quadrant
    r_obs, c_obs = wmap.world_to_grid(0.25, 0.15)
    wmap.obstacle[r_obs, c_obs] = True

    # Collision must be detected at (0, 0, 0)
    assert checker.is_footprint_collision(0.0, 0.0, 0.0, wmap)

    # Away at (2.0, 2.0, 0) should be clear
    assert not checker.is_footprint_collision(2.0, 2.0, 0.0, wmap)


def test_centerline_clear_footprint_collides():
    """Section 53 Mandatory Test: Path centerline is clear, but UGV footprint intersects obstacle."""
    wmap = create_empty_world_map()
    veh = VehicleKinematics(length_m=0.65, width_m=0.45)
    checker = FootprintCollisionChecker(veh)

    # Place obstacle at (0.0, 0.20): 0.20m to the left of the robot centerline
    # Centerline is at y=0.0 (distance 0.20m < half-width 0.225m)
    r_obs, c_obs = wmap.world_to_grid(0.0, 0.20)
    wmap.obstacle[r_obs, c_obs] = True

    # Centerline point check at (0, 0) is FALSE (no obstacle directly on centerline)
    assert not checker.is_point_collision(0.0, 0.0, wmap)

    # But footprint check at (0, 0, yaw=0) MUST be TRUE (collision detected!)
    assert checker.is_footprint_collision(0.0, 0.0, 0.0, wmap)


def test_critical_hazard_collision():
    """Verify cells with critical hazard (>= 0.70) trigger collision."""
    wmap = create_empty_world_map()
    veh = VehicleKinematics(length_m=0.65, width_m=0.45)
    checker = FootprintCollisionChecker(veh)

    # Place hazard at (1.0, 0.0)
    r, c = wmap.world_to_grid(1.0, 0.0)
    wmap.hazard[r, c] = 0.85  # Critical hazard

    assert checker.is_point_collision(1.0, 0.0, wmap)
    assert checker.is_footprint_collision(1.0, 0.0, 0.0, wmap)


def test_unknown_space_policy():
    """Verify UNKNOWN != SAFE: unknown cells are rejected when unknown_is_lethal=True."""
    wmap = create_empty_world_map()
    veh = VehicleKinematics(length_m=0.65, width_m=0.45)
    checker = FootprintCollisionChecker(veh)

    r, c = wmap.world_to_grid(0.0, 0.0)
    wmap.unknown[r, c] = True

    assert checker.is_point_collision(0.0, 0.0, wmap, unknown_is_lethal=True)
    assert not checker.is_point_collision(0.0, 0.0, wmap, unknown_is_lethal=False)


def test_narrow_passage_footprint():
    """Narrow passage: gap of 0.60m allows 0.45m UGV, but gap of 0.40m collides."""
    veh = VehicleKinematics(length_m=0.65, width_m=0.45)
    checker = FootprintCollisionChecker(veh)

    # 1. Wide passage: obstacles at y = +0.35 and y = -0.35 (gap = 0.70m > width 0.45m)
    wmap_wide = create_empty_world_map()
    r1, c1 = wmap_wide.world_to_grid(0.0, 0.35)
    r2, c2 = wmap_wide.world_to_grid(0.0, -0.35)
    wmap_wide.obstacle[r1, c1] = True
    wmap_wide.obstacle[r2, c2] = True
    assert not checker.is_footprint_collision(0.0, 0.0, 0.0, wmap_wide)

    # 2. Narrow passage: obstacles at y = +0.18 and y = -0.18 (gap = 0.36m < width 0.45m)
    wmap_narrow = create_empty_world_map()
    r1, c1 = wmap_narrow.world_to_grid(0.0, 0.18)
    r2, c2 = wmap_narrow.world_to_grid(0.0, -0.18)
    wmap_narrow.obstacle[r1, c1] = True
    wmap_narrow.obstacle[r2, c2] = True
    assert checker.is_footprint_collision(0.0, 0.0, 0.0, wmap_narrow)


def test_clearance_calculation():
    """Verify metric distance to nearest obstacle."""
    wmap = create_empty_world_map()
    checker = FootprintCollisionChecker()

    # Place obstacle 1.0 meter ahead at (1.0, 0.0)
    r, c = wmap.world_to_grid(1.0, 0.0)
    wmap.obstacle[r, c] = True

    clr = checker.get_clearance(0.0, 0.0, wmap, max_search_radius_m=2.0)
    assert 0.90 <= clr <= 1.10
