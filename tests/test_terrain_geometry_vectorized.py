"""Unit tests for Vectorized 2.5D Terrain Geometry Estimator & Lethal Gates (WS-1C).

Verifies:
1. Flat planar terrain -> 0 deg slope, [0, 0, 1] normal, 0 step height, 0 roughness.
2. Synthetic 15 deg, 25 deg, and 32 deg inclined ramps -> surface normal and slope angle error < 0.5 deg.
3. Synthetic vertical step obstacles (0.05m safe vs 0.15m and 0.25m lethal high-centering steps).
4. Lethal costmap gating on WorldTerrainMap: >= 28 deg slopes and >= 0.12m steps force cost=1.0 and obstacle=True.
5. Global A* routing detour around a 32 deg steep embankment along a gentle 10 deg trail.
6. Execution latency < 3.0 ms for 200x200 grid on CPU.
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.global_planners.astar import AStarPlanner
from ai.spatial_fusion.world_map import WorldTerrainMap
from production.perception.terrain_geometry import (
    TerrainGeometryConfig,
    VectorizedTerrainGeometryEstimator,
)


def test_flat_terrain_geometry():
    """Verify flat Z=0 grid yields vertical normals [0, 0, 1] and zero slope/step/roughness."""
    estimator = VectorizedTerrainGeometryEstimator()
    Z = np.zeros((100, 100), dtype=np.float32)
    res = estimator.estimate_from_elevation(Z, resolution_m=0.10)

    np.testing.assert_allclose(res.slope_deg, 0.0, atol=1e-5)
    np.testing.assert_allclose(res.step_height_m, 0.0, atol=1e-5)
    np.testing.assert_allclose(res.roughness_m, 0.0, atol=1e-5)
    np.testing.assert_allclose(res.normals[:, :, 2], 1.0, atol=1e-5)
    assert not np.any(res.lethal_slope_mask)
    assert not np.any(res.lethal_step_mask)


@pytest.mark.parametrize("target_slope_deg", [10.0, 15.0, 25.0, 32.0])
def test_inclined_ramp_normal_and_slope_accuracy(target_slope_deg: float):
    """Verify synthetic inclined ramp Z(x) = x * tan(alpha) recovers exact slope angle (< 0.1 deg error)."""
    estimator = VectorizedTerrainGeometryEstimator()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-10.0, 10.0], y_bounds_m=[-10.0, 10.0])

    rows = np.arange(wmap.n_rows, dtype=np.float32)[:, None]
    x_grid = wmap.x_bounds_m[1] - (rows + 0.5) * wmap.resolution_m
    tan_alpha = math.tan(math.radians(target_slope_deg))
    Z = np.broadcast_to(x_grid * tan_alpha, wmap.shape).copy()

    geom = estimator.estimate_from_elevation(Z, resolution_m=wmap.resolution_m)

    # Interior slope must match target_slope_deg within 0.05 deg
    interior_slope = geom.slope_deg[2:-2, 2:-2]
    np.testing.assert_allclose(interior_slope, target_slope_deg, atol=0.05)

    # Smooth constant-slope ramp must NOT trigger false step-height discontinuity
    assert np.all(geom.step_height_m[2:-2, 2:-2] < 0.01)

    if target_slope_deg >= 28.0:
        assert np.all(geom.lethal_slope_mask[2:-2, 2:-2])
    else:
        assert not np.any(geom.lethal_slope_mask[2:-2, 2:-2])


@pytest.mark.parametrize("step_h,expected_lethal", [(0.05, False), (0.15, True), (0.25, True)])
def test_step_obstacle_detection_and_hard_gates(step_h: float, expected_lethal: bool):
    """Verify vertical step height discontinuities >= 0.12m trigger lethal obstacle gates."""
    estimator = VectorizedTerrainGeometryEstimator()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-5.0, 5.0], y_bounds_m=[-5.0, 5.0])
    wmap.unknown.fill(False)
    wmap.cost.fill(0.10)

    Z = np.zeros(wmap.shape, dtype=np.float32)
    # Create an elevated rock ledge in center of map
    Z[40:60, 40:60] = step_h

    geom = estimator.update_world_map_from_elevation(wmap, Z)
    assert bool(np.any(geom.lethal_step_mask)) is expected_lethal
    assert not np.any(geom.negative_step_mask)
    if expected_lethal:
        assert np.any(wmap.obstacle[39:61, 39:61])
        assert np.max(wmap.cost[39:61, 39:61]) == 1.0


@pytest.mark.parametrize("ditch_drop_m,expected_lethal", [(-0.05, False), (-0.15, True), (-0.22, True)])
def test_negative_ditch_detection_and_hazard_gating(ditch_drop_m: float, expected_lethal: bool):
    """Verify negative obstacles / ditches (delta_Z- <= -0.12m) trigger negative_step_mask and lethal gates."""
    estimator = VectorizedTerrainGeometryEstimator()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-5.0, 5.0], y_bounds_m=[-5.0, 5.0])
    wmap.unknown.fill(False)
    wmap.cost.fill(0.10)

    Z = np.zeros(wmap.shape, dtype=np.float32)
    # Create a sunken trench / negative ditch in center of map
    Z[45:55, 35:65] = ditch_drop_m

    geom = estimator.update_world_map_from_elevation(wmap, Z)
    assert geom.negative_step_mask is not None
    assert geom.ditch_depth_m is not None
    assert bool(np.any(geom.negative_step_mask)) is expected_lethal

    if expected_lethal:
        assert float(np.min(geom.ditch_depth_m)) <= ditch_drop_m + 1e-3
        assert np.any(wmap.obstacle[44:56, 34:66])
        assert float(np.max(wmap.hazard[44:56, 34:66])) == 1.0
        assert float(np.max(wmap.cost[44:56, 34:66])) == 1.0


def test_boundary_aware_gradients_with_partial_elevation_valid_mask():
    """Verify boundary-aware finite differences across elevation_valid produce zero boundary cliff spikes."""
    estimator = VectorizedTerrainGeometryEstimator()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-5.0, 5.0], y_bounds_m=[-5.0, 5.0])
    wmap.unknown.fill(False)
    wmap.cost.fill(0.10)

    # Create a 14 deg ramp elevated by +0.80m ONLY inside rows [25:75, 25:75]; outside is unobserved Z=0.0
    rows = np.arange(wmap.n_rows, dtype=np.float32)[:, None]
    x_grid = wmap.x_bounds_m[1] - (rows + 0.5) * wmap.resolution_m
    tan_14 = math.tan(math.radians(14.0))

    Z = np.zeros(wmap.shape, dtype=np.float32)
    valid_mask = np.zeros(wmap.shape, dtype=bool)
    valid_mask[25:75, 25:75] = True
    Z[25:75, 25:75] = 0.80 + np.broadcast_to(x_grid * tan_14, wmap.shape)[25:75, 25:75]

    geom = estimator.update_world_map_from_elevation(wmap, Z, elevation_valid=valid_mask)

    # All valid cells (including the 1-cell border of the valid mask!) must recover 14.0 deg slope
    np.testing.assert_allclose(geom.slope_deg[25:75, 25:75], 14.0, atol=0.08)
    # Unobserved cells must have 0 slope
    np.testing.assert_allclose(geom.slope_deg[~valid_mask], 0.0, atol=1e-5)

    # Zero false step, ditch, or lethal slope spikes anywhere on the map (including the 0.80m FOV jump!)
    assert not np.any(geom.lethal_slope_mask)
    assert not np.any(geom.lethal_step_mask)
    assert not np.any(geom.negative_step_mask)
    assert not np.any(wmap.obstacle)


def test_astar_detours_around_steep_embankment_for_gentle_trail():
    """Verify Global A* rejects a direct 32 deg embankment and routes along a gentle 8 deg detour."""
    estimator = VectorizedTerrainGeometryEstimator()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-2.0, 10.0], y_bounds_m=[-5.0, 5.0])
    wmap.unknown.fill(False)
    wmap.cost.fill(0.10)
    wmap.traversability.fill(0.95)  # Visually 95% traversable everywhere (e.g. green grass)

    Z = np.zeros(wmap.shape, dtype=np.float32)
    # Place a steep 32 deg embankment across the direct centerline (x in [3.0, 5.0], y in [-1.5, 1.5])
    for r in range(wmap.n_rows):
        for c in range(wmap.n_cols):
            xw, yw = wmap.grid_to_world(r, c)
            if 3.0 <= xw <= 5.0 and -1.5 <= yw <= 1.5:
                Z[r, c] = (xw - 3.0) * math.tan(math.radians(32.0))

    estimator.update_world_map_from_elevation(wmap, Z)

    planner = AStarPlanner()
    path, status, _ = planner.plan(
        wmap,
        RobotState(x=0.0, y=0.0, yaw=0.0),
        GoalPose(x=8.0, y=0.0),
    )
    assert status == PlannerStatus.SUCCESS
    assert path is not None

    # Verify path detours outside y in [-1.5, 1.5] when crossing x in [3.5, 4.5]
    mid_pts = [pt for pt in path.points if 3.5 <= pt.x <= 4.5]
    assert len(mid_pts) > 0
    for pt in mid_pts:
        assert abs(pt.y) > 1.5, f"Path crossed lethal 32 deg embankment at ({pt.x:.2f}, {pt.y:.2f})"


def test_terrain_geometry_200x200_latency():
    """Verify vectorized 200x200 terrain geometry estimation runs in < 3.0 ms on CPU."""
    estimator = VectorizedTerrainGeometryEstimator()
    rng = np.random.RandomState(42)
    Z = rng.uniform(0.0, 0.5, size=(200, 200)).astype(np.float32)

    # Warmup
    _ = estimator.estimate_from_elevation(Z, resolution_m=0.10)

    latencies = []
    for _ in range(20):
        res = estimator.estimate_from_elevation(Z, resolution_m=0.10)
        latencies.append(res.compute_ms)

    mean_ms = float(np.mean(latencies))
    assert mean_ms < 3.0, f"200x200 terrain geometry took {mean_ms:.2f}ms (> 3.0ms budget)"
