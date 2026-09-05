"""Unit tests for SpatialFusionEngine: determinism, safety precedence, and repeated observation consistency."""

import numpy as np
import pytest

from ai.costmap.base import CostmapMetadata
from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.localization.base import TrackingState
from ai.spatial_fusion.fusion_engine import SpatialFusionEngine
from ai.spatial_fusion.transforms import create_se3_matrix, euler_to_rotation_matrix
from ai.spatial_fusion.world_map import WorldTerrainMap


def create_test_bev(
    timestamp: float = 10.0,
    shape: tuple = (58, 60),
    fill_cost: float = 0.20,
    fill_trav: float = 0.80,
    fill_conf: float = 0.90
) -> BEVTerrainCostmap:
    """Create a synthetic BEVTerrainCostmap conforming to Phase 5 specs."""
    H, W = shape
    cost = np.full((H, W), fill_cost, dtype=np.float32)
    trav = np.full((H, W), fill_trav, dtype=np.float32)
    obst = np.zeros((H, W), dtype=bool)
    hazard = np.zeros((H, W), dtype=np.float32)
    unknown = np.zeros((H, W), dtype=bool)
    conf = np.full((H, W), fill_conf, dtype=np.float32)
    slope = np.zeros((H, W), dtype=np.float32)
    rough = np.zeros((H, W), dtype=np.float32)
    hvar = np.zeros((H, W), dtype=np.float32)
    obsc = np.ones((H, W), dtype=np.int32)

    meta = CostmapMetadata(
        resolution_m=0.10,
        shape=shape,
        x_range_m=(0.2, 6.0),
        y_range_m=(-3.0, 3.0),
        timestamp=timestamp
    )
    return BEVTerrainCostmap(
        cost=cost,
        traversability=trav,
        obstacle=obst,
        hazard=hazard,
        unknown=unknown,
        confidence=conf,
        slope_deg=slope,
        roughness=rough,
        height_var=hvar,
        observation_count=obsc,
        metadata=meta
    )


def test_deterministic_fusion():
    """Verify fusing identical inputs produces identical world map states."""
    wmap1 = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    wmap2 = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))

    engine1 = SpatialFusionEngine(world_map=wmap1)
    engine2 = SpatialFusionEngine(world_map=wmap2)

    bev = create_test_bev(timestamp=1.0)
    T_wb = create_se3_matrix(t=[1.0, 2.0, 0.0])

    res1 = engine1.fuse_local_costmap(bev, direct_pose=T_wb)
    res2 = engine2.fuse_local_costmap(bev, direct_pose=T_wb)

    assert res1.success and res2.success
    assert np.allclose(wmap1.cost, wmap2.cost)
    assert np.allclose(wmap1.confidence, wmap2.confidence)
    assert np.array_equal(wmap1.obstacle, wmap2.obstacle)


def test_safety_precedence_no_dilution():
    """Verify that a high hazard or obstacle is NEVER diluted into safe traversability."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    engine = SpatialFusionEngine(world_map=wmap)
    T_wb = np.eye(4)

    # 1. First observation: Detects critical hazard (hazard=0.95, cost=1.0) at local cell (20, 30)
    bev_hazard = create_test_bev(timestamp=1.0, fill_cost=0.10, fill_trav=0.90)
    bev_hazard.hazard[20, 30] = 0.95
    bev_hazard.cost[20, 30] = 1.0
    bev_hazard.obstacle[20, 30] = True

    # Map cell coords
    xw_target, yw_target = bev_hazard.grid_to_meter(20, 30)
    rw_target, cw_target = wmap.world_to_grid(xw_target, yw_target)

    res1 = engine.fuse_local_costmap(bev_hazard, direct_pose=T_wb)
    assert res1.success
    assert wmap.hazard[rw_target, cw_target] == 0.95
    assert wmap.obstacle[rw_target, cw_target] is True or wmap.obstacle[rw_target, cw_target] == 1
    assert wmap.cost[rw_target, cw_target] == 1.0

    # 2. Second observation: Sees the exact same area as safe (hazard=0.0, cost=0.10, trav=0.95)
    bev_safe = create_test_bev(timestamp=2.0, fill_cost=0.10, fill_trav=0.95)
    res2 = engine.fuse_local_costmap(bev_safe, direct_pose=T_wb)
    assert res2.success

    # Critical Safety Precedence: World cell MUST remain critical hazard and obstacle!
    assert wmap.hazard[rw_target, cw_target] == 0.95, "Hazard was diluted by subsequent safe observation!"
    assert wmap.obstacle[rw_target, cw_target] is True or wmap.obstacle[rw_target, cw_target] == 1
    assert wmap.cost[rw_target, cw_target] == 1.0


def test_order_independent_safety():
    """Verify fusing Safe then Hazard yields the exact same safety outcome as Hazard then Safe."""
    # Test A: Hazard then Safe
    wmap_A = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    engine_A = SpatialFusionEngine(world_map=wmap_A)

    bev1 = create_test_bev(timestamp=1.0)
    bev1.hazard[25, 25] = 0.85
    bev1.cost[25, 25] = 1.0

    bev2 = create_test_bev(timestamp=2.0, fill_cost=0.10)

    engine_A.fuse_local_costmap(bev1, direct_pose=np.eye(4))
    engine_A.fuse_local_costmap(bev2, direct_pose=np.eye(4))

    # Test B: Safe then Hazard
    wmap_B = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    engine_B = SpatialFusionEngine(world_map=wmap_B)

    engine_B.fuse_local_costmap(bev2, direct_pose=np.eye(4))
    engine_B.fuse_local_costmap(bev1, direct_pose=np.eye(4))

    xw, yw = bev1.grid_to_meter(25, 25)
    rw, cw = wmap_A.world_to_grid(xw, yw)

    assert wmap_A.hazard[rw, cw] == wmap_B.hazard[rw, cw] == 0.85
    assert wmap_A.cost[rw, cw] == wmap_B.cost[rw, cw] == 1.0


def test_repeated_observation_spatial_consistency():
    """Verify that an obstacle observed from two different robot poses lands in the exact same world cell."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    engine = SpatialFusionEngine(world_map=wmap)

    # Physical obstacle at world coordinates (x=3.0m, y=1.0m)
    xw_true = 3.0
    yw_true = 1.0
    rw_expected, cw_expected = wmap.world_to_grid(xw_true, yw_true)

    # Pose 1: Robot at world (0, 0, 0), yaw=0.
    # Obstacle is at body (3.0, 1.0).
    T1 = create_se3_matrix(t=[0.0, 0.0, 0.0])
    bev1 = create_test_bev(timestamp=1.0)
    r1, c1 = bev1.meter_to_grid(3.0, 1.0)
    bev1.obstacle[r1, c1] = True
    bev1.cost[r1, c1] = 1.0

    res1 = engine.fuse_local_costmap(bev1, direct_pose=T1)
    assert res1.success
    assert wmap.obstacle[rw_expected, cw_expected] is True or wmap.obstacle[rw_expected, cw_expected] == 1

    # Pose 2: Robot has moved to world (1.0, 1.0, 0), yaw=0.
    # Relative to robot 2, obstacle is now at body (2.0, 0.0).
    T2 = create_se3_matrix(t=[1.0, 1.0, 0.0])
    bev2 = create_test_bev(timestamp=2.0)
    r2, c2 = bev2.meter_to_grid(2.0, 0.0)
    bev2.obstacle[r2, c2] = True
    bev2.cost[r2, c2] = 1.0

    res2 = engine.fuse_local_costmap(bev2, direct_pose=T2)
    assert res2.success

    # Verify obstacle is in the expected cell and observation count incremented
    assert wmap.obstacle[rw_expected, cw_expected] is True or wmap.obstacle[rw_expected, cw_expected] == 1
    assert wmap.observation_count[rw_expected, cw_expected] >= 2


def test_unknown_space_preservation():
    """Verify that local unknown cells do not erase known world obstacles or turn cells to free space."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    engine = SpatialFusionEngine(world_map=wmap)
    T_wb = np.eye(4)

    # 1. Observe obstacle
    bev_known = create_test_bev(timestamp=1.0)
    r_l, c_l = bev_known.meter_to_grid(2.5, 0.5)
    bev_known.obstacle[r_l, c_l] = True
    bev_known.cost[r_l, c_l] = 1.0

    xw, yw = bev_known.grid_to_meter(r_l, c_l)
    rw, cw = wmap.world_to_grid(xw, yw)

    engine.fuse_local_costmap(bev_known, direct_pose=T_wb)
    assert wmap.obstacle[rw, cw] is True or wmap.obstacle[rw, cw] == 1

    # 2. Subsequent observation has that cell marked unknown (e.g. shadow or occlusion)
    bev_unknown = create_test_bev(timestamp=2.0)
    bev_unknown.unknown[r_l, c_l] = True
    bev_unknown.cost[r_l, c_l] = 0.0  # Corrupt / missing data

    engine.fuse_local_costmap(bev_unknown, direct_pose=T_wb)

    # Obstacle MUST be preserved!
    assert wmap.obstacle[rw, cw] is True or wmap.obstacle[rw, cw] == 1
    assert wmap.cost[rw, cw] == 1.0
    assert not wmap.unknown[rw, cw]
