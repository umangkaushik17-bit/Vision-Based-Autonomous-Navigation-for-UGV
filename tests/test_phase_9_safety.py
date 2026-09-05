"""Unit tests for Phase 9 safety rules, tracking state policies, and failure injection."""

import numpy as np
import pytest

from ai.costmap.base import CostmapMetadata
from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.localization.base import TrackingState
from ai.localization.result import LocalizationEstimate
from ai.spatial_fusion.fusion_engine import SpatialFusionEngine
from ai.spatial_fusion.synchronization import TemporalSynchronizer
from ai.spatial_fusion.transforms import create_se3_matrix, euler_to_rotation_matrix
from ai.spatial_fusion.world_map import WorldTerrainMap


def create_sample_bev(timestamp: float = 10.0) -> BEVTerrainCostmap:
    shape = (58, 60)
    H, W = shape
    cost = np.full((H, W), 0.20, dtype=np.float32)
    trav = np.full((H, W), 0.80, dtype=np.float32)
    obst = np.zeros((H, W), dtype=bool)
    hazard = np.zeros((H, W), dtype=np.float32)
    unknown = np.zeros((H, W), dtype=bool)
    conf = np.full((H, W), 0.90, dtype=np.float32)
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


def test_tracking_state_policies():
    """Verify that TRACKING_OK, DEGRADED, LOST, RELOCALIZING, and RECOVERED enforce their exact contracts."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    engine = SpatialFusionEngine(world_map=wmap, degraded_confidence_scale=0.50)
    T = np.eye(4)
    bev = create_sample_bev(timestamp=1.0)

    # 1. TRACKING_OK: Normal fusion
    res_ok = engine.fuse_local_costmap(bev, direct_pose=T, direct_state=TrackingState.TRACKING_OK, direct_confidence=0.90)
    assert res_ok.success
    assert res_ok.tracking_state == TrackingState.TRACKING_OK

    # 2. DEGRADED: Allowed but with scaled down confidence
    wmap_deg = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    engine_deg = SpatialFusionEngine(world_map=wmap_deg, degraded_confidence_scale=0.50)
    res_deg = engine_deg.fuse_local_costmap(bev, direct_pose=T, direct_state=TrackingState.DEGRADED, direct_confidence=0.50)
    assert res_deg.success
    assert res_deg.tracking_state == TrackingState.DEGRADED
    # Max confidence in wmap_deg should be 0.90 * 0.50 * 0.50 = 0.225
    observed_cells = ~wmap_deg.unknown
    assert np.max(wmap_deg.confidence[observed_cells]) <= 0.25

    # 3. LOST: Strictly blocked, zero stale poses emitted
    res_lost = engine.fuse_local_costmap(bev, direct_pose=T, direct_state=TrackingState.LOST, direct_confidence=0.10)
    assert not res_lost.success
    assert "blocked" in res_lost.reason

    # 4. RELOCALIZING: Strictly blocked
    res_reloc = engine.fuse_local_costmap(bev, direct_pose=T, direct_state=TrackingState.RELOCALIZING, direct_confidence=0.15)
    assert not res_reloc.success
    assert "blocked" in res_reloc.reason

    # 5. RECOVERED: Restores normal spatial fusion
    res_rec = engine.fuse_local_costmap(bev, direct_pose=T, direct_state=TrackingState.RECOVERED, direct_confidence=0.85)
    assert res_rec.success
    assert res_rec.tracking_state == TrackingState.RECOVERED


def test_stale_pose_protection_blocks_fusion():
    """Verify that an observation with no fresh localization within max_sync_delta_s is blocked."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    sync = TemporalSynchronizer(max_sync_delta_s=0.10)
    engine = SpatialFusionEngine(world_map=wmap, synchronizer=sync)

    # Add pose at t = 10.0
    T = np.eye(4)
    est = LocalizationEstimate(
        timestamp=10.0,
        pose=T,
        tracking_state=TrackingState.TRACKING_OK,
        confidence=0.90,
        is_pose_valid=True
    )
    engine.add_localization(est)

    # Observation at t = 10.25 (delta = 0.25s > 0.10s)
    bev_stale = create_sample_bev(timestamp=10.25)
    res = engine.fuse_local_costmap(bev_stale)
    assert not res.success
    assert "timestamp_gap_exceeded" in res.reason


def test_extreme_yaw_handling():
    """Verify transforms and grid mapping remain valid under extreme yaw angles (+pi, -pi, 2pi)."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    engine = SpatialFusionEngine(world_map=wmap)
    bev = create_sample_bev(timestamp=1.0)

    for yaw in [np.pi, -np.pi, 2 * np.pi, -np.pi / 2.0]:
        R = euler_to_rotation_matrix(0.0, 0.0, yaw)
        T = create_se3_matrix(R=R, t=[0.0, 0.0, 0.0])
        res = engine.fuse_local_costmap(bev, direct_pose=T)
        assert res.success
        assert res.fused_cells_count > 0


def test_out_of_bounds_rejection():
    """Verify that robot poses far from the world map origin filter out-of-bounds cells safely."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0))
    engine = SpatialFusionEngine(world_map=wmap)
    bev = create_sample_bev(timestamp=1.0)

    # Robot at world x = 50.0m (far outside the 5m world bounds)
    T = create_se3_matrix(t=[50.0, 50.0, 0.0])
    res = engine.fuse_local_costmap(bev, direct_pose=T)
    assert res.success
    # All 3480 local cells should fall outside world map bounds and be safely rejected without crashing
    assert res.fused_cells_count == 0
    assert res.rejected_cells_count == bev.shape[0] * bev.shape[1]
