"""Unit and performance tests for WS-2A: Native C++ Radial Costmap Inflation & Vectorized Spatial Fusion."""

from __future__ import annotations

import time
import numpy as np
import pytest

from ai.costmap.base import CostmapMetadata
from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.spatial_fusion.fusion_engine import SpatialFusionEngine
from ai.spatial_fusion.world_map import WorldTerrainMap
from ai.traversability.base import TraversabilityResult
from production.perception.vision_costmap_integrator import VisionCostmapIntegrator


def _make_bev(shape=(58, 60)) -> BEVTerrainCostmap:
    h, w = shape
    meta = CostmapMetadata(resolution_m=0.10, shape=shape, x_range_m=(0.2, 6.0), y_range_m=(-3.0, 3.0), timestamp=1.0)
    bev = BEVTerrainCostmap(
        cost=np.full(shape, 0.15, dtype=np.float32),
        traversability=np.full(shape, 0.85, dtype=np.float32),
        obstacle=np.zeros(shape, dtype=bool),
        hazard=np.zeros(shape, dtype=np.float32),
        unknown=np.zeros(shape, dtype=bool),
        confidence=np.full(shape, 0.90, dtype=np.float32),
        slope_deg=np.zeros(shape, dtype=np.float32),
        roughness=np.zeros(shape, dtype=np.float32),
        height_var=np.zeros(shape, dtype=np.float32),
        observation_count=np.ones(shape, dtype=np.int32),
        metadata=meta,
    )
    bev.obstacle[20:25, 25:35] = True
    bev.cost[20:25, 25:35] = 1.0
    bev.hazard[20:25, 25:35] = 0.95
    return bev


def test_vectorized_spatial_fusion_latency_under_2ms():
    """Verify SpatialFusionEngine.fuse_local_costmap runs in < 2.0 ms on CPU (down from 28.2 ms)."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    engine = SpatialFusionEngine(world_map=wmap)
    bev = _make_bev((58, 60))
    pose = np.eye(4, dtype=np.float64)

    # Warmup
    _ = engine.fuse_local_costmap(bev, direct_pose=pose)

    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        res = engine.fuse_local_costmap(bev, direct_pose=pose)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        assert res.success

    mean_ms = float(np.mean(latencies))
    assert mean_ms < 2.0, f"SpatialFusionEngine.fuse_local_costmap took {mean_ms:.2f}ms (> 2.0ms budget)"


def test_radial_inflation_and_world_fusion_latency():
    """Verify VisionCostmapIntegrator radial obstacle inflation and world map fusion execute in < 1.5 ms."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))

    mask = np.full((240, 320), 0.90, dtype=np.float32)
    mask[120:160, 140:180] = 0.05  # Obstacle patch
    res = TraversabilityResult(mask=mask, timestamp=1.0, valid=True)

    bev = integrator.project_traversability(res, current_time_s=1.0)
    assert np.any(bev.obstacle)
    inflated = (bev.cost >= 0.75) & (~bev.obstacle) & (~bev.unknown)
    assert np.any(inflated)

    # Warmup fusion
    _ = integrator.fuse_into_world_map(bev, (0.0, 0.0, 0.0), wmap)

    latencies = []
    for i in range(20):
        bev.timestamp = 2.0 + i * 0.1
        stat = integrator.fuse_into_world_map(bev, (0.0, 0.0, 0.0), wmap)
        latencies.append(stat["fusion_ms"])

    mean_fuse_ms = float(np.mean(latencies))
    assert mean_fuse_ms < 1.5, f"VisionCostmapIntegrator.fuse_into_world_map took {mean_fuse_ms:.2f}ms (> 1.5ms budget)"
