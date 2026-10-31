"""Tests for Phase 14.3 Costmap Integration & Safety Invariants.

Verifies traversability-to-cost mapping, obstacle inflation, UNKNOWN != SAFE preservation,
world coordinate transformation, and obstacle persistence across dropped frames.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.spatial_fusion.world_map import WorldTerrainMap
from ai.traversability.base import TraversabilityResult
from production.perception.vision_costmap_integrator import (
    LocalBEVCostmap,
    VisionCostmapConfig,
    VisionCostmapIntegrator,
)


def _make_dummy_result(
    mask_val: float,
    shape: tuple[int, int] = (240, 320),
    timestamp: float = 1.0,
    valid: bool = True,
) -> TraversabilityResult:
    """Helper to generate uniform TraversabilityResult."""
    mask = np.full(shape, mask_val, dtype=np.float32)
    return TraversabilityResult(
        mask=mask,
        confidence=np.ones(shape, dtype=np.float32),
        timestamp=timestamp,
        frame_id="camera_optical",
        valid=valid,
    )


def test_drivability_probability_mapping():
    """Verify continuous probability mapping to navigation costs."""
    integrator = VisionCostmapIntegrator.create_simulation_default()

    # 1. High traversability (safe path: 0.90)
    res_safe = _make_dummy_result(0.90)
    bev_safe = integrator.project_traversability(res_safe, current_time_s=1.0)
    assert bev_safe.valid
    known_safe = ~bev_safe.unknown
    assert np.any(known_safe)
    assert np.all(bev_safe.cost[known_safe] <= 0.20)
    assert np.all(~bev_safe.obstacle[known_safe])

    # 2. Risky / off-trail terrain (0.40)
    res_risky = _make_dummy_result(0.40)
    bev_risky = integrator.project_traversability(res_risky, current_time_s=1.0)
    known_risky = ~bev_risky.unknown
    assert np.all((bev_risky.cost[known_risky] >= 0.40) & (bev_risky.cost[known_risky] <= 0.70))
    assert np.all(~bev_risky.obstacle[known_risky])

    # 3. Lethal obstacle (0.10)
    res_obs = _make_dummy_result(0.10)
    bev_obs = integrator.project_traversability(res_obs, current_time_s=1.0)
    known_obs = ~bev_obs.unknown
    assert np.all(bev_obs.obstacle[known_obs])
    assert np.allclose(bev_obs.cost[known_obs], 1.0)


def test_unknown_not_safe_invariant():
    """Verify unobserved / void cells are strictly preserved as unknown with neutral/conservative cost."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    res = _make_dummy_result(0.85)
    bev = integrator.project_traversability(res, current_time_s=1.0)

    # Some cells (e.g. out of camera FOV or beyond ground range) must be unknown
    assert np.any(bev.unknown)
    # UNKNOWN cells must NEVER have low cost (free space)
    assert np.all(bev.cost[bev.unknown] >= 0.50)
    assert np.all(~bev.obstacle[bev.unknown])


def test_obstacle_inflation_buffer():
    """Verify lethal obstacles receive safety inflation buffer into adjacent cells."""
    integrator = VisionCostmapIntegrator.create_simulation_default()

    # Create mask with single obstacle patch in center ground region
    mask = np.full((240, 320), 0.90, dtype=np.float32)  # nominal safe
    mask[150:180, 140:180] = 0.05  # lethal obstacle patch

    res = TraversabilityResult(
        mask=mask,
        confidence=np.ones((240, 320), dtype=np.float32),
        timestamp=1.0,
        frame_id="camera_optical",
        valid=True,
    )
    bev = integrator.project_traversability(res, current_time_s=1.0)

    # Must contain both lethal obstacle cells and inflated buffer cells
    assert np.any(bev.obstacle)
    # Inflated non-obstacle cells should have cost >= inflation_cost (0.75)
    inflated_mask = (bev.cost >= 0.75) & (~bev.obstacle) & (~bev.unknown)
    assert np.any(inflated_mask)


def test_world_map_fusion_heading_transforms():
    """Verify fusion into WorldTerrainMap correctly rotates relative coordinates using robot yaw."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-5.0, 10.0], y_bounds_m=[-5.0, 10.0])

    # Obstacle patch centered at optical center (row 120 in 240x320 image corresponds to cy=240 in 480x640)
    # Forward distance is ~2.1m, lateral 0.0m
    mask = np.full((240, 320), 0.90, dtype=np.float32)
    mask[115:125, 155:165] = 0.05  # Center pixel region -> lethal obstacle

    res = TraversabilityResult(
        mask=mask,
        confidence=np.ones((240, 320), dtype=np.float32),
        timestamp=1.0,
        valid=True,
    )
    bev = integrator.project_traversability(res, current_time_s=1.0)

    # Robot at (0, 0) facing North (yaw = +pi/2, where +Y is forward)
    pose = (0.0, 0.0, math.pi / 2.0)
    fusion_res = integrator.fuse_into_world_map(bev, pose, wmap)
    assert fusion_res["valid"]
    assert fusion_res["cells_updated"] > 0

    # In odom frame, forward is +Y when yaw = pi/2
    # Verify obstacle is placed at x ~ 0.0, y ~ 2.1
    r_obs, c_obs = wmap.world_to_grid(0.0, 2.1)
    # Nearby cells should be marked as obstacle or high cost
    window = wmap.cost[r_obs-2:r_obs+3, c_obs-2:c_obs+3]
    assert np.any(window >= 0.75)



def test_persistent_hazard_latching_across_dropped_frames():
    """Verify that an observed obstacle persists and is not wiped out by a single clear frame."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-5.0, 10.0], y_bounds_m=[-5.0, 10.0])

    # 1. Observe obstacle in frame 1
    obs_res = _make_dummy_result(0.10, timestamp=1.0)
    bev_obs = integrator.project_traversability(obs_res, current_time_s=1.0)
    integrator.fuse_into_world_map(bev_obs, (0.0, 0.0, 0.0), wmap)

    obs_cells_before = np.sum(wmap.obstacle)
    assert obs_cells_before > 0

    # 2. Dropped / invalid frame: must NOT clear world map
    invalid_res = _make_dummy_result(0.90, timestamp=1.1, valid=False)
    bev_invalid = integrator.project_traversability(invalid_res, current_time_s=1.1)
    fuse_inv = integrator.fuse_into_world_map(bev_invalid, (0.0, 0.0, 0.0), wmap)
    assert fuse_inv["status"] == "DEGRADED"
    assert np.sum(wmap.obstacle) == obs_cells_before

    # 3. Single clear frame: must NOT clear obstacle immediately (persistent latching)
    clear_res = _make_dummy_result(0.90, timestamp=1.2)
    bev_clear = integrator.project_traversability(clear_res, current_time_s=1.2)
    integrator.fuse_into_world_map(bev_clear, (0.0, 0.0, 0.0), wmap)
    # Obstacles must still be latched
    assert np.sum(wmap.obstacle) == obs_cells_before


def test_perception_timeout_degrades_safely():
    """Verify stale perception frames (> 0.5s) are rejected and mark status as timeout."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    # Frame timestamp 1.0s, current time 2.0s (age = 1.0s > 0.5s max age)
    res = _make_dummy_result(0.90, timestamp=1.0)
    bev = integrator.project_traversability(res, current_time_s=2.0)

    assert not bev.valid
    assert "PERCEPTION_TIMEOUT" in bev.status_message
    assert np.all(bev.unknown)  # Safely returns all-unknown conservative costmap
