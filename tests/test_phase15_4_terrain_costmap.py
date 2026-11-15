"""Unit and integration tests for Subphase 15.4: Multi-Factor 2.5D Costmap Fusion & Belly Clearance Guard.

Verifies:
1. Undercarriage high-centering crown check (`check_belly_clearance_grid`) detects +14cm belly-high-centering
   mounds/boulders even when single-cell steps (< 0.10m) and apex slope (0 deg) are below step/slope limits.
2. Safe 6cm mounds and smooth inclined ramps (10 deg, 20 deg) do NOT trigger false belly clearance violations.
3. Multi-factor cost model (`TraversabilityCostModel`) and `VectorizedTerrainGeometryEstimator` mark:
   - 30 deg steep slopes,
   - +15 cm positive step ledges,
   - -15 cm negative ditches, and
   - +14 cm belly-high-centering mounds
   all as lethal (`cost = 1.0`, `obstacle = True`, `hazard = 1.0`) even when visual traversability is 0.98 (green grass).
4. `CostModelWeights.from_yaml("configs/terrain_safety.yaml")` loads vehicle ground clearance and safety limits.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.perception_fusion.cost_model import CostModelWeights, TraversabilityCostModel
from ai.spatial_fusion.world_map import WorldTerrainMap
from production.perception.terrain_geometry import (
    VectorizedTerrainGeometryEstimator,
    check_belly_clearance_grid,
)


def test_belly_clearance_guard_detects_14cm_high_centering_mound():
    """Verify a +14cm rounded crown/mound triggers lethal belly clearance violation (h_crown >= 0.12m)."""
    Z = np.zeros((80, 80), dtype=np.float32)

    # Create a rounded mound at (40, 40) where adjacent 10cm steps are <= 0.07m (below 0.12m step limit)
    # and apex slope is 0 deg, yet total crown height above the 5x5/7x7 wheel support ring is +0.14m.
    Z[38:43, 38:43] = 0.05
    Z[39:42, 39:42] = 0.09
    Z[40, 40] = 0.14

    crown_h, lethal_belly = check_belly_clearance_grid(
        elevation_m=Z,
        resolution_m=0.10,
        ground_clearance_m=0.12,
    )

    assert float(crown_h[40, 40]) >= 0.12, f"Expected crown height >= 0.12m, got {crown_h[40, 40]:.3f}m"
    assert bool(lethal_belly[40, 40]) is True


def test_belly_clearance_guard_ignores_safe_mound_and_smooth_ramps():
    """Verify a benign +6cm bump and a smooth 18 deg ramp do NOT trigger false belly clearance violations."""
    Z_bump = np.zeros((60, 60), dtype=np.float32)
    Z_bump[29:32, 29:32] = 0.06
    _, lethal_bump = check_belly_clearance_grid(Z_bump, resolution_m=0.10, ground_clearance_m=0.12)
    assert not np.any(lethal_bump)

    # Smooth 18 deg inclined ramp
    rows = np.arange(80, dtype=np.float32)[:, None]
    x_grid = 4.0 - (rows + 0.5) * 0.10
    Z_ramp = np.broadcast_to(x_grid * math.tan(math.radians(18.0)), (80, 80)).copy()
    _, lethal_ramp = check_belly_clearance_grid(Z_ramp, resolution_m=0.10, ground_clearance_m=0.12)
    assert not np.any(lethal_ramp[3:-3, 3:-3])


def test_all_four_2_5d_hazards_marked_lethal_despite_high_visual_traversability():
    """Verify 30 deg slopes, +15cm steps, -15cm ditches, and +14cm belly mounds are all marked C=1.0
    even when visual semantic traversability is 0.98 everywhere.
    """
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    wmap.unknown.fill(False)
    wmap.traversability.fill(0.98)  # 98% visually traversable green grass
    wmap.cost.fill(0.05)

    Z = np.zeros(wmap.shape, dtype=np.float32)

    # Hazard 1: 30 deg steep slope embankment in quadrant 1 (rows 20:45, cols 20:45)
    for r in range(20, 45):
        Z[r, 20:45] = (45 - r) * 0.10 * math.tan(math.radians(30.0))

    # Hazard 2: +15 cm positive step plateau in quadrant 2 (rows 20:40, cols 130:155)
    Z[20:40, 130:155] = 0.15

    # Hazard 3: -15 cm negative ditch trench in quadrant 3 (rows 130:145, cols 25:60)
    Z[130:145, 25:60] = -0.15

    # Hazard 4: +14 cm rounded belly-high-centering mound in quadrant 4 at (150, 150)
    Z[148:153, 148:153] = 0.05
    Z[149:152, 149:152] = 0.09
    Z[150, 150] = 0.14

    estimator = VectorizedTerrainGeometryEstimator()
    geom = estimator.update_world_map_from_elevation(wmap, Z)

    cost_model = TraversabilityCostModel(CostModelWeights.from_yaml("configs/terrain_safety.yaml"))
    cost_model.fuse_world_map_cost(
        wmap,
        ditch_depth_grid=geom.ditch_depth_m,
        crown_height_grid=geom.crown_height_m,
    )

    # 1. Verify 30 deg slope interior is lethal (C = 1.0, obstacle = True)
    assert float(wmap.cost[32, 32]) == 1.0
    assert bool(wmap.obstacle[32, 32]) is True

    # 2. Verify +15 cm positive step perimeter is lethal (C = 1.0, obstacle = True)
    assert float(np.max(wmap.cost[19:41, 129:156])) == 1.0
    assert bool(np.any(wmap.obstacle[19:41, 129:156])) is True

    # 3. Verify -15 cm negative ditch is lethal (C = 1.0, obstacle = True)
    assert float(np.max(wmap.cost[129:146, 24:61])) == 1.0
    assert bool(np.any(wmap.obstacle[129:146, 24:61])) is True

    # 4. Verify +14 cm belly-high-centering mound apex (150, 150) is lethal (C = 1.0, obstacle = True)
    assert float(wmap.cost[150, 150]) == 1.0
    assert bool(wmap.obstacle[150, 150]) is True

    # 5. Verify flat safe ground at (100, 100) remains low cost (< 0.10) and obstacle-free
    assert float(wmap.cost[100, 100]) < 0.10
    assert bool(wmap.obstacle[100, 100]) is False


def test_cost_model_scalar_and_yaml_gates():
    """Verify CostModelWeights YAML loading and scalar compute_cell_cost gates."""
    weights = CostModelWeights.from_yaml("configs/terrain_safety.yaml")
    assert abs(weights.ground_clearance_m - 0.12) < 1e-6
    assert abs(weights.critical_slope_deg - 28.0) < 1e-6
    assert abs(weights.max_step_height_m - 0.12) < 1e-6

    model = TraversabilityCostModel(weights)
    assert model.compute_cell_cost(trav_score=0.99, slope_deg=30.0) == 1.0
    assert model.compute_cell_cost(trav_score=0.99, height_var_m=0.15) == 1.0
    assert model.compute_cell_cost(trav_score=0.99, ditch_depth_m=-0.15) == 1.0
    assert model.compute_cell_cost(trav_score=0.99, crown_height_m=0.14) == 1.0
    assert model.compute_cell_cost(trav_score=0.95, slope_deg=5.0) < 0.20
