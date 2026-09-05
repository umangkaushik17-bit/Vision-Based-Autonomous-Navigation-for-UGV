"""Unit tests verifying safety principles, false-safe preservation, and the planner consumer contract."""

import pytest
import numpy as np

from ai.costmap.base import CostmapMetadata, UnknownPolicyMode
from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.costmap.fusion import BEVCostmapFuser
from ai.costmap.layers import LayerGenerator
from ai.perception_fusion import SafetyAwarePerceptionResult


def test_bev_false_safe_preservation():
    """Verify that camera-space hazard detections remain strictly lethal in BEV space."""
    h, w = 480, 640
    # Create synthetic frame with water hazard patch
    rgb = np.full((h, w, 3), 110, dtype=np.uint8)
    trav = np.full((h, w), 0.90, dtype=np.float32)  # Optimistic vision
    hazard = np.zeros((h, w), dtype=bool)
    hazard[300:360, 200:300] = True  # Water reflection patch detected
    trav[hazard] = 0.0  # Phase 4 override

    perception_res = SafetyAwarePerceptionResult(
        final_traversability=trav,
        hazard_mask=hazard,
        hazard_classes=["WATER"],
        risk_map=hazard.astype(np.float32),
        unknown_mask=np.zeros((h, w), dtype=bool),
        confidence_map=np.full((h, w), 0.95, dtype=np.float32),
        detections=[],
        overlay=rgb,
        latency_ms=5.0
    )

    layer_gen = LayerGenerator()
    layers = layer_gen.generate_layers(perception_res, rgb_image=rgb)
    metadata = CostmapMetadata(
        resolution_m=layer_gen.resolution_m,
        shape=(layer_gen.n_x, layer_gen.n_y),
        x_range_m=layer_gen.x_range_m,
        y_range_m=layer_gen.y_range_m
    )

    fuser = BEVCostmapFuser()
    costmap = fuser.fuse_layers(layers, metadata)

    # In the hazard zone, cost MUST be 1.00 and obstacle MUST be True
    hazard_cells = (costmap.hazard >= 0.85)
    assert hazard_cells.any()
    assert (costmap.cost[hazard_cells] == 1.00).all()
    assert (costmap.obstacle[hazard_cells]).all()


def test_unknown_policy_modes():
    """Verify CONSERVATIVE, MODERATE, and EXPLORATORY unknown costs."""
    metadata = CostmapMetadata(resolution_m=0.10, shape=(5, 5), x_range_m=(0, 1), y_range_m=(-0.5, 0.5))
    mock_layers = {
        "traversability": np.full((5, 5), 0.50, dtype=np.float32),
        "hazard": np.zeros((5, 5), dtype=np.float32),
        "obstacle": np.zeros((5, 5), dtype=bool),
        "unknown": np.ones((5, 5), dtype=bool),  # Entire map is unknown
        "confidence": np.zeros((5, 5), dtype=np.float32),
        "slope_deg": np.zeros((5, 5), dtype=np.float32),
        "roughness": np.zeros((5, 5), dtype=np.float32),
        "height_var": np.zeros((5, 5), dtype=np.float32),
        "observation_count": np.zeros((5, 5), dtype=np.int32)
    }

    fuser_cons = BEVCostmapFuser(unknown_mode=UnknownPolicyMode.CONSERVATIVE)
    map_cons = fuser_cons.fuse_layers(mock_layers, metadata)
    assert map_cons.cost[0, 0] == 0.65

    fuser_mod = BEVCostmapFuser(unknown_mode=UnknownPolicyMode.MODERATE)
    map_mod = fuser_mod.fuse_layers(mock_layers, metadata)
    assert map_mod.cost[0, 0] == 0.50

    fuser_exp = BEVCostmapFuser(unknown_mode=UnknownPolicyMode.EXPLORATORY)
    map_exp = fuser_exp.fuse_layers(mock_layers, metadata)
    assert map_exp.cost[0, 0] == 0.35


def test_planner_input_contract_stub():
    """Phase 6 Planner Consumer Stub Test (Section 53 of Phase 5 Directive).

    Proves that downstream path planning algorithms can consume the BEVTerrainCostmap
    without needing any knowledge of SAM-TP, RUGD, or perception internals.
    """
    # 1. Consumer requests local costmap
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.0),
        y_range_m=(-2.5, 2.5),
        resolution_m=0.10,
        default_cost=0.10
    )
    costmap.unknown.fill(False)

    # Place a simulated obstacle at x=3.0m, y=0.0m
    r_obs, c_obs = costmap.meter_to_grid(3.0, 0.0)
    costmap.obstacle[r_obs, c_obs] = True
    costmap.cost[r_obs, c_obs] = 1.00

    # Place an unknown region at x=4.5m, y=1.0m
    r_unk, c_unk = costmap.meter_to_grid(4.5, 1.0)
    costmap.unknown[r_unk, c_unk] = True
    costmap.cost[r_unk, c_unk] = 0.65

    # 2. Verify metadata
    assert costmap.metadata.frame_id == "base_link"
    assert costmap.resolution_m == 0.10
    assert costmap.shape == (48, 50)

    # 3. Inspect obstacle cells
    assert not costmap.is_passable(3.0, 0.0)
    assert costmap.get_cost(3.0, 0.0) == 1.00

    # 4. Inspect unknown cells
    assert costmap.is_unknown(4.5, 1.0)
    assert abs(costmap.get_cost(4.5, 1.0) - 0.65) < 1e-4

    # 5. Inspect safe cells
    assert costmap.is_passable(1.5, 0.0)
    assert abs(costmap.get_cost(1.5, 0.0) - 0.10) < 1e-4

    # 6. Verify coordinate mapping and out-of-bounds safety
    assert costmap.get_cost(10.0, 0.0) == 1.00  # Out-of-bounds returns safe lethal default
    assert not costmap.is_passable(10.0, 0.0)


def test_failure_modes_and_graceful_degradation():
    """Verify safe handling of missing depth, NaNs, and extreme inputs."""
    h, w = 480, 640
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    trav = np.full((h, w), 0.50, dtype=np.float32)

    # Inject NaNs into depth
    nan_depth = np.full((h, w), np.nan, dtype=np.float32)

    layer_gen = LayerGenerator()
    # Should handle NaNs gracefully without crashing
    layers = layer_gen.generate_layers(trav, rgb_image=rgb, depth_override_m=nan_depth)

    # All cells must be flagged as UNKNOWN due to missing depth
    assert layers["unknown"].all()
    assert (layers["observation_count"] == 0).all()
