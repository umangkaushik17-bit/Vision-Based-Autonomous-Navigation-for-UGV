"""Unit tests for BEVTerrainCostmap, layer generation, cost fusion, and obstacle inflation."""

import pytest
import numpy as np
from pathlib import Path

from ai.costmap.base import CostmapLayerType, CostmapMetadata, UnknownPolicyMode, VehicleFootprint
from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.costmap.fusion import BEVCostmapFuser
from ai.costmap.inflation import ObstacleInflator
from ai.costmap.layers import LayerGenerator
from ai.costmap.serialization import CostmapSerializer
from ai.perception_fusion import SafetyAwarePerceptionResult


def test_bev_costmap_creation():
    """Verify BEVTerrainCostmap properties, shapes, and layer retrieval."""
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 6.0),
        y_range_m=(-3.0, 3.0),
        resolution_m=0.10
    )

    assert costmap.shape == (58, 60)
    assert costmap.resolution_m == 0.10
    assert costmap.x_range_m == (0.2, 6.0)
    assert costmap.y_range_m == (-3.0, 3.0)
    assert costmap.unknown.all()
    assert (costmap.cost == 0.50).all()

    # Check all layers retrievable
    for l_type in [
        CostmapLayerType.FINAL_COST, CostmapLayerType.TRAVERSABILITY,
        CostmapLayerType.HAZARD, CostmapLayerType.SLOPE,
        CostmapLayerType.ROUGHNESS, CostmapLayerType.HEIGHT,
        CostmapLayerType.UNKNOWN, CostmapLayerType.CONFIDENCE,
        CostmapLayerType.TEMPORAL_PERSISTENCE
    ]:
        layer = costmap.get_layer(l_type)
        assert layer.shape == (58, 60)


def test_physical_extent_resolution_consistency():
    """Verify exact mathematical relationship: grid_dimension = physical_extent / resolution."""
    x_min, x_max = 0.2, 6.0
    y_min, y_max = -3.0, 3.0
    res = 0.10

    expected_forward_extent_m = x_max - x_min  # 5.8 meters
    expected_lateral_extent_m = y_max - y_min  # 6.0 meters

    assert abs(expected_forward_extent_m - 5.8) < 1e-6
    assert abs(expected_lateral_extent_m - 6.0) < 1e-6

    expected_rows = int(np.ceil(expected_forward_extent_m / res))  # 58 rows
    expected_cols = int(np.ceil(expected_lateral_extent_m / res))  # 60 cols

    assert expected_rows == 58
    assert expected_cols == 60

    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(x_min, x_max),
        y_range_m=(y_min, y_max),
        resolution_m=res
    )

    assert costmap.shape == (expected_rows, expected_cols)
    assert abs(costmap.shape[0] * res - expected_forward_extent_m) < 1e-6
    assert abs(costmap.shape[1] * res - expected_lateral_extent_m) < 1e-6


def test_coordinate_mapping_and_conventions():
    """Verify coordinate mapping: X forward, Y left, round-trip consistency."""
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.0, 10.0),
        y_range_m=(-5.0, 5.0),
        resolution_m=0.10
    )

    # Robot forward point: (x=5.0m, y=0.0m)
    r, c = costmap.meter_to_grid(5.0, 0.0)
    x_rec, y_rec = costmap.grid_to_meter(r, c)
    assert abs(x_rec - 5.0) < 0.10
    assert abs(y_rec - 0.0) < 0.10

    # Robot left point: (x=2.0m, y=3.0m)
    r_l, c_l = costmap.meter_to_grid(2.0, 3.0)
    # Robot right point: (x=2.0m, y=-3.0m)
    r_r, c_r = costmap.meter_to_grid(2.0, -3.0)

    # In ISO standard, Y=left means positive Y has lower column index than negative Y
    assert c_l < c_r
    assert r_l == r_r  # Same forward distance means same row index

    # Boundary checks
    assert costmap.is_in_bounds(5.0, 2.0)
    assert not costmap.is_in_bounds(12.0, 0.0)
    assert not costmap.is_in_bounds(5.0, 7.0)


def test_costmap_fuser_weights_and_overrides():
    """Verify cost weighting formula and non-negotiable hard safety overrides."""
    fuser = BEVCostmapFuser()
    metadata = CostmapMetadata(
        resolution_m=0.10,
        shape=(10, 10),
        x_range_m=(0.0, 1.0),
        y_range_m=(-0.5, 0.5)
    )

    # Base benign layers
    mock_layers = {
        "traversability": np.full((10, 10), 0.90, dtype=np.float32),
        "hazard": np.zeros((10, 10), dtype=np.float32),
        "obstacle": np.zeros((10, 10), dtype=bool),
        "unknown": np.zeros((10, 10), dtype=bool),
        "confidence": np.ones((10, 10), dtype=np.float32),
        "slope_deg": np.zeros((10, 10), dtype=np.float32),
        "roughness": np.zeros((10, 10), dtype=np.float32),
        "height_var": np.zeros((10, 10), dtype=np.float32),
        "observation_count": np.ones((10, 10), dtype=np.int32)
    }

    # 1. Safe cell test
    costmap = fuser.fuse_layers(mock_layers, metadata)
    assert costmap.cost[0, 0] < 0.20
    assert costmap.is_passable(0.5, 0.0)

    # 2. Hard Obstacle / Critical Hazard Override (Cell 2, 2)
    mock_layers["hazard"][2, 2] = 1.0
    mock_layers["obstacle"][2, 2] = True
    costmap = fuser.fuse_layers(mock_layers, metadata)
    assert costmap.cost[2, 2] == 1.00  # Hard lethal override
    assert costmap.obstacle[2, 2]

    # 3. Critical Slope Override (Cell 3, 3: 30 degrees >= 28 degrees)
    mock_layers["slope_deg"][3, 3] = 30.0
    costmap = fuser.fuse_layers(mock_layers, metadata)
    assert costmap.cost[3, 3] == 1.00  # Rollover lethal override
    assert costmap.obstacle[3, 3]

    # 4. Critical Step Height Override (Cell 4, 4: 0.25m >= 0.20m)
    mock_layers["height_var"][4, 4] = 0.25
    costmap = fuser.fuse_layers(mock_layers, metadata)
    assert costmap.cost[4, 4] == 1.00  # High-centering lethal override


def test_obstacle_inflator():
    """Verify Euclidean distance transform inflation creates a gradient around obstacles."""
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.0, 2.0),
        y_range_m=(-1.0, 1.0),
        resolution_m=0.10,
        default_cost=0.10
    )
    costmap.unknown.fill(False)

    # Place a single obstacle at row 10, col 10
    costmap.obstacle[10, 10] = True
    costmap.cost[10, 10] = 1.00

    inflator = ObstacleInflator(resolution_m=0.10)
    inflated = inflator.inflate_costmap(costmap)

    # Direct obstacle cell remains lethal
    assert inflated.cost[10, 10] == 1.00
    assert inflated.obstacle[10, 10]

    # Adjacent cell within inscribed radius (1 cell away = 0.10m <= 0.35m) becomes lethal
    assert inflated.cost[10, 11] == 1.00
    assert inflated.obstacle[10, 11]

    # Cell at 4 cells away (0.40m, between 0.35m and 0.55m) has decayed cost
    assert 0.10 < inflated.cost[10, 14] < 1.00

    # Cell far away (9 cells away = 0.9m > 0.55m) retains base cost
    assert inflated.cost[10, 19] == 0.10


def test_costmap_serialization(tmp_path):
    """Verify NPZ and JSON serialization round-trip fidelity."""
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.0),
        y_range_m=(-2.5, 2.5),
        resolution_m=0.10
    )
    costmap.cost[5, 5] = 0.88
    costmap.obstacle[5, 5] = True
    costmap.slope_deg[5, 5] = 15.2

    save_prefix = tmp_path / "test_costmap"
    npz_p, json_p = CostmapSerializer.save(costmap, save_prefix)

    assert npz_p.exists()
    assert json_p.exists()

    loaded = CostmapSerializer.load(save_prefix)
    assert loaded.shape == costmap.shape
    assert loaded.resolution_m == costmap.resolution_m
    assert loaded.cost[5, 5] == np.float32(0.88)
    assert bool(loaded.obstacle[5, 5]) is True
    assert abs(loaded.slope_deg[5, 5] - 15.2) < 1e-4
