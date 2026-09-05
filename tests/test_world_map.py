"""Unit tests for WorldTerrainMap grid indexing, layers, and memory bounds."""

import numpy as np
import pytest

from ai.spatial_fusion.world_map import WorldTerrainMap


def test_world_map_initialization():
    """Verify world map initial layer states, shapes, and unknown space preservation."""
    wmap = WorldTerrainMap(
        resolution_m=0.10,
        x_bounds_m=(-10.0, 10.0),
        y_bounds_m=(-10.0, 10.0),
        unknown_cost_default=0.50
    )
    assert wmap.shape == (200, 200)
    assert np.all(wmap.unknown)  # 100% initialized as UNKNOWN
    assert np.allclose(wmap.cost, 0.50)  # Neutral cost
    assert np.allclose(wmap.confidence, 0.0)
    assert np.all(~wmap.obstacle)


def test_coordinate_roundtrip_consistency():
    """Verify world_to_grid and grid_to_world are mathematically consistent."""
    wmap = WorldTerrainMap(
        resolution_m=0.10,
        x_bounds_m=(-20.0, 20.0),
        y_bounds_m=(-20.0, 20.0)
    )
    # Test specific cells across the grid
    test_cells = [(0, 0), (100, 150), (399, 399), (200, 200)]
    for r, c in test_cells:
        xw, yw = wmap.grid_to_world(r, c)
        r_rec, c_rec = wmap.world_to_grid(xw, yw)
        assert (r, c) == (r_rec, c_rec), f"Failed for cell ({r}, {c}) -> ({xw}, {yw}) -> ({r_rec}, {c_rec})"


def test_vectorized_coordinate_mapping():
    """Verify vectorized world_to_grid_vec matches scalar world_to_grid."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    xs = np.array([5.0, 0.0, -5.0])
    ys = np.array([2.5, 0.0, -7.5])

    rows_vec, cols_vec = wmap.world_to_grid_vec(xs, ys)
    for i in range(len(xs)):
        r_sc, c_sc = wmap.world_to_grid(xs[i], ys[i])
        assert rows_vec[i] == r_sc
        assert cols_vec[i] == c_sc


def test_bounds_checks():
    """Verify boundary conditions and out-of-bounds rejection."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0))
    assert wmap.is_in_bounds(0.0, 0.0)
    assert wmap.is_in_bounds(-4.99, 4.99)
    assert not wmap.is_in_bounds(5.01, 0.0)
    assert not wmap.is_in_bounds(-5.01, 0.0)
    assert not wmap.is_in_bounds(0.0, 5.5)

    # Out-of-bounds get_cost returns default
    assert wmap.get_cost(10.0, 10.0, default=1.0) == 1.0


def test_submap_extraction():
    """Verify submap extraction returns valid window slices."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-20.0, 20.0), y_bounds_m=(-20.0, 20.0))
    # Modify a specific cell in world
    r, c = wmap.world_to_grid(2.0, 3.0)
    wmap.obstacle[r, c] = True
    wmap.cost[r, c] = 1.0

    submap = wmap.export_submap(center_x_m=2.0, center_y_m=3.0, extent_x_m=4.0, extent_y_m=4.0)
    assert "cost_slice" in submap
    assert "obstacle_slice" in submap
    assert np.any(submap["obstacle_slice"])


def test_memory_footprint():
    """Verify memory_bytes calculates realistic non-zero footprint."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-20.0, 20.0), y_bounds_m=(-20.0, 20.0))
    n_bytes = wmap.memory_bytes()
    assert n_bytes > 0
    # 400x400 grid with 11 layers is roughly 5-10 MB
    mb = n_bytes / (1024 * 1024)
    assert 2.0 <= mb <= 20.0
