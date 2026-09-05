"""Unit tests for perception fusion, cost models, and lethal threshold cutoffs."""

import numpy as np
import pytest

from ai.perception_fusion.cost_model import CostModelWeights, TraversabilityCostModel
from ai.perception_fusion.interfaces import FusedTraversabilityMap
from ai.perception_fusion.traversability_fusion import TraversabilityFusion
from ai.traversability.bev_projection import BEVTraversabilityMap


def test_cost_model_free_ground():
    """Verify high traversability on flat ground produces near-zero driving cost."""
    cost_model = TraversabilityCostModel()
    # 1.0 traversability (perfect road), 0 slope, 0 height
    cost = cost_model.compute_cell_cost(trav_score=1.0, slope_deg=0.0, height_var_m=0.0)
    assert np.isclose(cost, 0.0, atol=1e-3)


def test_cost_model_lethal_slope():
    """Verify slopes exceeding critical rollover angle (>28 deg) force cost to 1.0 (lethal)."""
    cost_model = TraversabilityCostModel()
    cost = cost_model.compute_cell_cost(trav_score=1.0, slope_deg=30.0)
    assert cost == 1.0


def test_cost_model_lethal_step():
    """Verify vertical step height exceeding vehicle chassis capability forces cost to 1.0."""
    cost_model = TraversabilityCostModel()
    cost = cost_model.compute_cell_cost(trav_score=1.0, height_var_m=0.25)
    assert cost == 1.0


def test_cost_model_unknown_cost():
    """Verify unobserved / out-of-view cells receive configured exploration penalty."""
    weights = CostModelWeights(unknown_cost=0.55)
    cost_model = TraversabilityCostModel(weights=weights)
    cost = cost_model.compute_cell_cost(trav_score=1.0, is_unknown=True)
    assert cost == 0.55


def test_traversability_fusion_grid():
    """Verify 2D grid fusion produces valid FusedTraversabilityMap."""
    bev_map = BEVTraversabilityMap(
        grid=np.full((20, 20), 0.8, dtype=np.float32),
        unknown_mask=np.zeros((20, 20), dtype=bool),
        resolution_m=0.1,
        x_range_m=(0.5, 2.5),
        y_range_m=(-1.0, 1.0)
    )
    # Mark corner unknown
    bev_map.unknown_mask[0:5, 0:5] = True

    fusion = TraversabilityFusion()
    fused = fusion.fuse(bev_map)

    assert isinstance(fused, FusedTraversabilityMap)
    assert fused.cost_grid.shape == (20, 20)
    # Known region cost
    known_cost = fused.cost_grid[~bev_map.unknown_mask]
    assert (known_cost >= 0.0).all() and (known_cost <= 1.0).all()
    # Unknown region cost equals default 0.50
    assert (fused.cost_grid[bev_map.unknown_mask] == 0.50).all()
