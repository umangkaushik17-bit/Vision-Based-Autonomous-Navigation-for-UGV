"""Multi-modal traversability and terrain geometry fusion engine."""

from __future__ import annotations

from typing import Optional
import numpy as np

from ai.perception_fusion.cost_model import CostModelWeights, TraversabilityCostModel
from ai.perception_fusion.interfaces import FusedTraversabilityMap
from ai.traversability.bev_projection import BEVTraversabilityMap


class TraversabilityFusion:
    """Combines 2D BEV learned traversability with 3D physical terrain properties."""

    def __init__(
        self,
        weights: Optional[CostModelWeights] = None,
    ) -> None:
        self.cost_model = TraversabilityCostModel(weights=weights)

    def fuse(
        self,
        bev_map: BEVTraversabilityMap,
        slope_grid: Optional[np.ndarray] = None,
        flatness_grid: Optional[np.ndarray] = None,
        height_var_grid: Optional[np.ndarray] = None,
        obstacle_grid: Optional[np.ndarray] = None
    ) -> FusedTraversabilityMap:
        """Fuse inputs into a unified FusedTraversabilityMap."""
        cost_grid = self.cost_model.compute_grid_cost(
            bev_traversability=bev_map.grid,
            unknown_mask=bev_map.unknown_mask,
            slope_grid=slope_grid,
            flatness_grid=flatness_grid,
            height_var_grid=height_var_grid,
            obstacle_grid=obstacle_grid
        )

        # Confidence map is high for observed cells with clear safe/unsafe separation
        confidence = np.ones_like(cost_grid, dtype=np.float32)
        confidence[bev_map.unknown_mask] = 0.0

        return FusedTraversabilityMap(
            cost_grid=cost_grid,
            confidence_grid=confidence,
            unknown_mask=bev_map.unknown_mask,
            resolution_m=bev_map.resolution_m,
            x_range_m=bev_map.x_range_m,
            y_range_m=bev_map.y_range_m,
            metadata={
                "mean_cost": float(cost_grid[~bev_map.unknown_mask].mean()) if (~bev_map.unknown_mask).any() else 0.5,
                "free_ratio": float((cost_grid < 0.3).mean()),
                "lethal_ratio": float((cost_grid >= 0.95).mean())
            }
        )
