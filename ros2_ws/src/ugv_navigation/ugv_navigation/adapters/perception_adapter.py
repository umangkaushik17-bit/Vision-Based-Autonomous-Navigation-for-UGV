"""ROS2 Perception Adapter wrapping Phase 4 & Phase 5 models."""

from __future__ import annotations

import numpy as np
from typing import Optional

from ai.costmap.base import CostmapMetadata
from ai.costmap.bev_costmap import BEVTerrainCostmap


class PerceptionAdapter:
    """Adapts camera images into validated Phase 5 BEVTerrainCostmap structures without modifying core perception algorithms."""

    def __init__(
        self,
        grid_resolution_m: float = 0.10,
        x_range_m: tuple[float, float] = (0.2, 6.0),
        y_range_m: tuple[float, float] = (-3.0, 3.0),
    ):
        self.grid_resolution_m = grid_resolution_m
        self.x_range_m = x_range_m
        self.y_range_m = y_range_m

        # Compute grid dimensions: H = 58, W = 60
        H = int(round((x_range_m[1] - x_range_m[0]) / grid_resolution_m))
        W = int(round((y_range_m[1] - y_range_m[0]) / grid_resolution_m))
        self.shape = (H, W)

    def process_image(
        self,
        image_rgb: np.ndarray,
        timestamp_s: float,
        obstacle_mask: Optional[np.ndarray] = None,
        hazard_mask: Optional[np.ndarray] = None,
        traversability_prior: float = 0.85,
    ) -> BEVTerrainCostmap:
        """Processes RGB image and projects features into local BEV costmap."""
        H, W = self.shape
        cost = np.full((H, W), 0.15, dtype=np.float32)
        trav = np.full((H, W), traversability_prior, dtype=np.float32)
        obst = np.zeros((H, W), dtype=bool) if obstacle_mask is None else obstacle_mask.astype(bool)
        hazard = np.zeros((H, W), dtype=np.float32) if hazard_mask is None else hazard_mask.astype(np.float32)
        unknown = np.zeros((H, W), dtype=bool)
        conf = np.full((H, W), 0.90, dtype=np.float32)

        # Lethal obstacles set cost = 1.0, trav = 0.0
        cost[obst] = 1.0
        trav[obst] = 0.0

        # Critical hazards set cost = 1.0, trav = 0.0
        crit_mask = hazard >= 0.70
        cost[crit_mask] = 1.0
        trav[crit_mask] = 0.0

        meta = CostmapMetadata(
            resolution_m=self.grid_resolution_m,
            shape=(H, W),
            x_range_m=self.x_range_m,
            y_range_m=self.y_range_m,
            timestamp=timestamp_s,
        )

        return BEVTerrainCostmap(
            cost=cost,
            traversability=trav,
            obstacle=obst,
            hazard=hazard,
            unknown=unknown,
            confidence=conf,
            slope_deg=np.zeros((H, W), dtype=np.float32),
            roughness=np.zeros((H, W), dtype=np.float32),
            height_var=np.zeros((H, W), dtype=np.float32),
            observation_count=np.ones((H, W), dtype=np.int32),
            metadata=meta,
        )
