"""Obstacle inflation and vehicle footprint expansion using Euclidean Distance Transform."""

from __future__ import annotations

from typing import Optional
import cv2
import numpy as np

from ai.costmap.base import VehicleFootprint
from ai.costmap.bev_costmap import BEVTerrainCostmap


class ObstacleInflator:
    """Expands lethal obstacles using vehicle safety footprint and exponential cost decay."""

    def __init__(
        self,
        footprint: Optional[VehicleFootprint] = None,
        resolution_m: float = 0.10
    ) -> None:
        self.footprint = footprint or VehicleFootprint()
        self.res = resolution_m
        self.r_inscribed = self.footprint.inscribed_radius_m
        self.r_inflation = self.footprint.inflation_radius_m
        self.alpha = self.footprint.cost_scaling_factor

    def inflate_costmap(self, costmap: BEVTerrainCostmap) -> BEVTerrainCostmap:
        """Apply obstacle inflation to a BEVTerrainCostmap.

        Returns:
            A new BEVTerrainCostmap with inflated obstacle and cost layers.
        """
        obstacle_mask = costmap.obstacle.astype(np.uint8)

        # Invert obstacle mask for distance transform (0 at obstacles, 1 at free space)
        free_mask = (1 - obstacle_mask).astype(np.uint8)

        # Distance to nearest obstacle in pixels
        dist_pixels = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)
        dist_m = dist_pixels * self.res

        # 1. Inscribed Zone: Inside physical vehicle collision footprint -> Lethal cost
        inscribed_zone = (dist_m <= self.r_inscribed) & (~costmap.unknown)
        new_cost = costmap.cost.copy()
        new_cost[inscribed_zone] = 1.00
        new_obstacle = costmap.obstacle | (inscribed_zone & ~costmap.unknown)

        # 2. Inflation Zone: Between inscribed and inflation radius -> Exponential decay
        inflation_zone = (dist_m > self.r_inscribed) & (dist_m <= self.r_inflation) & (~costmap.unknown)
        decay_cost = np.exp(-self.alpha * (dist_m - self.r_inscribed))
        new_cost[inflation_zone] = np.maximum(new_cost[inflation_zone], decay_cost[inflation_zone])

        return BEVTerrainCostmap(
            cost=new_cost.astype(np.float32),
            traversability=costmap.traversability,
            obstacle=new_obstacle,
            hazard=costmap.hazard,
            unknown=costmap.unknown,
            confidence=costmap.confidence,
            slope_deg=costmap.slope_deg,
            roughness=costmap.roughness,
            height_var=costmap.height_var,
            observation_count=costmap.observation_count,
            metadata=costmap.metadata
        )
