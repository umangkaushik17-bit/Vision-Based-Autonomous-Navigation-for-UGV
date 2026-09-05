"""Motion compensation and spatial alignment for robot-centric BEV costmaps.

Transforms previous BEV grids into the current vehicle frame under an SE(2) pose delta
(dx_forward, dy_lateral, dyaw_rad) using rigid Euclidean affine warping.
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

from ai.costmap.base import CostmapMetadata
from ai.costmap.bev_costmap import BEVTerrainCostmap


class MotionCompensator:
    """Warps and aligns prior BEV grids to the current vehicle coordinate frame."""

    def __init__(self, metadata: CostmapMetadata) -> None:
        self.metadata = metadata
        self.res = metadata.resolution_m
        self.x_min, self.x_max = metadata.x_range_m
        self.y_min, self.y_max = metadata.y_range_m
        self.n_rows, self.n_cols = metadata.shape

    def compute_affine_matrix(self, dx_m: float, dy_m: float, dyaw_rad: float) -> np.ndarray:
        """Compute 2x3 affine transformation matrix mapping prior grid (col, row) to current grid (col, row).

        In metric robot frame:
          P_curr = R(-dyaw) * (P_prev - [dx, dy]^T)
        """
        c_th = float(np.cos(dyaw_rad))
        s_th = float(np.sin(dyaw_rad))

        M_prev_to_metric = np.array([
            [-self.res, 0.0, self.y_max - 0.5 * self.res],
            [0.0, -self.res, self.x_max - 0.5 * self.res],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        T_prev_to_curr = np.array([
            [c_th, -s_th, -c_th * dy_m + s_th * dx_m],
            [s_th, c_th, -s_th * dy_m - c_th * dx_m],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        M_metric_to_curr = np.array([
            [-1.0 / self.res, 0.0, (self.y_max - 0.5 * self.res) / self.res],
            [0.0, -1.0 / self.res, (self.x_max - 0.5 * self.res) / self.res],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        M_full = M_metric_to_curr @ T_prev_to_curr @ M_prev_to_metric
        return M_full[:2, :].astype(np.float32)

    def warp_layer(
        self,
        layer: np.ndarray,
        dx_m: float,
        dy_m: float,
        dyaw_rad: float,
        border_value: float = 0.5,
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """Warp an individual 2D raster layer under vehicle relative motion."""
        # Short-circuit identity transform
        if abs(dx_m) < 1e-4 and abs(dy_m) < 1e-4 and abs(dyaw_rad) < 1e-4:
            return layer.copy()

        M = self.compute_affine_matrix(dx_m, dy_m, dyaw_rad)
        warped = cv2.warpAffine(
            layer.astype(np.float32),
            M,
            (self.n_cols, self.n_rows),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value
        )
        return warped

    def warp_costmap(
        self,
        costmap: BEVTerrainCostmap,
        dx_m: float,
        dy_m: float,
        dyaw_rad: float
    ) -> BEVTerrainCostmap:
        """Warp all layers of a BEVTerrainCostmap into current vehicle frame."""
        if abs(dx_m) < 1e-4 and abs(dy_m) < 1e-4 and abs(dyaw_rad) < 1e-4:
            return costmap

        warped_cost = self.warp_layer(costmap.cost, dx_m, dy_m, dyaw_rad, border_value=0.50)
        warped_trav = self.warp_layer(costmap.traversability, dx_m, dy_m, dyaw_rad, border_value=0.50)
        warped_hazard = self.warp_layer(costmap.hazard, dx_m, dy_m, dyaw_rad, border_value=0.00)
        warped_obstacle = self.warp_layer(costmap.obstacle.astype(np.float32), dx_m, dy_m, dyaw_rad, border_value=0.00, interpolation=cv2.INTER_NEAREST) > 0.5
        warped_unknown = self.warp_layer(costmap.unknown.astype(np.float32), dx_m, dy_m, dyaw_rad, border_value=1.00, interpolation=cv2.INTER_NEAREST) > 0.5
        warped_conf = self.warp_layer(costmap.confidence, dx_m, dy_m, dyaw_rad, border_value=0.00)
        warped_slope = self.warp_layer(costmap.slope_deg, dx_m, dy_m, dyaw_rad, border_value=0.00)
        warped_rough = self.warp_layer(costmap.roughness, dx_m, dy_m, dyaw_rad, border_value=0.00)
        warped_height = self.warp_layer(costmap.height_var, dx_m, dy_m, dyaw_rad, border_value=0.00)
        warped_obs = self.warp_layer(costmap.observation_count.astype(np.float32), dx_m, dy_m, dyaw_rad, border_value=0.00, interpolation=cv2.INTER_NEAREST).astype(np.int32)

        return BEVTerrainCostmap(
            cost=warped_cost.astype(np.float32),
            traversability=warped_trav.astype(np.float32),
            obstacle=warped_obstacle,
            hazard=warped_hazard.astype(np.float32),
            unknown=warped_unknown,
            confidence=warped_conf.astype(np.float32),
            slope_deg=warped_slope.astype(np.float32),
            roughness=warped_rough.astype(np.float32),
            height_var=warped_height.astype(np.float32),
            observation_count=warped_obs,
            metadata=costmap.metadata
        )
