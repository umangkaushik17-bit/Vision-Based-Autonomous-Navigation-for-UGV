"""Canonical BEV Terrain Costmap representation.

High-performance raster tensor representation storing multi-layer terrain
semantics, physical geometry, explicit obstacles, and continuous traversal costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from ai.costmap.base import CostmapLayerType, CostmapMetadata


@dataclass
class BEVTerrainCostmap:
    """Robot-centric Bird's-Eye-View Terrain Costmap.

    Coordinate system (ISO 8855 / ROS standard):
      - X forward: Distance ahead of vehicle (meters).
      - Y lateral: Left (+) / Right (-) relative to vehicle centerline (meters).
      - Z up: Elevation above ground contact.
      - Grid origin: Top-left of array (row=0, col=0) maps to (x_max, y_max).
    """

    cost: np.ndarray                 # (H, W) float32 in [0.0, 1.0], 1.0 = LETHAL/BLOCKED
    traversability: np.ndarray       # (H, W) float32 in [0.0, 1.0], 1.0 = HIGH DRIVABILITY
    obstacle: np.ndarray             # (H, W) bool, True = hard impassable obstacle
    hazard: np.ndarray               # (H, W) float32 in [0.0, 1.0], explicit hazard severity
    unknown: np.ndarray              # (H, W) bool, True = unobserved or high uncertainty
    confidence: np.ndarray           # (H, W) float32 in [0.0, 1.0], sensory confidence
    slope_deg: np.ndarray            # (H, W) float32, surface inclination in degrees
    roughness: np.ndarray            # (H, W) float32, surface coplanarity variance
    height_var: np.ndarray           # (H, W) float32, vertical step variation in meters
    observation_count: np.ndarray    # (H, W) int32, number of times cell was observed
    metadata: CostmapMetadata

    @property
    def shape(self) -> Tuple[int, int]:
        """Return grid shape (rows, cols)."""
        return self.cost.shape

    @property
    def resolution_m(self) -> float:
        """Cell resolution in meters/cell."""
        return self.metadata.resolution_m

    @property
    def x_range_m(self) -> Tuple[float, float]:
        """Forward metric bounds [min_x, max_x]."""
        return self.metadata.x_range_m

    @property
    def y_range_m(self) -> Tuple[float, float]:
        """Lateral metric bounds [min_y, max_y]."""
        return self.metadata.y_range_m

    def meter_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert robot metric coordinate (x_forward, y_left) to (row, col) grid indices."""
        x_min, x_max = self.x_range_m
        y_min, y_max = self.y_range_m
        res = self.resolution_m
        row = int((x_max - x_m) / res)
        col = int((y_max - y_m) / res)
        return row, col

    def grid_to_meter(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid (row, col) back to metric coordinates (x_forward, y_left) at cell center."""
        x_min, x_max = self.x_range_m
        y_min, y_max = self.y_range_m
        res = self.resolution_m
        x_m = x_max - (row + 0.5) * res
        y_m = y_max - (col + 0.5) * res
        return float(x_m), float(y_m)

    def meters_to_grid_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized conversion of metric coordinates to grid coordinates."""
        x_min, x_max = self.x_range_m
        y_min, y_max = self.y_range_m
        res = self.resolution_m
        rows = ((x_max - xs) / res).astype(np.int32)
        cols = ((y_max - ys) / res).astype(np.int32)
        return rows, cols

    def is_in_bounds(self, x_m: float, y_m: float) -> bool:
        """Check if metric coordinate falls inside the active BEV map bounds."""
        x_min, x_max = self.x_range_m
        y_min, y_max = self.y_range_m
        return (x_min <= x_m < x_max) and (y_min <= y_m < y_max)

    def is_grid_in_bounds(self, row: int, col: int) -> bool:
        """Check if row and col are within the raster grid dimensions."""
        n_rows, n_cols = self.shape
        return (0 <= row < n_rows) and (0 <= col < n_cols)

    def get_cost(self, x_m: float, y_m: float, default: float = 1.0) -> float:
        """Query continuous cost at metric coordinate (x, y). Out of bounds returns default."""
        if not self.is_in_bounds(x_m, y_m):
            return default
        r, c = self.meter_to_grid(x_m, y_m)
        if self.is_grid_in_bounds(r, c):
            return float(self.cost[r, c])
        return default

    def is_passable(self, x_m: float, y_m: float, max_cost: float = 0.65) -> bool:
        """Determine if a metric coordinate is safe and traversable for navigation."""
        if not self.is_in_bounds(x_m, y_m):
            return False
        r, c = self.meter_to_grid(x_m, y_m)
        if not self.is_grid_in_bounds(r, c):
            return False
        if self.obstacle[r, c]:
            return False
        return bool(self.cost[r, c] < max_cost)

    def is_unknown(self, x_m: float, y_m: float) -> bool:
        """Check if a coordinate has unknown or unobserved status."""
        if not self.is_in_bounds(x_m, y_m):
            return True
        r, c = self.meter_to_grid(x_m, y_m)
        if not self.is_grid_in_bounds(r, c):
            return True
        return bool(self.unknown[r, c])

    def get_layer(self, layer_type: CostmapLayerType) -> np.ndarray:
        """Retrieve 2D array corresponding to requested layer type."""
        mapping = {
            CostmapLayerType.FINAL_COST: self.cost,
            CostmapLayerType.TRAVERSABILITY: self.traversability,
            CostmapLayerType.HAZARD: self.hazard,
            CostmapLayerType.SLOPE: self.slope_deg,
            CostmapLayerType.ROUGHNESS: self.roughness,
            CostmapLayerType.HEIGHT: self.height_var,
            CostmapLayerType.UNKNOWN: self.unknown.astype(np.float32),
            CostmapLayerType.CONFIDENCE: self.confidence,
            CostmapLayerType.TEMPORAL_PERSISTENCE: self.observation_count.astype(np.float32)
        }
        if layer_type in mapping:
            return mapping[layer_type]
        raise ValueError(f"Unknown layer type: {layer_type}")

    def summary_stats(self) -> Dict[str, Any]:
        """Compute statistical breakdown of costmap composition."""
        total_cells = float(self.cost.size)
        safe_cells = (self.cost < 0.35) & (~self.obstacle) & (~self.unknown)
        caution_cells = (self.cost >= 0.35) & (self.cost < 0.70) & (~self.obstacle) & (~self.unknown)
        blocked_cells = self.obstacle | (self.cost >= 0.70)
        unknown_cells = self.unknown

        return {
            "total_cells": int(total_cells),
            "resolution_m": self.resolution_m,
            "grid_dimensions": list(self.shape),
            "safe_cell_ratio": round(float(safe_cells.sum()) / total_cells, 4),
            "caution_cell_ratio": round(float(caution_cells.sum()) / total_cells, 4),
            "blocked_cell_ratio": round(float(blocked_cells.sum()) / total_cells, 4),
            "unknown_cell_ratio": round(float(unknown_cells.sum()) / total_cells, 4),
            "mean_cost": round(float(self.cost.mean()), 4),
            "max_slope_deg": round(float(self.slope_deg.max()), 2),
            "mean_confidence": round(float(self.confidence.mean()), 4)
        }

    @classmethod
    def create_empty(
        cls,
        x_range_m: Tuple[float, float] = (0.2, 6.0),
        y_range_m: Tuple[float, float] = (-3.0, 3.0),
        resolution_m: float = 0.10,
        frame_id: str = "base_link",
        timestamp: float = 0.0,
        default_cost: float = 0.50
    ) -> BEVTerrainCostmap:
        """Factory creating a blank initial costmap."""
        n_x = int(np.ceil((x_range_m[1] - x_range_m[0]) / resolution_m))
        n_y = int(np.ceil((y_range_m[1] - y_range_m[0]) / resolution_m))

        metadata = CostmapMetadata(
            resolution_m=resolution_m,
            shape=(n_x, n_y),
            x_range_m=x_range_m,
            y_range_m=y_range_m,
            frame_id=frame_id,
            timestamp=timestamp
        )

        return cls(
            cost=np.full((n_x, n_y), default_cost, dtype=np.float32),
            traversability=np.full((n_x, n_y), 0.50, dtype=np.float32),
            obstacle=np.zeros((n_x, n_y), dtype=bool),
            hazard=np.zeros((n_x, n_y), dtype=np.float32),
            unknown=np.ones((n_x, n_y), dtype=bool),
            confidence=np.zeros((n_x, n_y), dtype=np.float32),
            slope_deg=np.zeros((n_x, n_y), dtype=np.float32),
            roughness=np.zeros((n_x, n_y), dtype=np.float32),
            height_var=np.zeros((n_x, n_y), dtype=np.float32),
            observation_count=np.zeros((n_x, n_y), dtype=np.int32),
            metadata=metadata
        )
