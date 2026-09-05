"""World-referenced multi-layer terrain map for GPS-denied autonomous navigation.

Stores globally referenced terrain drivability, explicit hazards, physical obstacles,
elevation geometry, confidence, and observation history on a metric raster grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


@dataclass
class WorldMapMetadata:
    """Metadata describing the spatial reference and raster resolution of the WorldTerrainMap."""

    resolution_m: float = 0.10          # Metric size of each grid cell (meters/cell)
    x_bounds_m: Tuple[float, float] = (-20.0, 20.0) # World X bounds [min_x, max_x]
    y_bounds_m: Tuple[float, float] = (-20.0, 20.0) # World Y bounds [min_y, max_y]
    frame_id: str = "odom"              # Canonical reference frame
    created_timestamp: float = 0.0      # Initialization timestamp (seconds)


class WorldTerrainMap:
    """Global-aligned multi-layer metric terrain representation.
    
    Coordinate Convention:
      - World X: East / Initial forward (meters)
      - World Y: North / Initial left (meters)
      - Grid origin (row=0, col=0) corresponds to (max_x, max_y).
      - row index increases as X decreases (top-down raster).
      - col index increases as Y decreases.
    """

    def __init__(
        self,
        resolution_m: float = 0.10,
        x_bounds_m: Tuple[float, float] = (-20.0, 20.0),
        y_bounds_m: Tuple[float, float] = (-20.0, 20.0),
        frame_id: str = "odom",
        created_timestamp: float = 0.0,
        unknown_cost_default: float = 0.50
    ) -> None:
        self.resolution_m = float(resolution_m)
        self.x_bounds_m = (float(x_bounds_m[0]), float(x_bounds_m[1]))
        self.y_bounds_m = (float(y_bounds_m[0]), float(y_bounds_m[1]))
        self.frame_id = frame_id
        self.created_timestamp = created_timestamp
        self.unknown_cost_default = float(unknown_cost_default)

        # Compute grid raster dimensions
        self.n_rows = int(np.round((self.x_bounds_m[1] - self.x_bounds_m[0]) / self.resolution_m))
        self.n_cols = int(np.round((self.y_bounds_m[1] - self.y_bounds_m[0]) / self.resolution_m))

        # Multi-layer raster tensors
        self.cost = np.full((self.n_rows, self.n_cols), self.unknown_cost_default, dtype=np.float32)
        self.traversability = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        self.obstacle = np.zeros((self.n_rows, self.n_cols), dtype=bool)
        self.hazard = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        self.unknown = np.ones((self.n_rows, self.n_cols), dtype=bool) # Unknown space strictly preserved!
        self.confidence = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        self.slope = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        self.roughness = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        self.height_var = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        self.observation_count = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        self.last_observed_timestamp = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)

        # Metadata object
        self.metadata = WorldMapMetadata(
            resolution_m=self.resolution_m,
            x_bounds_m=self.x_bounds_m,
            y_bounds_m=self.y_bounds_m,
            frame_id=self.frame_id,
            created_timestamp=self.created_timestamp
        )

    @property
    def shape(self) -> Tuple[int, int]:
        """Raster tensor shape (n_rows, n_cols)."""
        return self.cost.shape

    def world_to_grid(self, x_w: float, y_w: float) -> Tuple[int, int]:
        """Convert metric world coordinates (x, y) to integer raster indices (row, col)."""
        row = int((self.x_bounds_m[1] - x_w) / self.resolution_m)
        col = int((self.y_bounds_m[1] - y_w) / self.resolution_m)
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert raster indices (row, col) to metric world coordinates (x, y) at cell center."""
        x_w = self.x_bounds_m[1] - (row + 0.5) * self.resolution_m
        y_w = self.y_bounds_m[1] - (col + 0.5) * self.resolution_m
        return float(x_w), float(y_w)

    def world_to_grid_vec(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized conversion of metric world coordinates to grid indices."""
        rows = np.floor((self.x_bounds_m[1] - xs) / self.resolution_m).astype(np.int32)
        cols = np.floor((self.y_bounds_m[1] - ys) / self.resolution_m).astype(np.int32)
        return rows, cols

    def is_in_bounds(self, x_w: float, y_w: float) -> bool:
        """Check if metric coordinate falls inside active world map extent."""
        return (self.x_bounds_m[0] <= x_w < self.x_bounds_m[1]) and (self.y_bounds_m[0] <= y_w < self.y_bounds_m[1])

    def is_grid_in_bounds(self, row: int, col: int) -> bool:
        """Check if raster index falls within allocated tensor dimensions."""
        return (0 <= row < self.n_rows) and (0 <= col < self.n_cols)

    def get_cost(self, x_w: float, y_w: float, default: float = 1.0) -> float:
        """Query continuous traversal cost at world metric coordinate."""
        if not self.is_in_bounds(x_w, y_w):
            return default
        r, c = self.world_to_grid(x_w, y_w)
        if not self.is_grid_in_bounds(r, c):
            return default
        return float(self.cost[r, c])

    def is_cell_traversable(self, x_w: float, y_w: float, cost_threshold: float = 0.65) -> bool:
        """Query binary traversability safe flag at world metric coordinate."""
        if not self.is_in_bounds(x_w, y_w):
            return False
        r, c = self.world_to_grid(x_w, y_w)
        if not self.is_grid_in_bounds(r, c):
            return False
        return (
            (not self.obstacle[r, c])
            and (self.hazard[r, c] < 0.40)
            and (self.cost[r, c] < cost_threshold)
            and (not self.unknown[r, c])
        )

    def memory_bytes(self) -> int:
        """Calculate exact RAM footprint of raster layers in bytes."""
        total = (
            self.cost.nbytes +
            self.traversability.nbytes +
            self.obstacle.nbytes +
            self.hazard.nbytes +
            self.unknown.nbytes +
            self.confidence.nbytes +
            self.slope.nbytes +
            self.roughness.nbytes +
            self.height_var.nbytes +
            self.observation_count.nbytes +
            self.last_observed_timestamp.nbytes
        )
        return total

    def export_submap(
        self,
        center_x_m: float,
        center_y_m: float,
        extent_x_m: float,
        extent_y_m: float
    ) -> Dict[str, Any]:
        """Extract a localized world submap window centered around (center_x, center_y)."""
        x_min = center_x_m - extent_x_m / 2.0
        x_max = center_x_m + extent_x_m / 2.0
        y_min = center_y_m - extent_y_m / 2.0
        y_max = center_y_m + extent_y_m / 2.0

        r_min, c_max = self.world_to_grid(x_max, y_min)
        r_max, c_min = self.world_to_grid(x_min, y_max)

        r_min = max(0, r_min)
        r_max = min(self.n_rows, r_max)
        c_min = max(0, c_min)
        c_max = min(self.n_cols, c_max)

        return {
            "center": (float(center_x_m), float(center_y_m)),
            "x_bounds": (float(x_min), float(x_max)),
            "y_bounds": (float(y_min), float(y_max)),
            "cost_slice": self.cost[r_min:r_max, c_min:c_max].copy(),
            "obstacle_slice": self.obstacle[r_min:r_max, c_min:c_max].copy(),
            "hazard_slice": self.hazard[r_min:r_max, c_min:c_max].copy(),
            "confidence_slice": self.confidence[r_min:r_max, c_min:c_max].copy(),
            "unknown_slice": self.unknown[r_min:r_max, c_min:c_max].copy(),
        }

    def summary_stats(self) -> Dict[str, Any]:
        """Compute coverage and safety statistics across the world map."""
        total_cells = self.n_rows * self.n_cols
        observed_mask = ~self.unknown
        observed_cells = int(np.sum(observed_mask))
        obstacle_cells = int(np.sum(self.obstacle))
        hazard_cells = int(np.sum(self.hazard >= 0.40))
        critical_cells = int(np.sum(self.hazard >= 0.70))

        return {
            "total_cells": total_cells,
            "observed_cells": observed_cells,
            "coverage_pct": float(observed_cells / max(1, total_cells) * 100.0),
            "obstacle_cells": obstacle_cells,
            "hazard_cells": hazard_cells,
            "critical_cells": critical_cells,
            "mean_confidence": float(np.mean(self.confidence[observed_mask])) if observed_cells > 0 else 0.0,
            "memory_mb": float(self.memory_bytes() / (1024 * 1024)),
        }
