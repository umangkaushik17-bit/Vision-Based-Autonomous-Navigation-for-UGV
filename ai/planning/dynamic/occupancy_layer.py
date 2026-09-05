"""Time-indexed dynamic occupancy representation for dynamic obstacle predictive collision checking.

Milestone 13.7:
Converts predicted obstacle trajectories (from Milestone 13.6) into a multi-horizon,
spatiotemporally indexed raster layer. Preserves distinct nominal physical obstacle
footprints vs graded uncertainty expansion to prevent pathological map blockage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrajectoryWaypoint,
    compute_obb_corners,
)


class DynamicRiskLevel(IntEnum):
    """Categorical risk classification for dynamic obstacle occupancy."""

    FREE = 0            # No dynamic obstacle influence (cost = 0.0)
    LOW_RISK = 1        # Outer uncertainty zone (2-3 sigma, cost in [0.15, 0.40))
    MEDIUM_RISK = 2     # Intermediate uncertainty zone (1-2 sigma, cost in [0.40, 0.70))
    HIGH_RISK = 3       # Immediate uncertainty envelope (0-1 sigma, cost in [0.70, 0.95))
    LETHAL = 4          # Physical obstacle footprint boundary (cost = 1.0)


@dataclass
class DynamicOccupancyConfig:
    """Configuration parameters for the Dynamic Occupancy Layer.

    Attributes:
        resolution_m: Metric raster resolution (meters/cell). Default 0.10 matches Phase 5/9.
        x_bounds_m: World X bounds [min_x, max_x] in meters. Default (-20.0, 20.0).
        y_bounds_m: World Y bounds [min_y, max_y] in meters. Default (-20.0, 20.0).
        frame_id: Canonical reference coordinate frame. Default "odom".
        time_bins: Discrete relative future time offsets in seconds.
        collision_cost_threshold: Cost threshold above which cell is considered in dynamic collision.
            Default 0.65 matches Phase 12 max_cost contract.
        high_risk_threshold: Cost threshold for HIGH_RISK classification (0.70).
        medium_risk_threshold: Cost threshold for MEDIUM_RISK classification (0.40).
        low_risk_threshold: Cost threshold for LOW_RISK classification (0.15).
        max_horizon_s: Maximum temporal prediction horizon (seconds).
    """

    resolution_m: float = 0.10
    x_bounds_m: Tuple[float, float] = (-20.0, 20.0)
    y_bounds_m: Tuple[float, float] = (-20.0, 20.0)
    frame_id: str = "odom"
    time_bins: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    )
    collision_cost_threshold: float = 0.65
    high_risk_threshold: float = 0.70
    medium_risk_threshold: float = 0.40
    low_risk_threshold: float = 0.15
    sigma_scale: float = 3.0
    max_horizon_s: float = 2.00

    def __post_init__(self) -> None:
        if self.resolution_m <= 0.0:
            raise ValueError(f"resolution_m must be strictly positive, got {self.resolution_m}")
        if self.sigma_scale <= 0.0:
            raise ValueError(f"sigma_scale must be strictly positive, got {self.sigma_scale}")
        if self.x_bounds_m[0] >= self.x_bounds_m[1]:
            raise ValueError(f"Invalid x_bounds_m: {self.x_bounds_m}")
        if self.y_bounds_m[0] >= self.y_bounds_m[1]:
            raise ValueError(f"Invalid y_bounds_m: {self.y_bounds_m}")
        if not self.time_bins:
            raise ValueError("time_bins must not be empty")
        for i, t in enumerate(self.time_bins):
            if t < 0.0:
                raise ValueError(f"time_bins must be non-negative, got {t}")
            if i > 0 and t <= self.time_bins[i - 1]:
                raise ValueError(f"time_bins must be strictly increasing: {t} <= {self.time_bins[i-1]}")


@dataclass
class DynamicOccupancyGridSlice:
    """Spatial 2D raster occupancy grid at a single future time offset."""

    time_offset: float
    timestamp: float
    cost: np.ndarray              # Shape (n_rows, n_cols), dtype float32, range [0.0, 1.0]
    risk_level: np.ndarray        # Shape (n_rows, n_cols), dtype uint8 (DynamicRiskLevel)
    nominal_obstacle: np.ndarray  # Shape (n_rows, n_cols), dtype bool (physical footprint)
    uncertainty_zone: np.ndarray  # Shape (n_rows, n_cols), dtype bool (uncertainty expansion)
    active_obstacle_ids: List[int] = field(default_factory=list)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.cost.shape

    def copy(self) -> DynamicOccupancyGridSlice:
        """Deep copy of the raster slice."""
        return DynamicOccupancyGridSlice(
            time_offset=self.time_offset,
            timestamp=self.timestamp,
            cost=self.cost.copy(),
            risk_level=self.risk_level.copy(),
            nominal_obstacle=self.nominal_obstacle.copy(),
            uncertainty_zone=self.uncertainty_zone.copy(),
            active_obstacle_ids=list(self.active_obstacle_ids),
        )


class DynamicOccupancyLayer:
    """Spatiotemporal dynamic occupancy raster layer for dynamic collision checking."""

    def __init__(self, config: Optional[DynamicOccupancyConfig] = None) -> None:
        self.config = config or DynamicOccupancyConfig()

        self.resolution_m = float(self.config.resolution_m)
        self.x_bounds_m = (float(self.config.x_bounds_m[0]), float(self.config.x_bounds_m[1]))
        self.y_bounds_m = (float(self.config.y_bounds_m[0]), float(self.config.y_bounds_m[1]))
        self.frame_id = str(self.config.frame_id)
        self.time_bins = [float(t) for t in self.config.time_bins]

        # Raster grid dimensions (consistent with WorldTerrainMap in Phase 5/9)
        self.n_rows = int(np.round((self.x_bounds_m[1] - self.x_bounds_m[0]) / self.resolution_m))
        self.n_cols = int(np.round((self.y_bounds_m[1] - self.y_bounds_m[0]) / self.resolution_m))

        self.header_timestamp: float = 0.0
        self._slices: Dict[float, DynamicOccupancyGridSlice] = {}
        self._reset_slices(timestamp=0.0)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the 2D spatial grid (n_rows, n_cols)."""
        return (self.n_rows, self.n_cols)

    def _reset_slices(self, timestamp: float) -> None:
        """Initialize or reset all time slices to empty/free space."""
        self.header_timestamp = float(timestamp)
        self._slices = {}
        for t_offset in self.time_bins:
            self._slices[t_offset] = DynamicOccupancyGridSlice(
                time_offset=t_offset,
                timestamp=float(timestamp + t_offset),
                cost=np.zeros((self.n_rows, self.n_cols), dtype=np.float32),
                risk_level=np.zeros((self.n_rows, self.n_cols), dtype=np.uint8),
                nominal_obstacle=np.zeros((self.n_rows, self.n_cols), dtype=bool),
                uncertainty_zone=np.zeros((self.n_rows, self.n_cols), dtype=bool),
                active_obstacle_ids=[],
            )

    # -------------------------------------------------------------------------
    # Coordinate Transformation Methods (Identical to Phase 5/9 WorldTerrainMap)
    # -------------------------------------------------------------------------
    def world_to_grid(self, x_w: float, y_w: float) -> Tuple[int, int]:
        """Convert metric world coordinate (x, y) to raster indices (row, col)."""
        row = int(math.floor((self.x_bounds_m[1] - x_w) / self.resolution_m))
        col = int(math.floor((self.y_bounds_m[1] - y_w) / self.resolution_m))
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert raster indices (row, col) to metric cell center coordinate (x, y)."""
        x_w = self.x_bounds_m[1] - (row + 0.5) * self.resolution_m
        y_w = self.y_bounds_m[1] - (col + 0.5) * self.resolution_m
        return float(x_w), float(y_w)

    def is_in_bounds(self, x_w: float, y_w: float) -> bool:
        """Check if metric coordinate falls inside grid extent."""
        return (self.x_bounds_m[0] <= x_w < self.x_bounds_m[1]) and (
            self.y_bounds_m[0] <= y_w < self.y_bounds_m[1]
        )

    def is_grid_in_bounds(self, row: int, col: int) -> bool:
        """Check if raster indices fall inside grid tensor bounds."""
        return (0 <= row < self.n_rows) and (0 <= col < self.n_cols)

    # -------------------------------------------------------------------------
    # Ingestion / Rasterization Logic
    # -------------------------------------------------------------------------
    def update_predictions(
        self,
        predictions: Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]],
        header_timestamp: Optional[float] = None,
    ) -> None:
        """Ingest predicted obstacle trajectories and rasterize into spatiotemporal slices.

        Args:
            predictions: PredictedTrajectoryArray or list of PredictedObstacleTrajectory.
            header_timestamp: Base timestamp for predictions (default: from predictions).

        Raises:
            ValueError: If an EXPIRED track is encountered.
        """
        if isinstance(predictions, PredictedTrajectoryArray):
            traj_list = predictions.predictions
            ts = float(predictions.header_timestamp if header_timestamp is None else header_timestamp)
        else:
            traj_list = list(predictions)
            if header_timestamp is not None:
                ts = float(header_timestamp)
            elif traj_list:
                ts = float(traj_list[0].current_state.timestamp)
            else:
                ts = 0.0

        # Validate no EXPIRED tracks exist in input
        for pred in traj_list:
            if pred.current_state.status.upper() == "EXPIRED":
                raise ValueError(
                    f"Cannot ingest prediction for EXPIRED track (id={pred.track_id}). "
                    "Expired tracks must be pruned before occupancy rasterization."
                )

        # Reset all time slices to free space
        self._reset_slices(timestamp=ts)

        if not traj_list:
            return

        # Rasterize each obstacle trajectory across the defined time bins
        for pred in traj_list:
            self._rasterize_trajectory(pred)

    def _find_or_interpolate_waypoint(
        self,
        pred: PredictedObstacleTrajectory,
        target_offset: float,
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Extract or interpolate (x, y, vx, vy, r_unc, bbox_yaw) at target_offset.

        Returns:
            Tuple of (x, y, vx, vy, r_unc, bbox_yaw) or None if target_offset exceeds track horizon.
        """
        if target_offset < -1e-6 or target_offset > pred.horizon_s + 1e-4:
            return None

        # Check exact or nearest waypoint
        for wp in pred.waypoints:
            if abs(wp.time_offset - target_offset) < 1e-4:
                return (wp.x, wp.y, wp.vx, wp.vy, wp.uncertainty_radius, pred.current_state.bbox_yaw)

        # Interpolate between two enclosing waypoints
        if not pred.waypoints:
            return None

        if target_offset <= pred.waypoints[0].time_offset:
            wp0 = pred.waypoints[0]
            return (wp0.x, wp0.y, wp0.vx, wp0.vy, wp0.uncertainty_radius, pred.current_state.bbox_yaw)

        if target_offset >= pred.waypoints[-1].time_offset:
            wp1 = pred.waypoints[-1]
            return (wp1.x, wp1.y, wp1.vx, wp1.vy, wp1.uncertainty_radius, pred.current_state.bbox_yaw)

        for i in range(len(pred.waypoints) - 1):
            w0 = pred.waypoints[i]
            w1 = pred.waypoints[i + 1]
            if w0.time_offset <= target_offset <= w1.time_offset:
                span = w1.time_offset - w0.time_offset
                if span < 1e-6:
                    return (w0.x, w0.y, w0.vx, w0.vy, w0.uncertainty_radius, pred.current_state.bbox_yaw)
                ratio = (target_offset - w0.time_offset) / span
                ix = w0.x + ratio * (w1.x - w0.x)
                iy = w0.y + ratio * (w1.y - w0.y)
                ivx = w0.vx + ratio * (w1.vx - w0.vx)
                ivy = w0.vy + ratio * (w1.vy - w0.vy)
                ir_unc = w0.uncertainty_radius + ratio * (w1.uncertainty_radius - w0.uncertainty_radius)
                return (ix, iy, ivx, ivy, ir_unc, pred.current_state.bbox_yaw)

        return None

    def _rasterize_trajectory(self, pred: PredictedObstacleTrajectory) -> None:
        """Rasterize a single obstacle's trajectory into all active time slices."""
        track_id = pred.track_id
        bbox_size = pred.current_state.bbox_size  # (length, width)
        length_m, width_m = float(bbox_size[0]), float(bbox_size[1])

        half_l = length_m / 2.0
        half_w = width_m / 2.0

        for t_bin in self.time_bins:
            wp_data = self._find_or_interpolate_waypoint(pred, t_bin)
            if wp_data is None:
                # Target time exceeds obstacle prediction horizon (e.g. LOST track capped at 1.0s)
                continue

            px, py, vx, vy, r_unc, yaw = wp_data
            grid_slice = self._slices[t_bin]
            grid_slice.active_obstacle_ids.append(track_id)

            # Determine local bounding box of expanded region to minimize raster computation
            total_margin = float(r_unc)
            corners = compute_obb_corners((px, py), bbox_size, yaw, margin_m=total_margin)
            min_x_m = np.min(corners[:, 0])
            max_x_m = np.max(corners[:, 0])
            min_y_m = np.min(corners[:, 1])
            max_y_m = np.max(corners[:, 1])

            # Convert to grid index bounds
            # Recall: row increases as x decreases: row_min corresponds to max_x
            r0 = int(math.floor((self.x_bounds_m[1] - max_x_m) / self.resolution_m))
            r1 = int(math.ceil((self.x_bounds_m[1] - min_x_m) / self.resolution_m))
            c0 = int(math.floor((self.y_bounds_m[1] - max_y_m) / self.resolution_m))
            c1 = int(math.ceil((self.y_bounds_m[1] - min_y_m) / self.resolution_m))

            # Clamp to grid limits
            r_start = max(0, r0)
            r_end = min(self.n_rows, r1 + 1)
            c_start = max(0, c0)
            c_end = min(self.n_cols, c1 + 1)

            if r_start >= r_end or c_start >= c_end:
                # Obstacle is completely outside the grid map boundary
                continue

            # Generate metric coordinates for sub-window cells
            rows = np.arange(r_start, r_end, dtype=np.float64)
            cols = np.arange(c_start, c_end, dtype=np.float64)
            cols_grid, rows_grid = np.meshgrid(cols, rows)

            cell_xs = self.x_bounds_m[1] - (rows_grid + 0.5) * self.resolution_m
            cell_ys = self.y_bounds_m[1] - (cols_grid + 0.5) * self.resolution_m

            # Transform world points into obstacle-centric local frame
            dx = cell_xs - px
            dy = cell_ys - py
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy

            # Distance to the nominal physical bounding box
            d_x = np.maximum(0.0, np.abs(local_x) - half_l)
            d_y = np.maximum(0.0, np.abs(local_y) - half_w)
            dist_to_nominal = np.sqrt(d_x ** 2 + d_y ** 2)

            # 1. Nominal footprint (strictly LETHAL, cost = 1.0)
            is_nominal = dist_to_nominal <= 1e-5

            # 2. Graded uncertainty region
            sigma_pos = max(1e-4, r_unc / self.config.sigma_scale)
            in_uncertainty = (dist_to_nominal > 1e-5) & (dist_to_nominal <= r_unc)

            # Initialize local cost and risk arrays
            local_cost = np.zeros_like(dist_to_nominal, dtype=np.float32)
            local_risk = np.zeros_like(dist_to_nominal, dtype=np.uint8)

            # Assign nominal cells
            local_cost[is_nominal] = 1.0
            local_risk[is_nominal] = DynamicRiskLevel.LETHAL

            # Assign graded uncertainty cells based on metric distance from boundary
            # High-risk zone: [0, 1 * sigma_pos] -> cost in [0.70, 0.95)
            mask_high = in_uncertainty & (dist_to_nominal <= sigma_pos)
            if np.any(mask_high):
                norm_d = dist_to_nominal[mask_high] / sigma_pos
                local_cost[mask_high] = (0.95 - 0.25 * norm_d).astype(np.float32)
                local_risk[mask_high] = DynamicRiskLevel.HIGH_RISK

            # Medium-risk zone: (1 * sigma_pos, 2 * sigma_pos] -> cost in [0.40, 0.70)
            mask_med = in_uncertainty & (dist_to_nominal > sigma_pos) & (dist_to_nominal <= 2.0 * sigma_pos)
            if np.any(mask_med):
                norm_d = (dist_to_nominal[mask_med] - sigma_pos) / sigma_pos
                local_cost[mask_med] = (0.70 - 0.30 * norm_d).astype(np.float32)
                local_risk[mask_med] = DynamicRiskLevel.MEDIUM_RISK

            # Low-risk zone: (2 * sigma_pos, 3 * sigma_pos] -> cost in [0.15, 0.40)
            mask_low = in_uncertainty & (dist_to_nominal > 2.0 * sigma_pos) & (dist_to_nominal <= 3.0 * sigma_pos)
            if np.any(mask_low):
                norm_d = (dist_to_nominal[mask_low] - 2.0 * sigma_pos) / sigma_pos
                local_cost[mask_low] = (0.40 - 0.25 * norm_d).astype(np.float32)
                local_risk[mask_low] = DynamicRiskLevel.LOW_RISK

            # Multi-obstacle conservative safety fusion (max cost and max risk precedence)
            target_cost_sub = grid_slice.cost[r_start:r_end, c_start:c_end]
            target_risk_sub = grid_slice.risk_level[r_start:r_end, c_start:c_end]
            target_nom_sub = grid_slice.nominal_obstacle[r_start:r_end, c_start:c_end]
            target_unc_sub = grid_slice.uncertainty_zone[r_start:r_end, c_start:c_end]

            np.maximum(target_cost_sub, local_cost, out=target_cost_sub)
            np.maximum(target_risk_sub, local_risk, out=target_risk_sub)
            target_nom_sub |= is_nominal
            target_unc_sub |= in_uncertainty

    # -------------------------------------------------------------------------
    # Query API
    # -------------------------------------------------------------------------
    def get_time_bins(self) -> List[float]:
        """Return the sequence of discrete relative prediction time bins."""
        return list(self.time_bins)

    def get_slice(
        self,
        time_offset: float,
        conservative_union: bool = False,
    ) -> DynamicOccupancyGridSlice:
        """Retrieve dynamic occupancy raster slice for a given future time offset.

        Args:
            time_offset: Future time delta in seconds (must be non-negative).
            conservative_union: If True and time_offset falls strictly between two bins,
                returns the conservative element-wise maximum of the two enclosing slices.
                If False (default), returns the slice corresponding to the nearest time bin.

        Returns:
            DynamicOccupancyGridSlice at or bounding the requested future time offset.

        Raises:
            ValueError: If time_offset is negative or non-finite.
        """
        f_t = float(time_offset)
        if not math.isfinite(f_t):
            raise ValueError(f"time_offset must be finite, got {time_offset}")
        if f_t < -1e-6:
            raise ValueError(f"time_offset cannot be negative, got {time_offset}")

        f_t = max(0.0, f_t)

        # Out-of-horizon policy: beyond max_horizon_s, dynamic occupancy expires to free space
        if f_t > self.config.max_horizon_s + 1e-4:
            return DynamicOccupancyGridSlice(
                time_offset=f_t,
                timestamp=self.header_timestamp + f_t,
                cost=np.zeros((self.n_rows, self.n_cols), dtype=np.float32),
                risk_level=np.zeros((self.n_rows, self.n_cols), dtype=np.uint8),
                nominal_obstacle=np.zeros((self.n_rows, self.n_cols), dtype=bool),
                uncertainty_zone=np.zeros((self.n_rows, self.n_cols), dtype=bool),
                active_obstacle_ids=[],
            )

        # Check exact match with configured bins
        for b in self.time_bins:
            if abs(b - f_t) < 1e-4:
                return self._slices[b]

        # Find enclosing time bins [t0, t1]
        t0 = self.time_bins[0]
        t1 = self.time_bins[-1]
        for i in range(len(self.time_bins) - 1):
            if self.time_bins[i] <= f_t <= self.time_bins[i + 1]:
                t0 = self.time_bins[i]
                t1 = self.time_bins[i + 1]
                break

        slice0 = self._slices[t0]
        slice1 = self._slices[t1]

        if not conservative_union:
            # Nearest bin lookup
            if abs(f_t - t0) <= abs(f_t - t1):
                return slice0
            return slice1

        # Conservative union: element-wise maximum over enclosing bins
        merged_cost = np.maximum(slice0.cost, slice1.cost)
        merged_risk = np.maximum(slice0.risk_level, slice1.risk_level)
        merged_nom = slice0.nominal_obstacle | slice1.nominal_obstacle
        merged_unc = slice0.uncertainty_zone | slice1.uncertainty_zone
        merged_ids = sorted(list(set(slice0.active_obstacle_ids + slice1.active_obstacle_ids)))

        return DynamicOccupancyGridSlice(
            time_offset=f_t,
            timestamp=self.header_timestamp + f_t,
            cost=merged_cost,
            risk_level=merged_risk,
            nominal_obstacle=merged_nom,
            uncertainty_zone=merged_unc,
            active_obstacle_ids=merged_ids,
        )

    def get_cost(
        self,
        x_w: float,
        y_w: float,
        time_offset: float,
        conservative_union: bool = False,
    ) -> float:
        """Query continuous dynamic obstacle cost [0.0, 1.0] at world coordinate (x, y) and future time."""
        if not self.is_in_bounds(x_w, y_w):
            return 0.0
        r, c = self.world_to_grid(x_w, y_w)
        if not self.is_grid_in_bounds(r, c):
            return 0.0
        grid_slice = self.get_slice(time_offset, conservative_union=conservative_union)
        return float(grid_slice.cost[r, c])

    def get_risk_level(
        self,
        x_w: float,
        y_w: float,
        time_offset: float,
        conservative_union: bool = False,
    ) -> DynamicRiskLevel:
        """Query categorical dynamic risk level at world coordinate (x, y) and future time."""
        if not self.is_in_bounds(x_w, y_w):
            return DynamicRiskLevel.FREE
        r, c = self.world_to_grid(x_w, y_w)
        if not self.is_grid_in_bounds(r, c):
            return DynamicRiskLevel.FREE
        grid_slice = self.get_slice(time_offset, conservative_union=conservative_union)
        return DynamicRiskLevel(int(grid_slice.risk_level[r, c]))

    def is_occupied(
        self,
        x_w: float,
        y_w: float,
        time_offset: float,
        cost_threshold: Optional[float] = None,
        conservative_union: bool = False,
    ) -> bool:
        """Evaluate whether cell is considered dynamically occupied at future time."""
        thresh = float(self.config.collision_cost_threshold if cost_threshold is None else cost_threshold)
        return self.get_cost(x_w, y_w, time_offset, conservative_union=conservative_union) >= thresh

    def is_lethal(
        self,
        x_w: float,
        y_w: float,
        time_offset: float,
        conservative_union: bool = False,
    ) -> bool:
        """Evaluate whether coordinate falls inside physical obstacle footprint at future time."""
        if not self.is_in_bounds(x_w, y_w):
            return False
        r, c = self.world_to_grid(x_w, y_w)
        if not self.is_grid_in_bounds(r, c):
            return False
        grid_slice = self.get_slice(time_offset, conservative_union=conservative_union)
        return bool(grid_slice.nominal_obstacle[r, c] or grid_slice.cost[r, c] >= 0.99)

    # -------------------------------------------------------------------------
    # Diagnostics & Blockage Reporting
    # -------------------------------------------------------------------------
    def get_slice_statistics(self, time_offset: float) -> Dict[str, Any]:
        """Compute spatial occupancy and blockage statistics for a given time slice."""
        grid_slice = self.get_slice(time_offset)
        total_cells = self.n_rows * self.n_cols
        total_area_m2 = total_cells * (self.resolution_m ** 2)

        nom_count = int(np.count_nonzero(grid_slice.nominal_obstacle))
        unc_count = int(np.count_nonzero(grid_slice.uncertainty_zone))
        lethal_count = int(np.count_nonzero(grid_slice.risk_level == DynamicRiskLevel.LETHAL))
        high_risk_count = int(np.count_nonzero(grid_slice.risk_level == DynamicRiskLevel.HIGH_RISK))
        med_risk_count = int(np.count_nonzero(grid_slice.risk_level == DynamicRiskLevel.MEDIUM_RISK))
        low_risk_count = int(np.count_nonzero(grid_slice.risk_level == DynamicRiskLevel.LOW_RISK))
        affected_count = int(np.count_nonzero(grid_slice.cost > 0.0))

        nom_area_m2 = nom_count * (self.resolution_m ** 2)
        unc_area_m2 = unc_count * (self.resolution_m ** 2)
        affected_area_m2 = affected_count * (self.resolution_m ** 2)
        affected_fraction = (affected_count / total_cells) * 100.0 if total_cells > 0 else 0.0

        return {
            "time_offset": float(time_offset),
            "total_cells": total_cells,
            "total_area_m2": total_area_m2,
            "nominal_cells": nom_count,
            "nominal_area_m2": nom_area_m2,
            "uncertainty_cells": unc_count,
            "uncertainty_area_m2": unc_area_m2,
            "lethal_cells": lethal_count,
            "high_risk_cells": high_risk_count,
            "medium_risk_cells": med_risk_count,
            "low_risk_cells": low_risk_count,
            "total_affected_cells": affected_count,
            "total_affected_area_m2": affected_area_m2,
            "affected_grid_percent": affected_fraction,
            "active_obstacle_count": len(grid_slice.active_obstacle_ids),
        }
