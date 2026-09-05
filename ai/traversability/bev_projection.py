"""Bird's-Eye-View (BEV) projection projecting image-space traversability to a robot-centric grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from ai.traversability.base import TraversabilityResult
from ai.traversability.depth_source import DepthSource, PlanarGroundDepthSource
from ai.traversability.geometry import CameraExtrinsics, CameraIntrinsics


@dataclass
class BEVTraversabilityMap:
    """Robot-centric 2D Bird's-Eye-View occupancy/traversability grid.

    Coordinate convention:
      - X forward: Distance ahead of vehicle (meters).
      - Y lateral: Left (+) / Right (-) relative to vehicle centerline (meters).
      - Grid origin: Top-left of array maps to (max_x, max_y).
    """

    grid: np.ndarray  # (H_grid, W_grid) float32 in [0.0, 1.0], 1.0 = SAFE/TRAVERSABLE
    unknown_mask: np.ndarray  # (H_grid, W_grid) bool, True = unobserved/no ray hits
    resolution_m: float  # Grid cell physical dimension (meters/cell)
    x_range_m: Tuple[float, float]  # (min_x, max_x) forward bounds
    y_range_m: Tuple[float, float]  # (min_y, max_y) lateral bounds

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid.shape

    def meter_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert robot metric coordinate (x_forward, y_left) to (row, col) indices."""
        x_min, x_max = self.x_range_m
        y_min, y_max = self.y_range_m
        row = int((x_max - x_m) / self.resolution_m)
        col = int((y_max - y_m) / self.resolution_m)
        return row, col

    def grid_to_meter(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid (row, col) back to metric coordinates (x_forward, y_left)."""
        x_min, x_max = self.x_range_m
        y_min, y_max = self.y_range_m
        x_m = x_max - (row + 0.5) * self.resolution_m
        y_m = y_max - (col + 0.5) * self.resolution_m
        return float(x_m), float(y_m)


class BEVProjector:
    """Projects image-space traversability probability masks into a local BEV occupancy grid."""

    def __init__(
        self,
        intrinsics: Optional[CameraIntrinsics] = None,
        extrinsics: Optional[CameraExtrinsics] = None,
        depth_source: Optional[DepthSource] = None,
        resolution_m: float = 0.05,
        x_range_m: Tuple[float, float] = (0.2, 5.0),
        y_range_m: Tuple[float, float] = (-2.5, 2.5),
        pooling: str = "mean"
    ) -> None:
        """
        Args:
            intrinsics: Camera focal length and optical centers.
            extrinsics: Mounting height and downward pitch angle.
            depth_source: Optional custom depth source (defaults to analytical ground plane).
            resolution_m: Cell size in meters (default: 0.05m = 5cm/cell).
            x_range_m: Forward coverage [min_x, max_x] meters.
            y_range_m: Lateral coverage [min_y, max_y] meters.
            pooling: Cell aggregation method ('mean', 'min', 'max').
        """
        self.intrinsics = intrinsics or CameraIntrinsics.default_640x480()
        self.extrinsics = extrinsics or CameraExtrinsics(pitch_deg=18.0, height_m=0.35)
        self.depth_source = depth_source or PlanarGroundDepthSource(self.intrinsics, self.extrinsics)
        self.resolution_m = float(resolution_m)
        self.x_range_m = x_range_m
        self.y_range_m = y_range_m
        self.pooling = pooling

        # Grid dimensions in cells
        self.n_x = int(np.ceil((x_range_m[1] - x_range_m[0]) / self.resolution_m))
        self.n_y = int(np.ceil((y_range_m[1] - y_range_m[0]) / self.resolution_m))

    def project(
        self,
        trav_result: TraversabilityResult | np.ndarray,
        rgb_frame: Optional[np.ndarray] = None
    ) -> BEVTraversabilityMap:
        """Project image-space traversability into robot-centric BEV grid.

        Args:
            trav_result: Either a TraversabilityResult or a 2D float probability mask.
            rgb_frame: Optional RGB frame if trav_result is raw numpy mask.

        Returns:
            BEVTraversabilityMap populated with drivability scores.
        """
        if isinstance(trav_result, TraversabilityResult):
            mask = trav_result.mask
            rgb = trav_result.image
        else:
            mask = np.asarray(trav_result, dtype=np.float32)
            rgb = rgb_frame

        h, w = mask.shape
        depth_m = self.depth_source.get_depth_meters((h, w), rgb_image=rgb)

        # Generate pixel coordinates grid
        v_indices, u_indices = np.mgrid[0:h, 0:w]

        # Valid depth mask
        valid = ~np.isnan(depth_m) & (depth_m > 0.0)
        u_val = u_indices[valid].astype(np.float32)
        v_val = v_indices[valid].astype(np.float32)
        z_val = depth_m[valid].astype(np.float32)
        scores = mask[valid].astype(np.float32)

        # 3D points in camera optical frame
        x_c = (u_val - self.intrinsics.cx) * z_val / self.intrinsics.fx
        y_c = (v_val - self.intrinsics.cy) * z_val / self.intrinsics.fy
        z_c = z_val
        pts_cam = np.stack([x_c, y_c, z_c], axis=0)  # (3, N)

        # Transform from camera optical frame to vehicle base_link ground frame
        # P_robot = R_opt_to_robot @ P_cam + T_robot
        R_robot_to_opt = self.extrinsics.rotation_matrix
        R_opt_to_robot = R_robot_to_opt.T

        t_robot = np.array([[self.extrinsics.forward_offset_m], [0.0], [self.extrinsics.height_m]], dtype=np.float32)
        pts_robot = R_opt_to_robot @ pts_cam + t_robot  # (3, N)

        x_robot = pts_robot[0, :]
        y_robot = pts_robot[1, :]

        # Filter within BEV bounding box
        x_min, x_max = self.x_range_m
        y_min, y_max = self.y_range_m
        in_bounds = (x_robot >= x_min) & (x_robot < x_max) & (y_robot >= y_min) & (y_robot < y_max)

        x_in = x_robot[in_bounds]
        y_in = y_robot[in_bounds]
        scores_in = scores[in_bounds]

        # Calculate grid cell coordinates
        row_indices = ((x_max - x_in) / self.resolution_m).astype(np.int32)
        col_indices = ((y_max - y_in) / self.resolution_m).astype(np.int32)

        # Clip for safety
        row_indices = np.clip(row_indices, 0, self.n_x - 1)
        col_indices = np.clip(col_indices, 0, self.n_y - 1)

        # Accumulator grids
        grid_sum = np.zeros((self.n_x, self.n_y), dtype=np.float32)
        grid_count = np.zeros((self.n_x, self.n_y), dtype=np.int32)
        grid_min = np.full((self.n_x, self.n_y), 1.0, dtype=np.float32)

        for r, c, s in zip(row_indices, col_indices, scores_in):
            grid_sum[r, c] += s
            grid_count[r, c] += 1
            if s < grid_min[r, c]:
                grid_min[r, c] = s

        unknown_mask = (grid_count == 0)

        if self.pooling == "min":
            bev_grid = np.where(unknown_mask, 0.5, grid_min)
        else:  # mean pooling
            bev_grid = np.where(unknown_mask, 0.5, grid_sum / np.maximum(grid_count, 1))

        return BEVTraversabilityMap(
            grid=bev_grid.astype(np.float32),
            unknown_mask=unknown_mask,
            resolution_m=self.resolution_m,
            x_range_m=self.x_range_m,
            y_range_m=self.y_range_m
        )
