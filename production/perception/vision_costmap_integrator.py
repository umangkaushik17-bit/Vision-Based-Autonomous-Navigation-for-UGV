"""End-to-End Vision-in-the-Loop: Metric Ground-Plane Projection and Costmap Fusion.

Converts live camera traversability perception outputs into a metric robot-centric
Bird's-Eye-View (BEV) costmap and fuses it into the world terrain costmap consumed
by the A* replanner and native C++ DWA local trajectory generator.

Enforces strict safety invariants:
  - UNKNOWN != SAFE
  - Missing/blackout frame != free space
  - Camera timeout != free space
  - Invalid calibration != free space
  - Zero collision false negatives on bounded obstacles
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ai.spatial_fusion.world_map import WorldTerrainMap
from ai.traversability.base import TraversabilityResult
from ai.traversability.geometry import CameraExtrinsics, CameraIntrinsics


@dataclass
class VisionCostmapConfig:
    """Configuration for metric projection and costmap fusion."""

    # Metric camera geometry
    intrinsics: CameraIntrinsics = field(
        default_factory=lambda: CameraIntrinsics(
            fx=381.3614,
            fy=381.3614,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )
    )
    extrinsics: CameraExtrinsics = field(
        default_factory=lambda: CameraExtrinsics(
            pitch_deg=15.0,
            roll_deg=0.0,
            yaw_deg=0.0,
            height_m=0.525,
            forward_offset_m=0.15,
        )
    )

    # Local BEV metric grid coverage
    resolution_m: float = 0.10             # Grid cell dimension (10 cm/cell)
    x_range_m: Tuple[float, float] = (0.20, 6.0)  # Forward bounds [min_x, max_x]
    y_range_m: Tuple[float, float] = (-3.0, 3.0)  # Lateral bounds [min_y, max_y]

    # Cost mapping thresholds
    safe_prob_thresh: float = 0.50          # tau >= 0.50 is safe path
    obstacle_prob_thresh: float = 0.30      # tau < 0.30 is lethal obstacle
    unknown_cost_default: float = 0.50      # UNKNOWN != SAFE

    # Obstacle inflation
    obstacle_inflation_radius_m: float = 0.20  # Inflation buffer around obstacles
    inflation_cost: float = 0.75               # Cost assigned to inflated buffer cells

    # Temporal safety & persistence
    max_frame_age_s: float = 0.50              # Timeout after which vision is stale
    persistent_hazard_latch: bool = True       # Latches detected obstacles against frame drops
    min_clear_observations_to_erase: int = 5   # Consecutive clear frames needed to unmark an obstacle


@dataclass
class LocalBEVCostmap:
    """Metric robot-centric Bird's-Eye-View costmap."""

    cost: np.ndarray             # (nx, ny) float32 in [0.0, 1.0], 1.0 = lethal
    traversability: np.ndarray   # (nx, ny) float32 in [0.0, 1.0], 1.0 = safe
    obstacle: np.ndarray         # (nx, ny) bool, True = lethal obstacle
    hazard: np.ndarray           # (nx, ny) float32 in [0.0, 1.0]
    unknown: np.ndarray          # (nx, ny) bool, True = unobserved
    confidence: np.ndarray       # (nx, ny) float32 in [0.0, 1.0]

    # Metric coordinates of cell centers in robot frame
    x_robot: np.ndarray          # (nx, ny) forward coordinate (meters)
    y_robot: np.ndarray          # (nx, ny) lateral coordinate (meters)

    resolution_m: float
    x_range_m: Tuple[float, float]
    y_range_m: Tuple[float, float]
    timestamp: float = 0.0
    frame_id: str = "base_link"
    valid: bool = True
    status_message: str = "OK"

    @property
    def shape(self) -> Tuple[int, int]:
        return self.cost.shape

    def meter_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert robot metric coordinate (x_forward, y_lateral) to (row, col)."""
        x_min, x_max = self.x_range_m
        y_min, y_max = self.y_range_m
        r = int((x_max - x_m) / self.resolution_m)
        c = int((y_max - y_m) / self.resolution_m)
        return r, c

    def grid_to_meter(self, r: int, c: int) -> Tuple[float, float]:
        """Convert grid (row, col) to robot metric coordinate (x_forward, y_lateral)."""
        x_min, x_max = self.x_range_m
        y_min, y_max = self.y_range_m
        x_m = x_max - (r + 0.5) * self.resolution_m
        y_m = y_max - (c + 0.5) * self.resolution_m
        return float(x_m), float(y_m)


class VisionCostmapIntegrator:
    """Precomputes ground-plane IPM ray mapping, converts perception results to local BEV,

    and updates the world terrain map with persistent obstacle tracking.
    """

    def __init__(self, config: Optional[VisionCostmapConfig] = None) -> None:
        self.config = config or VisionCostmapConfig()

        # Grid dimensions in cells
        self.nx = int(np.ceil((self.config.x_range_m[1] - self.config.x_range_m[0]) / self.config.resolution_m))
        self.ny = int(np.ceil((self.config.y_range_m[1] - self.config.y_range_m[0]) / self.config.resolution_m))

        # Metric cell centers in robot frame
        row_indices, col_indices = np.mgrid[0:self.nx, 0:self.ny]
        x_max = self.config.x_range_m[1]
        y_max = self.config.y_range_m[1]
        self.grid_x_m = (x_max - (row_indices + 0.5) * self.config.resolution_m).astype(np.float32)
        self.grid_y_m = (y_max - (col_indices + 0.5) * self.config.resolution_m).astype(np.float32)

        # Precomputed ray-projection lookup tables (LUT)
        self._lut_shape: Optional[Tuple[int, int]] = None
        self._lut_valid_v: Optional[np.ndarray] = None
        self._lut_valid_u: Optional[np.ndarray] = None
        self._lut_rows: Optional[np.ndarray] = None
        self._lut_cols: Optional[np.ndarray] = None

        # Precompute for default 240x320 model output resolution
        self.precompute_projection((240, 320))

        # Persistent obstacle tracking in world map: (row, col) -> consecutive clear count
        self._obstacle_clear_counter: Dict[Tuple[int, int], int] = {}
        self.last_update_timestamp: float = 0.0

    @classmethod
    def create_simulation_default(cls) -> VisionCostmapIntegrator:
        """Create integrator calibrated for Gazebo demo_outdoor simulation."""
        cfg = VisionCostmapConfig(
            intrinsics=CameraIntrinsics(
                fx=381.3614,
                fy=381.3614,
                cx=320.0,
                cy=240.0,
                width=640,
                height=480,
            ),
            extrinsics=CameraExtrinsics(
                pitch_deg=15.0,
                roll_deg=0.0,
                yaw_deg=0.0,
                height_m=0.525,
                forward_offset_m=0.15,
            ),
            resolution_m=0.10,
            x_range_m=(0.20, 6.0),
            y_range_m=(-3.0, 3.0),
            safe_prob_thresh=0.50,
            obstacle_prob_thresh=0.30,
            obstacle_inflation_radius_m=0.20,
            inflation_cost=0.75,
            unknown_cost_default=0.50,
        )
        return cls(cfg)

    @classmethod
    def create_physical_default(
        cls,
        pitch_deg: float = 18.0,
        height_m: float = 0.45,
        forward_offset_m: float = 0.15,
        hfov_deg: float = 69.4,
    ) -> VisionCostmapIntegrator:
        """Create integrator calibrated for physical UGV camera."""
        intrinsics = CameraIntrinsics.default_640x480(fov_deg=hfov_deg)
        extrinsics = CameraExtrinsics(
            pitch_deg=pitch_deg,
            height_m=height_m,
            forward_offset_m=forward_offset_m,
        )
        cfg = VisionCostmapConfig(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            resolution_m=0.10,
            x_range_m=(0.20, 6.0),
            y_range_m=(-3.0, 3.0),
        )
        return cls(cfg)

    def precompute_projection(self, image_shape: Tuple[int, int]) -> None:
        """Precomputes analytical ground-plane intersection rays for given input shape."""
        h, w = image_shape
        intr = self.config.intrinsics
        extr = self.config.extrinsics

        # Scale intrinsics to image_shape if model output is lower resolution
        scale_x = w / float(intr.width)
        scale_y = h / float(intr.height)
        fx_scaled = intr.fx * scale_x
        fy_scaled = intr.fy * scale_y
        cx_scaled = intr.cx * scale_x
        cy_scaled = intr.cy * scale_y

        v_indices, u_indices = np.mgrid[0:h, 0:w]

        # Analytical ground-plane ray intersection:
        # Ground plane is Z_robot = 0
        pitch_rad = np.radians(extr.pitch_deg)
        height = extr.height_m

        ray_angle = (v_indices - cy_scaled) / fy_scaled
        denom = np.sin(pitch_rad) + ray_angle * np.cos(pitch_rad)

        valid_ground = denom > 1e-4
        depth_m = np.full((h, w), np.nan, dtype=np.float32)
        depth_m[valid_ground] = height / denom[valid_ground]
        depth_m[depth_m <= 0.0] = np.nan
        depth_m[depth_m > 12.0] = np.nan  # Cap max optical depth

        valid = ~np.isnan(depth_m) & (depth_m > 0.0)
        u_val = u_indices[valid].astype(np.float32)
        v_val = v_indices[valid].astype(np.float32)
        z_val = depth_m[valid].astype(np.float32)

        # 3D points in optical frame
        x_c = (u_val - cx_scaled) * z_val / fx_scaled
        y_c = (v_val - cy_scaled) * z_val / fy_scaled
        z_c = z_val
        pts_cam = np.stack([x_c, y_c, z_c], axis=0)  # (3, N)

        # Transform to robot base_link frame
        R_opt_to_robot = extr.rotation_matrix.T
        t_robot = np.array([[extr.forward_offset_m], [0.0], [extr.height_m]], dtype=np.float32)
        pts_robot = R_opt_to_robot @ pts_cam + t_robot  # (3, N)

        x_r = pts_robot[0, :]
        y_r = pts_robot[1, :]

        # Filter within defined BEV metric bounds
        x_min, x_max = self.config.x_range_m
        y_min, y_max = self.config.y_range_m
        in_bounds = (x_r >= x_min) & (x_r < x_max) & (y_r >= y_min) & (y_r < y_max)

        valid_u_in = u_indices[valid][in_bounds]
        valid_v_in = v_indices[valid][in_bounds]
        x_in = x_r[in_bounds]
        y_in = y_r[in_bounds]

        # Metric to grid index mapping
        res = self.config.resolution_m
        rows = np.clip(((x_max - x_in) / res).astype(np.int32), 0, self.nx - 1)
        cols = np.clip(((y_max - y_in) / res).astype(np.int32), 0, self.ny - 1)

        self._lut_shape = (h, w)
        self._lut_valid_v = valid_v_in
        self._lut_valid_u = valid_u_in
        self._lut_rows = rows
        self._lut_cols = cols

    def project_traversability(
        self,
        trav_result: TraversabilityResult,
        current_time_s: Optional[float] = None,
    ) -> LocalBEVCostmap:
        """Projects a TraversabilityResult into a local metric BEV costmap.

        Enforces fail-safe degradation on missing/invalid/stale inputs.
        """
        now_s = current_time_s if current_time_s is not None else time.time()

        # 1. Validate Input
        if trav_result is None or not trav_result.valid:
            # UNKNOWN != SAFE: return all-unknown conservative costmap
            return self._build_empty_costmap(
                timestamp=now_s,
                valid=False,
                status_message="INVALID_PERCEPTION_RESULT: Frame rejected by safety checks",
            )

        # Check frame freshness
        if trav_result.timestamp > 0.0 and (now_s - trav_result.timestamp) > self.config.max_frame_age_s:
            return self._build_empty_costmap(
                timestamp=now_s,
                valid=False,
                status_message=f"PERCEPTION_TIMEOUT: Frame age {now_s - trav_result.timestamp:.3f}s exceeds {self.config.max_frame_age_s}s",
            )

        mask = np.asarray(trav_result.mask, dtype=np.float32)
        h, w = mask.shape
        if self._lut_shape != (h, w):
            self.precompute_projection((h, w))

        # 2. Extract valid ground ray scores
        scores_in = mask[self._lut_valid_v, self._lut_valid_u]

        # Vectorized accumulator
        grid_sum = np.zeros((self.nx, self.ny), dtype=np.float32)
        grid_count = np.zeros((self.nx, self.ny), dtype=np.int32)
        grid_min = np.full((self.nx, self.ny), 1.0, dtype=np.float32)

        # Use conservative min-pooling for safety (any obstacle observation dominates)
        np.add.at(grid_sum, (self._lut_rows, self._lut_cols), scores_in)
        np.add.at(grid_count, (self._lut_rows, self._lut_cols), 1)

        # Compute minimum score per cell via bincount or grouped min
        flat_idx = self._lut_rows * self.ny + self._lut_cols
        total_cells = self.nx * self.ny

        # Vectorized minimum score per cell
        # Initialize flat minimum array
        flat_min = np.full(total_cells, 1.0, dtype=np.float32)
        np.minimum.at(flat_min, flat_idx, scores_in)
        grid_min = flat_min.reshape((self.nx, self.ny))

        observed_mask = grid_count > 0
        unknown_mask = ~observed_mask

        # Drivability score: use conservative min-pooling
        mean_trav = np.zeros((self.nx, self.ny), dtype=np.float32)
        mean_trav[observed_mask] = grid_sum[observed_mask] / grid_count[observed_mask]
        trav_score = np.where(observed_mask, grid_min, 0.0)

        # 3. Cost Mapping
        # High traversability (tau >= safe_thresh): low cost in [0.0, 0.20]
        # Risky/rough (obstacle_thresh <= tau < safe_thresh): medium cost in [0.40, 0.70]
        # Obstacle (tau < obstacle_thresh): lethal cost 1.0
        # Unknown (no rays): unknown_cost_default (0.50), unknown = True
        cost = np.full((self.nx, self.ny), self.config.unknown_cost_default, dtype=np.float32)
        obstacle = np.zeros((self.nx, self.ny), dtype=bool)
        hazard = np.zeros((self.nx, self.ny), dtype=np.float32)
        confidence = np.zeros((self.nx, self.ny), dtype=np.float32)

        if np.any(observed_mask):
            obs_scores = trav_score[observed_mask]

            # Lethal obstacles
            is_lethal = obs_scores < self.config.obstacle_prob_thresh
            is_safe = obs_scores >= self.config.safe_prob_thresh
            is_risky = (~is_lethal) & (~is_safe)

            obs_cost = np.zeros_like(obs_scores, dtype=np.float32)
            obs_hazard = np.zeros_like(obs_scores, dtype=np.float32)

            # Safe paths
            obs_cost[is_safe] = (1.0 - obs_scores[is_safe]) * 0.40  # e.g., tau=0.9 -> 0.04
            obs_hazard[is_safe] = 0.0

            # Risky paths
            obs_cost[is_risky] = 0.40 + (self.config.safe_prob_thresh - obs_scores[is_risky]) * 1.50
            obs_hazard[is_risky] = 0.35 + (self.config.safe_prob_thresh - obs_scores[is_risky]) * 1.50

            # Lethal obstacles
            obs_cost[is_lethal] = 1.0
            obs_hazard[is_lethal] = 1.0

            cost[observed_mask] = obs_cost
            hazard[observed_mask] = obs_hazard
            obstacle[observed_mask] = is_lethal
            confidence[observed_mask] = np.clip(grid_count[observed_mask] / 10.0, 0.2, 1.0)

        # 4. Obstacle Inflation Buffer
        if self.config.obstacle_inflation_radius_m > 0 and np.any(obstacle):
            r_cells = int(np.ceil(self.config.obstacle_inflation_radius_m / self.config.resolution_m))
            inflated_cost = cost.copy()
            obs_rows, obs_cols = np.where(obstacle)
            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    if dr == 0 and dc == 0:
                        continue
                    if dr * dr + dc * dc <= r_cells * r_cells:
                        nr = obs_rows + dr
                        nc = obs_cols + dc
                        valid_nbr = (nr >= 0) & (nr < self.nx) & (nc >= 0) & (nc < self.ny)
                        nr_val = nr[valid_nbr]
                        nc_val = nc[valid_nbr]
                        # Set inflation cost on non-obstacle cells
                        non_obs = ~obstacle[nr_val, nc_val]
                        inflated_cost[nr_val[non_obs], nc_val[non_obs]] = np.maximum(
                            inflated_cost[nr_val[non_obs], nc_val[non_obs]],
                            self.config.inflation_cost
                        )
            cost = inflated_cost

        return LocalBEVCostmap(
            cost=cost,
            traversability=trav_score,
            obstacle=obstacle,
            hazard=hazard,
            unknown=unknown_mask,
            confidence=confidence,
            x_robot=self.grid_x_m,
            y_robot=self.grid_y_m,
            resolution_m=self.config.resolution_m,
            x_range_m=self.config.x_range_m,
            y_range_m=self.config.y_range_m,
            timestamp=trav_result.timestamp or now_s,
            frame_id="base_link",
            valid=True,
            status_message="OK",
        )

    def _build_empty_costmap(self, timestamp: float, valid: bool, status_message: str) -> LocalBEVCostmap:
        """Constructs an all-unknown conservative costmap when perception is unavailable."""
        return LocalBEVCostmap(
            cost=np.full((self.nx, self.ny), self.config.unknown_cost_default, dtype=np.float32),
            traversability=np.zeros((self.nx, self.ny), dtype=np.float32),
            obstacle=np.zeros((self.nx, self.ny), dtype=bool),
            hazard=np.zeros((self.nx, self.ny), dtype=np.float32),
            unknown=np.ones((self.nx, self.ny), dtype=bool),
            confidence=np.zeros((self.nx, self.ny), dtype=np.float32),
            x_robot=self.grid_x_m,
            y_robot=self.grid_y_m,
            resolution_m=self.config.resolution_m,
            x_range_m=self.config.x_range_m,
            y_range_m=self.config.y_range_m,
            timestamp=timestamp,
            frame_id="base_link",
            valid=valid,
            status_message=status_message,
        )

    def fuse_into_world_map(
        self,
        bev_costmap: LocalBEVCostmap,
        robot_pose: Tuple[float, float, float],
        world_map: WorldTerrainMap,
    ) -> Dict[str, Any]:
        """Fuses local BEV costmap into globally referenced WorldTerrainMap.

        Args:
            bev_costmap: Local metric BEV costmap from project_traversability.
            robot_pose: (x_m, y_m, yaw_rad) in world ("odom") frame.
            world_map: Multi-layer WorldTerrainMap to update.

        Returns:
            Dict containing fusion statistics and metrics.
        """
        t0 = time.perf_counter()

        if not bev_costmap.valid:
            # Conservative safety: do NOT modify world map on invalid perception
            return {
                "status": "DEGRADED",
                "valid": False,
                "reason": bev_costmap.status_message,
                "cells_updated": 0,
                "obstacles_marked": 0,
                "fusion_ms": (time.perf_counter() - t0) * 1000.0,
            }

        ugv_x, ugv_y, ugv_yaw = robot_pose
        cos_yaw = math.cos(ugv_yaw)
        sin_yaw = math.sin(ugv_yaw)

        # Only fuse observed (non-unknown) cells
        observed = ~bev_costmap.unknown
        if not np.any(observed):
            return {
                "status": "NO_OBSERVATIONS",
                "valid": True,
                "cells_updated": 0,
                "obstacles_marked": 0,
                "fusion_ms": (time.perf_counter() - t0) * 1000.0,
            }

        x_r = bev_costmap.x_robot[observed]
        y_r = bev_costmap.y_robot[observed]
        trav_val = bev_costmap.traversability[observed]
        cost_val = bev_costmap.cost[observed]
        obs_val = bev_costmap.obstacle[observed]
        haz_val = bev_costmap.hazard[observed]
        conf_val = bev_costmap.confidence[observed]

        # Transform to world coordinates:
        # x_w = ugv_x + x_r * cos(yaw) - y_r * sin(yaw)
        # y_w = ugv_y + x_r * sin(yaw) + y_r * cos(yaw)
        x_w = ugv_x + x_r * cos_yaw - y_r * sin_yaw
        y_w = ugv_y + x_r * sin_yaw + y_r * cos_yaw

        # Convert to world_map grid indices
        rows_w, cols_w = world_map.world_to_grid_vec(x_w, y_w)

        # Filter cells within world_map tensor bounds
        in_grid = (
            (rows_w >= 0)
            & (rows_w < world_map.n_rows)
            & (cols_w >= 0)
            & (cols_w < world_map.n_cols)
        )

        rows_in = rows_w[in_grid]
        cols_in = cols_w[in_grid]
        cost_in = cost_val[in_grid]
        trav_in = trav_val[in_grid]
        obs_in = obs_val[in_grid]
        haz_in = haz_val[in_grid]
        conf_in = conf_val[in_grid]

        cells_updated = len(rows_in)
        obstacles_marked = 0

        # Update layers with persistent hazard latching
        now_ts = bev_costmap.timestamp
        for i in range(cells_updated):
            r = rows_in[i]
            c = cols_in[i]
            is_obstacle = obs_in[i]
            cell_cost = cost_in[i]
            cell_hazard = haz_in[i]

            if is_obstacle:
                # Mark as obstacle and latch
                world_map.obstacle[r, c] = True
                world_map.cost[r, c] = 1.0
                world_map.hazard[r, c] = 1.0
                world_map.traversability[r, c] = 0.0
                world_map.unknown[r, c] = False
                world_map.confidence[r, c] = max(world_map.confidence[r, c], conf_in[i])
                world_map.last_observed_timestamp[r, c] = now_ts
                world_map.observation_count[r, c] += 1
                self._obstacle_clear_counter[(r, c)] = 0
                obstacles_marked += 1
            else:
                # Non-obstacle observation
                if world_map.obstacle[r, c]:
                    # Persistent hazard check: requires multiple consecutive clear observations
                    if self.config.persistent_hazard_latch:
                        count = self._obstacle_clear_counter.get((r, c), 0) + 1
                        self._obstacle_clear_counter[(r, c)] = count
                        if count < self.config.min_clear_observations_to_erase:
                            # Keep obstacle latched!
                            continue
                        else:
                            # Safely unlatch after sustained confirmation
                            world_map.obstacle[r, c] = False
                            self._obstacle_clear_counter.pop((r, c), None)

                # Update nominal terrain cost
                world_map.cost[r, c] = cell_cost
                world_map.hazard[r, c] = cell_hazard
                world_map.traversability[r, c] = trav_in[i]
                world_map.unknown[r, c] = False
                world_map.confidence[r, c] = max(world_map.confidence[r, c], conf_in[i])
                world_map.last_observed_timestamp[r, c] = now_ts
                world_map.observation_count[r, c] += 1

        fusion_ms = (time.perf_counter() - t0) * 1000.0
        self.last_update_timestamp = now_ts

        return {
            "status": "OK",
            "valid": True,
            "cells_updated": cells_updated,
            "obstacles_marked": obstacles_marked,
            "fusion_ms": fusion_ms,
        }
