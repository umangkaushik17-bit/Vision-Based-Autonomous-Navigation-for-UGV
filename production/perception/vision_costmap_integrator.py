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

import cv2
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
    elevation: Optional[np.ndarray] = None        # (nx, ny) float32 metric elevation Z(x, y)
    elevation_valid: Optional[np.ndarray] = None  # (nx, ny) bool mask of valid elevation cells

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

        # Precomputed ray-projection lookup tables (LUT) with shape cache
        self._lut_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._lut_shape: Optional[Tuple[int, int]] = None
        self._lut_valid_v: Optional[np.ndarray] = None
        self._lut_valid_u: Optional[np.ndarray] = None
        self._lut_rows: Optional[np.ndarray] = None
        self._lut_cols: Optional[np.ndarray] = None
        self._rays_base: Optional[np.ndarray] = None
        self._rx_base: Optional[np.ndarray] = None
        self._ry_base: Optional[np.ndarray] = None
        self._rz_base: Optional[np.ndarray] = None
        self._u_flat: Optional[np.ndarray] = None
        self._v_flat: Optional[np.ndarray] = None
        self._t_cam_base: Optional[np.ndarray] = None

        # Precompute for standard 480x640 camera and default 240x320 model output resolution
        self.precompute_projection((480, 640))
        self.precompute_projection((240, 320))

        # Precompute radial obstacle inflation structuring kernel (WS-2A)
        self._radial_kernel: Optional[np.ndarray] = None
        if self.config.obstacle_inflation_radius_m > 0:
            r_cells = int(np.ceil(self.config.obstacle_inflation_radius_m / self.config.resolution_m))
            yg, xg = np.ogrid[-r_cells : r_cells + 1, -r_cells : r_cells + 1]
            self._radial_kernel = (xg * xg + yg * yg <= r_cells * r_cells).astype(np.uint8)

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
        if image_shape in self._lut_cache:
            entry = self._lut_cache[image_shape]
            self._lut_shape = image_shape
            self._lut_valid_v = entry["valid_v"]
            self._lut_valid_u = entry["valid_u"]
            self._lut_rows = entry["rows"]
            self._lut_cols = entry["cols"]
            self._rays_base = entry["rays_base"]
            self._rx_base = entry["rx_base"]
            self._ry_base = entry["ry_base"]
            self._rz_base = entry["rz_base"]
            self._u_flat = entry["u_flat"]
            self._v_flat = entry["v_flat"]
            self._t_cam_base = entry["t_cam_base"]
            return

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

        # Precompute 3D camera ray vectors in nominal base_link frame for SE(3) attitude compensation
        # Pre-cull sky rays (> +20 deg above horizon, dz >= 0.35) which cannot hit ground under +/-20 deg pitch/roll
        u_flat = u_indices.ravel().astype(np.float32)
        v_flat = v_indices.ravel().astype(np.float32)
        rays_opt = np.stack([
            (u_flat - cx_scaled) / fx_scaled,
            (v_flat - cy_scaled) / fy_scaled,
            np.ones_like(u_flat),
        ], axis=0)  # (3, H*W)
        rays_base_all = (R_opt_to_robot @ rays_opt).astype(np.float32)
        u_raw = u_indices.ravel()
        v_raw = v_indices.ravel()
        # Keep 100% of mid/far-field pixels (v < 0.50*h, x >= 2.1m) and 2x2 subsample in ultra-dense near field
        near_keep = (v_raw < int(h * 0.50)) | (((u_raw & 1) == 0) & ((v_raw & 1) == 0))
        cand_mask = (rays_base_all[2, :] < 0.35) & near_keep
        self._rays_base = np.ascontiguousarray(rays_base_all[:, cand_mask])
        self._rx_base = np.ascontiguousarray(self._rays_base[0, :])
        self._ry_base = np.ascontiguousarray(self._rays_base[1, :])
        self._rz_base = np.ascontiguousarray(self._rays_base[2, :])
        self._u_flat = np.ascontiguousarray(u_raw[cand_mask])
        self._v_flat = np.ascontiguousarray(v_raw[cand_mask])
        self._t_cam_base = t_robot.astype(np.float32)

        self._lut_shape = (h, w)
        self._lut_valid_v = valid_v_in
        self._lut_valid_u = valid_u_in
        self._lut_rows = rows
        self._lut_cols = cols
        self._lut_cache[(h, w)] = {
            "valid_v": self._lut_valid_v,
            "valid_u": self._lut_valid_u,
            "rows": self._lut_rows,
            "cols": self._lut_cols,
            "rays_base": self._rays_base,
            "rx_base": self._rx_base,
            "ry_base": self._ry_base,
            "rz_base": self._rz_base,
            "u_flat": self._u_flat,
            "v_flat": self._v_flat,
            "t_cam_base": self._t_cam_base,
        }

    @staticmethod
    def _attitude_rotation_matrix(roll_rad: float, pitch_rad: float) -> np.ndarray:
        """Constructs 3x3 rigid-body rotation R_y(pitch) @ R_x(roll) from tilted base_link to level ground frame."""
        cr = math.cos(roll_rad)
        sr = math.sin(roll_rad)
        cp = math.cos(pitch_rad)
        sp = math.sin(pitch_rad)
        return np.array([
            [cp, sp * sr, sp * cr],
            [0.0, cr, -sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float32)

    def compute_attitude_compensated_lut(
        self,
        roll_rad: float,
        pitch_rad: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes dynamic SE(3) attitude-compensated ground-plane projection indices.

        Compensates camera ray directions and optical center height for instantaneous
        vehicle chassis roll and pitch (REP-103: X-forward, Y-left, Z-up).
        """
        if abs(roll_rad) <= 1e-4 and abs(pitch_rad) <= 1e-4:
            return self._lut_valid_v, self._lut_valid_u, self._lut_rows, self._lut_cols

        R_level = self._attitude_rotation_matrix(roll_rad, pitch_rad)
        t_level = R_level @ self._t_cam_base  # (3, 1)
        h_eff = np.float32(t_level[2, 0])
        if h_eff <= 0.0:
            empty_i = np.empty(0, dtype=np.int32)
            return empty_i, empty_i, empty_i, empty_i

        rx = self._rx_base
        ry = self._ry_base
        rz = self._rz_base

        r00, r01, r02 = np.float32(R_level[0, 0]), np.float32(R_level[0, 1]), np.float32(R_level[0, 2])
        r11, r12 = np.float32(R_level[1, 1]), np.float32(R_level[1, 2])
        r20, r21, r22 = np.float32(R_level[2, 0]), np.float32(R_level[2, 1]), np.float32(R_level[2, 2])

        dz = r20 * rx + r21 * ry + r22 * rz
        zc = -h_eff / np.minimum(dz, np.float32(-1e-5))
        x_r = np.float32(t_level[0, 0]) + zc * (r00 * rx + r01 * ry + r02 * rz)
        y_r = np.float32(t_level[1, 0]) + zc * (r11 * ry + r12 * rz)

        x_min, x_max = self.config.x_range_m
        y_min, y_max = self.config.y_range_m
        dz_thresh = min(-1e-4, float(-h_eff / 12.0))
        in_bounds = (
            (dz <= dz_thresh)
            & (x_r >= x_min)
            & (x_r < x_max)
            & (y_r >= y_min)
            & (y_r < y_max)
        )
        idx_in = np.nonzero(in_bounds)[0]
        if idx_in.size == 0:
            empty_i = np.empty(0, dtype=np.int32)
            return empty_i, empty_i, empty_i, empty_i

        inv_res = np.float32(1.0 / self.config.resolution_m)
        rows = ((np.float32(x_max) - x_r[idx_in]) * inv_res).astype(np.int32)
        cols = ((np.float32(y_max) - y_r[idx_in]) * inv_res).astype(np.int32)

        return self._v_flat[idx_in], self._u_flat[idx_in], rows, cols

    def project_ground_point_to_pixel(
        self,
        x_ground_m: float,
        y_ground_m: float,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Projects a 3D point (x_ground_m, y_ground_m, 0.0) on the level ground plane into image pixel (v, u)
        under chassis attitude (roll_rad, pitch_rad). Useful for synthetic testing and calibration verification.
        """
        h, w = image_shape if image_shape is not None else (self._lut_shape or (240, 320))
        intr = self.config.intrinsics
        extr = self.config.extrinsics

        scale_x = w / float(intr.width)
        scale_y = h / float(intr.height)
        fx_s = intr.fx * scale_x
        fy_s = intr.fy * scale_y
        cx_s = intr.cx * scale_x
        cy_s = intr.cy * scale_y

        R_level = self._attitude_rotation_matrix(roll_rad, pitch_rad)
        p_level = np.array([x_ground_m, y_ground_m, 0.0], dtype=np.float32)
        p_base = R_level.T @ p_level

        t_cam = np.array([extr.forward_offset_m, 0.0, extr.height_m], dtype=np.float32)
        p_opt = extr.rotation_matrix @ (p_base - t_cam)

        if p_opt[2] <= 0.05:
            return None

        u = int(round(fx_s * (p_opt[0] / p_opt[2]) + cx_s))
        v = int(round(fy_s * (p_opt[1] / p_opt[2]) + cy_s))
        if 0 <= v < h and 0 <= u < w:
            return v, u
        return None

    def project_traversability(
        self,
        trav_result: TraversabilityResult,
        current_time_s: Optional[float] = None,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        photometric_confidence: Optional[np.ndarray] = None,
    ) -> LocalBEVCostmap:
        """Projects a TraversabilityResult into a local metric BEV costmap.

        Supports real-time 3D SE(3) attitude compensation (roll_rad, pitch_rad),
        photometric glare/shadow confidence attenuation, and fail-safe degradation
        on missing/invalid/stale inputs.
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

        # 1b. Photometric glare/shadow confidence attenuation (Subphase 16.1)
        conf_src = photometric_confidence if photometric_confidence is not None else trav_result.confidence
        if conf_src is not None:
            conf_arr = np.asarray(conf_src, dtype=np.float32)
            if conf_arr.shape != (h, w):
                conf_arr = cv2.resize(conf_arr, (w, h), interpolation=cv2.INTER_NEAREST)
            low_conf = conf_arr < np.float32(0.45)
            if np.any(low_conf):
                mask = np.where(
                    low_conf & (mask < np.float32(self.config.safe_prob_thresh)),
                    conf_arr * mask + (np.float32(1.0) - conf_arr) * np.float32(0.75),
                    mask,
                )

        # 2. Extract valid ground ray scores (using attitude-compensated LUT when tilted)
        if abs(roll_rad) > 1e-4 or abs(pitch_rad) > 1e-4:
            lut_v, lut_u, lut_rows, lut_cols = self.compute_attitude_compensated_lut(roll_rad, pitch_rad)
        else:
            lut_v = self._lut_valid_v
            lut_u = self._lut_valid_u
            lut_rows = self._lut_rows
            lut_cols = self._lut_cols

        scores_in = mask.ravel()[lut_v * w + lut_u]

        # Compute flat cell index and fast bincount observation count
        flat_idx = lut_rows * self.ny + lut_cols
        total_cells = self.nx * self.ny
        flat_count = np.bincount(flat_idx, minlength=total_cells)
        grid_count = flat_count.reshape((self.nx, self.ny))

        # Exact O(1)-seeded min-pooling: seed each cell with a valid ray score in that cell,
        # then only invoke unbuffered minimum.at on rays strictly smaller than the seed
        flat_min = np.ones(total_cells, dtype=np.float32)
        flat_min[flat_idx] = scores_in
        lower_mask = scores_in < flat_min[flat_idx]
        if np.any(lower_mask):
            np.minimum.at(flat_min, flat_idx[lower_mask], scores_in[lower_mask])
        grid_min = flat_min.reshape((self.nx, self.ny))

        observed_mask = grid_count > 0
        unknown_mask = ~observed_mask

        # Drivability score: use conservative min-pooling
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

        # 4. Obstacle Inflation Buffer (WS-2A Native C++ Radial Dilation)
        if self._radial_kernel is not None and np.any(obstacle):
            dilated = cv2.dilate(obstacle.astype(np.uint8), self._radial_kernel) > 0
            inflated_ring = dilated & (~obstacle)
            cost[inflated_ring] = np.maximum(cost[inflated_ring], self.config.inflation_cost)

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
        now_ts = bev_costmap.timestamp

        # 1. Vectorized fast-path for non-obstacle observations on non-latched cells
        existing_obs = world_map.obstacle[rows_in, cols_in]
        free_direct_mask = (~obs_in) & (~existing_obs)
        if np.any(free_direct_mask):
            rf = rows_in[free_direct_mask]
            cf = cols_in[free_direct_mask]
            world_map.cost[rf, cf] = cost_in[free_direct_mask]
            world_map.hazard[rf, cf] = haz_in[free_direct_mask]
            world_map.traversability[rf, cf] = trav_in[free_direct_mask]
            world_map.unknown[rf, cf] = False
            world_map.confidence[rf, cf] = np.maximum(world_map.confidence[rf, cf], conf_in[free_direct_mask])
            world_map.last_observed_timestamp[rf, cf] = now_ts
            world_map.observation_count[rf, cf] += 1

        # 2. Vectorized fast-path for new/confirmed obstacle observations
        obstacles_marked = int(np.sum(obs_in))
        if obstacles_marked > 0:
            ro = rows_in[obs_in]
            co = cols_in[obs_in]
            world_map.obstacle[ro, co] = True
            world_map.cost[ro, co] = 1.0
            world_map.hazard[ro, co] = 1.0
            world_map.traversability[ro, co] = 0.0
            world_map.unknown[ro, co] = False
            world_map.confidence[ro, co] = np.maximum(world_map.confidence[ro, co], conf_in[obs_in])
            world_map.last_observed_timestamp[ro, co] = now_ts
            world_map.observation_count[ro, co] += 1
            for r_val, c_val in zip(ro.tolist(), co.tolist()):
                self._obstacle_clear_counter[(r_val, c_val)] = 0

        # 3. Persistent hazard latch hysteresis only for previously-latched obstacle cells now seen as clear
        latched_check_mask = (~obs_in) & existing_obs
        if np.any(latched_check_mask):
            for idx in np.nonzero(latched_check_mask)[0]:
                r = int(rows_in[idx])
                c = int(cols_in[idx])
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
                else:
                    world_map.obstacle[r, c] = False

                world_map.cost[r, c] = cost_in[idx]
                world_map.hazard[r, c] = haz_in[idx]
                world_map.traversability[r, c] = trav_in[idx]
                world_map.unknown[r, c] = False
                world_map.confidence[r, c] = max(world_map.confidence[r, c], conf_in[idx])
                world_map.last_observed_timestamp[r, c] = now_ts
                world_map.observation_count[r, c] += 1

        # 4. Fuse 2.5D elevation layer if populated on bev_costmap
        if bev_costmap.elevation is not None:
            elev_obs_mask = (
                bev_costmap.elevation_valid
                if bev_costmap.elevation_valid is not None
                else observed
            )
            elev_valid_obs = elev_obs_mask[observed][in_grid]
            if np.any(elev_valid_obs):
                elev_vals = bev_costmap.elevation[observed][in_grid][elev_valid_obs].astype(np.float32)
                re = rows_in[elev_valid_obs]
                ce = cols_in[elev_valid_obs]
                was_valid = world_map.elevation_valid[re, ce]
                alpha = np.float32(0.35)
                world_map.elevation[re, ce] = np.where(
                    was_valid,
                    (np.float32(1.0) - alpha) * world_map.elevation[re, ce] + alpha * elev_vals,
                    elev_vals,
                )
                world_map.elevation_valid[re, ce] = True

        fusion_ms = (time.perf_counter() - t0) * 1000.0
        self.last_update_timestamp = now_ts

        return {
            "status": "OK",
            "valid": True,
            "cells_updated": cells_updated,
            "obstacles_marked": obstacles_marked,
            "fusion_ms": fusion_ms,
        }
