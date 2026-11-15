"""2.5D Elevation Mapping & Temporal Height Fusion (Subphase 15.1).

Continuously fuses unprojected 3D terrain points into the metric `elevation`
layer Z(x, y) of `LocalBEVCostmap` and `WorldTerrainMap` in < 2.0 ms on CPU:
  1. Fast flat-index raster binning via `np.bincount`, `np.minimum.at`, and `np.maximum.at`.
  2. Vectorized per-cell outlier rejection suppressing depth sensor spikes / flying pixels.
  3. Minimum point-count gating (`min_points_per_cell = 2`) preventing sparse-ray noise.
  4. Variance-weighted Kalman / Exponential Moving Average (EMA, `alpha = 0.35`) temporal
     smoothing that suppresses Gaussian sensor noise while preserving sharp step edges.
  5. Explicit `elevation_valid` mask tracking so unobserved cells outside the camera FOV
     are never hallucinated as Z = 0 next to elevated terrain (zero FOV boundary cliff artifacts).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, Optional, Tuple, Union
import numpy as np

from ai.spatial_fusion.world_map import WorldTerrainMap
from ai.traversability.geometry import CameraExtrinsics, CameraIntrinsics
from production.perception.vision_costmap_integrator import LocalBEVCostmap


@dataclass
class ElevationMapperConfig:
    """Configuration for 2.5D elevation binning, outlier rejection, and temporal height fusion."""

    min_points_per_cell: int = 2
    ema_alpha: float = 0.35
    use_kalman_variance_weighting: bool = True
    base_measurement_var_m2: float = 0.0004      # (0.02 m)^2 nominal depth variance
    process_noise_var_m2: float = 0.00005        # Process variance per frame
    outlier_rejection_enabled: bool = True
    outlier_deviation_m: float = 0.18            # Leave-one-out spike rejection threshold (meters)
    outlier_zscore_thresh: float = 1.15          # Within-cell z-score threshold (separates >=1.414 outlier from <=0.707 inliers)
    min_valid_z_m: float = -5.0
    max_valid_z_m: float = 10.0


@dataclass
class ElevationUpdateResult:
    """Diagnostics and timing returned from an elevation mapping update."""

    cells_updated: int
    newly_observed_cells: int
    outliers_rejected: int
    total_valid_cells: int
    update_ms: float


class ElevationMapper:
    """High-speed vectorized 2.5D elevation mapper with outlier rejection and Kalman/EMA fusion."""

    def __init__(self, config: Optional[ElevationMapperConfig] = None) -> None:
        self.config = config or ElevationMapperConfig()
        self._ray_cache: Dict[Tuple[int, int, int, float, float, float, float], Tuple[np.ndarray, np.ndarray]] = {}

    def update_world_map(
        self,
        world_map: WorldTerrainMap,
        points_world: np.ndarray,
        timestamp: float = 0.0,
    ) -> ElevationUpdateResult:
        """Fuse 3D world-frame points (X_w, Y_w, Z_w) into WorldTerrainMap.elevation.

        Args:
            world_map: Target WorldTerrainMap instance.
            points_world: (N, 3) or (3, N) float array of 3D points in world frame.
            timestamp: Observation timestamp in seconds.

        Returns:
            ElevationUpdateResult with update counts and execution latency.
        """
        t0 = time.perf_counter()
        pts = np.asarray(points_world, dtype=np.float32)
        if pts.ndim != 2 or pts.size == 0:
            return ElevationUpdateResult(
                cells_updated=0,
                newly_observed_cells=0,
                outliers_rejected=0,
                total_valid_cells=int(np.sum(world_map.elevation_valid)),
                update_ms=(time.perf_counter() - t0) * 1000.0,
            )

        if pts.shape[1] == 3:
            xw = pts[:, 0]
            yw = pts[:, 1]
            zw = pts[:, 2]
        elif pts.shape[0] == 3:
            xw = pts[0, :]
            yw = pts[1, :]
            zw = pts[2, :]
        else:
            raise ValueError(f"Expected points_world of shape (N, 3) or (3, N), got {pts.shape}")

        # 1. Filter finite points within valid metric bounds and Z envelope
        x_min, x_max = world_map.x_bounds_m
        y_min, y_max = world_map.y_bounds_m
        valid_pt = (
            np.isfinite(zw)
            & (xw >= x_min)
            & (xw < x_max)
            & (yw >= y_min)
            & (yw < y_max)
            & (zw >= self.config.min_valid_z_m)
            & (zw <= self.config.max_valid_z_m)
        )
        if not np.any(valid_pt):
            return ElevationUpdateResult(
                cells_updated=0,
                newly_observed_cells=0,
                outliers_rejected=0,
                total_valid_cells=int(np.sum(world_map.elevation_valid)),
                update_ms=(time.perf_counter() - t0) * 1000.0,
            )

        xw_in = xw[valid_pt]
        yw_in = yw[valid_pt]
        zw_in = zw[valid_pt]

        inv_res = np.float32(1.0 / world_map.resolution_m)
        rows = ((np.float32(x_max) - xw_in) * inv_res).astype(np.int32)
        cols = ((np.float32(y_max) - yw_in) * inv_res).astype(np.int32)

        n_rows = world_map.n_rows
        n_cols = world_map.n_cols
        in_grid = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
        if not np.any(in_grid):
            return ElevationUpdateResult(
                cells_updated=0,
                newly_observed_cells=0,
                outliers_rejected=0,
                total_valid_cells=int(np.sum(world_map.elevation_valid)),
                update_ms=(time.perf_counter() - t0) * 1000.0,
            )

        flat_idx = rows[in_grid] * n_cols + cols[in_grid]
        z_vals = zw_in[in_grid]

        min_flat = int(flat_idx.min())
        max_flat = int(flat_idx.max())
        span = max_flat - min_flat + 1
        rel_idx = flat_idx - min_flat

        # 2. Fast bounded-span bincount and vectorized outlier rejection
        cnt = np.bincount(rel_idx, minlength=span)
        sum_z = np.bincount(rel_idx, weights=z_vals, minlength=span)
        outliers_rejected = 0

        wm_valid_flat = world_map.elevation_valid.ravel()
        wm_elev_flat = world_map.elevation.ravel()
        wm_var_flat = world_map.elevation_var.ravel()

        if self.config.outlier_rejection_enabled and z_vals.size >= 3:
            pt_cnt = cnt[rel_idx]
            pt_sum = sum_z[rel_idx]
            dev_scaled = np.abs(z_vals * pt_cnt - pt_sum)
            cand_multi = (pt_cnt >= 3) & (
                dev_scaled > (pt_cnt - 1) * np.float32(self.config.outlier_deviation_m)
            )

            if np.any(cand_multi):
                sum_z2 = np.bincount(rel_idx, weights=z_vals * z_vals, minlength=span)
                inv_cnt = np.float32(1.0) / np.maximum(1, pt_cnt).astype(np.float32)
                pt_mean = pt_sum * inv_cnt
                pt_var = np.maximum(np.float32(0.0), (sum_z2[rel_idx] * inv_cnt) - pt_mean * pt_mean)
                is_spike_multi = cand_multi & (
                    np.abs(z_vals - pt_mean) > np.float32(self.config.outlier_zscore_thresh) * np.sqrt(pt_var)
                )
            else:
                is_spike_multi = cand_multi

            if np.any(is_spike_multi):
                outliers_rejected = int(np.count_nonzero(is_spike_multi))
                inlier_mask = ~is_spike_multi
                flat_idx = flat_idx[inlier_mask]
                rel_idx = rel_idx[inlier_mask]
                z_vals = z_vals[inlier_mask]
                cnt = np.bincount(rel_idx, minlength=span)
                sum_z = np.bincount(rel_idx, weights=z_vals, minlength=span)

        # 3. Minimum point-count gating
        min_pts = max(1, int(self.config.min_points_per_cell))
        obs_rel = np.nonzero(cnt >= min_pts)[0]
        cells_updated = int(obs_rel.size)

        if cells_updated == 0:
            return ElevationUpdateResult(
                cells_updated=0,
                newly_observed_cells=0,
                outliers_rejected=outliers_rejected,
                total_valid_cells=int(np.count_nonzero(wm_valid_flat)),
                update_ms=(time.perf_counter() - t0) * 1000.0,
            )

        obs_indices = obs_rel + min_flat
        was_valid = wm_valid_flat[obs_indices]
        newly_observed = int(cells_updated - np.count_nonzero(was_valid))

        # 4. Compute cell mean observation and fuse via Kalman / EMA
        obs_cnt_f = cnt[obs_rel].astype(np.float32)
        z_meas = (sum_z[obs_rel] / obs_cnt_f).astype(np.float32)
        r_meas = np.float32(self.config.base_measurement_var_m2) / obs_cnt_f

        wm_min_flat = world_map.elevation_min.ravel()
        wm_max_flat = world_map.elevation_max.ravel()

        if newly_observed == cells_updated:
            # Fast path: all updated cells are observed for the first time
            wm_elev_flat[obs_indices] = z_meas
            wm_var_flat[obs_indices] = r_meas
            wm_valid_flat[obs_indices] = True
            wm_min_flat[obs_indices] = z_meas
            wm_max_flat[obs_indices] = z_meas
        elif newly_observed == 0:
            # Fast path: all updated cells were already valid
            z_prev = wm_elev_flat[obs_indices]
            alpha = np.float32(self.config.ema_alpha)
            if self.config.use_kalman_variance_weighting:
                p_pred = wm_var_flat[obs_indices] + np.float32(self.config.process_noise_var_m2)
                k_gain = p_pred / np.maximum(np.float32(1e-7), p_pred + r_meas)
                eff_gain = np.clip(np.float32(0.5) * alpha + np.float32(0.5) * k_gain, np.float32(0.15), np.float32(0.85))
                wm_var_flat[obs_indices] = (np.float32(1.0) - eff_gain) * p_pred
            else:
                eff_gain = alpha
            wm_elev_flat[obs_indices] = (np.float32(1.0) - eff_gain) * z_prev + eff_gain * z_meas
            wm_min_flat[obs_indices] = np.minimum(wm_min_flat[obs_indices], z_meas)
            wm_max_flat[obs_indices] = np.maximum(wm_max_flat[obs_indices], z_meas)
        else:
            first_cells = obs_indices[~was_valid]
            wm_elev_flat[first_cells] = z_meas[~was_valid]
            wm_var_flat[first_cells] = r_meas[~was_valid]
            wm_valid_flat[first_cells] = True
            wm_min_flat[first_cells] = z_meas[~was_valid]
            wm_max_flat[first_cells] = z_meas[~was_valid]

            rep_cells = obs_indices[was_valid]
            z_prev = wm_elev_flat[rep_cells]
            z_obs_rep = z_meas[was_valid]
            r_rep = r_meas[was_valid]
            alpha = np.float32(self.config.ema_alpha)
            if self.config.use_kalman_variance_weighting:
                p_pred = wm_var_flat[rep_cells] + np.float32(self.config.process_noise_var_m2)
                k_gain = p_pred / np.maximum(np.float32(1e-7), p_pred + r_rep)
                eff_gain = np.clip(np.float32(0.5) * alpha + np.float32(0.5) * k_gain, np.float32(0.15), np.float32(0.85))
                wm_var_flat[rep_cells] = (np.float32(1.0) - eff_gain) * p_pred
            else:
                eff_gain = alpha
            wm_elev_flat[rep_cells] = (np.float32(1.0) - eff_gain) * z_prev + eff_gain * z_obs_rep
            wm_min_flat[rep_cells] = np.minimum(wm_min_flat[rep_cells], z_obs_rep)
            wm_max_flat[rep_cells] = np.maximum(wm_max_flat[rep_cells], z_obs_rep)

        if timestamp > 0.0:
            world_map.last_observed_timestamp.ravel()[obs_indices] = timestamp

        update_ms = (time.perf_counter() - t0) * 1000.0
        return ElevationUpdateResult(
            cells_updated=cells_updated,
            newly_observed_cells=newly_observed,
            outliers_rejected=outliers_rejected,
            total_valid_cells=int(np.count_nonzero(wm_valid_flat)),
            update_ms=update_ms,
        )

    def update_from_robot_points(
        self,
        world_map: WorldTerrainMap,
        points_robot: np.ndarray,
        robot_pose: Union[Tuple[float, ...], np.ndarray],
        timestamp: float = 0.0,
    ) -> ElevationUpdateResult:
        """Transform 3D points in gravity-leveled or robot frame into world frame and fuse."""
        pts = np.asarray(points_robot, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] == 3 and pts.shape[1] != 3:
            pts = pts.T

        if isinstance(robot_pose, np.ndarray) and robot_pose.shape == (4, 4):
            R = robot_pose[:3, :3].astype(np.float32)
            t = robot_pose[:3, 3].astype(np.float32)
            pts_world = (pts @ R.T) + t
        else:
            pose_tuple = tuple(float(v) for v in robot_pose)
            if len(pose_tuple) == 3:
                ugv_x, ugv_y, ugv_yaw = pose_tuple
                ugv_z = 0.0
            elif len(pose_tuple) >= 6:
                ugv_x, ugv_y, ugv_z = pose_tuple[0], pose_tuple[1], pose_tuple[2]
                ugv_yaw = pose_tuple[5]
            else:
                raise ValueError(f"Unsupported robot_pose length: {len(pose_tuple)}")

            cy = np.float32(math.cos(ugv_yaw))
            sy = np.float32(math.sin(ugv_yaw))
            xr = pts[:, 0]
            yr = pts[:, 1]
            zr = pts[:, 2]
            xw = np.float32(ugv_x) + xr * cy - yr * sy
            yw = np.float32(ugv_y) + xr * sy + yr * cy
            zw = np.float32(ugv_z) + zr
            pts_world = np.stack([xw, yw, zw], axis=-1)

        return self.update_world_map(world_map, pts_world, timestamp=timestamp)

    def update_local_bev(
        self,
        bev_costmap: LocalBEVCostmap,
        points_robot: np.ndarray,
    ) -> ElevationUpdateResult:
        """Bin 3D points (x_robot, y_robot, z_robot) into LocalBEVCostmap.elevation."""
        t0 = time.perf_counter()
        pts = np.asarray(points_robot, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] == 3 and pts.shape[1] != 3:
            pts = pts.T

        nx, ny = bev_costmap.shape
        if bev_costmap.elevation is None:
            bev_costmap.elevation = np.zeros((nx, ny), dtype=np.float32)
        if bev_costmap.elevation_valid is None:
            bev_costmap.elevation_valid = np.zeros((nx, ny), dtype=bool)

        if pts.size == 0:
            return ElevationUpdateResult(0, 0, 0, int(np.sum(bev_costmap.elevation_valid)), 0.0)

        xr = pts[:, 0]
        yr = pts[:, 1]
        zr = pts[:, 2]

        x_min, x_max = bev_costmap.x_range_m
        y_min, y_max = bev_costmap.y_range_m
        inv_res = np.float32(1.0 / bev_costmap.resolution_m)

        valid = (
            np.isfinite(zr)
            & (xr >= x_min)
            & (xr < x_max)
            & (yr >= y_min)
            & (yr < y_max)
            & (zr >= self.config.min_valid_z_m)
            & (zr <= self.config.max_valid_z_m)
        )
        if not np.any(valid):
            return ElevationUpdateResult(
                0, 0, 0, int(np.sum(bev_costmap.elevation_valid)), (time.perf_counter() - t0) * 1000.0
            )

        rows = np.clip(((np.float32(x_max) - xr[valid]) * inv_res).astype(np.int32), 0, nx - 1)
        cols = np.clip(((np.float32(y_max) - yr[valid]) * inv_res).astype(np.int32), 0, ny - 1)
        flat_idx = rows * ny + cols
        z_vals = zr[valid]
        total_cells = nx * ny

        cnt = np.bincount(flat_idx, minlength=total_cells)
        sum_z = np.bincount(flat_idx, weights=z_vals, minlength=total_cells)

        min_pts = max(1, int(self.config.min_points_per_cell))
        obs_mask = cnt >= min_pts
        obs_indices = np.nonzero(obs_mask)[0]
        cells_updated = int(obs_indices.size)
        if cells_updated == 0:
            return ElevationUpdateResult(
                0, 0, 0, int(np.sum(bev_costmap.elevation_valid)), (time.perf_counter() - t0) * 1000.0
            )

        elev_flat = bev_costmap.elevation.ravel()
        valid_flat = bev_costmap.elevation_valid.ravel()
        was_valid = valid_flat[obs_indices]
        newly_obs = int(np.sum(~was_valid))
        z_meas = (sum_z[obs_indices] / cnt[obs_indices]).astype(np.float32)

        alpha = np.float32(self.config.ema_alpha)
        elev_flat[obs_indices] = np.where(
            was_valid,
            (np.float32(1.0) - alpha) * elev_flat[obs_indices] + alpha * z_meas,
            z_meas,
        )
        valid_flat[obs_indices] = True

        return ElevationUpdateResult(
            cells_updated=cells_updated,
            newly_observed_cells=newly_obs,
            outliers_rejected=0,
            total_valid_cells=int(np.sum(valid_flat)),
            update_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def unproject_depth_to_points(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        min_depth_m: float = 0.25,
        max_depth_m: float = 8.0,
        stride: int = 1,
    ) -> np.ndarray:
        """Unproject (H, W) optical depth array into (N, 3) gravity-aligned robot-frame points."""
        depth = np.asarray(depth_m, dtype=np.float32)
        h, w = depth.shape
        step = max(1, int(stride))
        cache_key = (
            h,
            w,
            step,
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(extrinsics.pitch_deg),
            float(extrinsics.height_m),
        )
        cached = self._ray_cache.get(cache_key)
        if cached is not None:
            rays_base, t_cam_base = cached
        else:
            scale_x = w / float(intrinsics.width)
            scale_y = h / float(intrinsics.height)
            fx_s = intrinsics.fx * scale_x
            fy_s = intrinsics.fy * scale_y
            cx_s = intrinsics.cx * scale_x
            cy_s = intrinsics.cy * scale_y

            v_idx, u_idx = np.mgrid[0:h:step, 0:w:step]
            u_flat = u_idx.ravel().astype(np.float32)
            v_flat = v_idx.ravel().astype(np.float32)
            rays_opt = np.stack(
                [
                    (u_flat - cx_s) / fx_s,
                    (v_flat - cy_s) / fy_s,
                    np.ones_like(u_flat),
                ],
                axis=0,
            )
            R_opt_to_robot = extrinsics.rotation_matrix.T.astype(np.float32)
            rays_base = np.ascontiguousarray(R_opt_to_robot @ rays_opt)
            t_cam_base = np.array(
                [[extrinsics.forward_offset_m], [0.0], [extrinsics.height_m]], dtype=np.float32
            )
            self._ray_cache[cache_key] = (rays_base, t_cam_base)

        d_sub = depth[0:h:step, 0:w:step].ravel()
        valid = np.isfinite(d_sub) & (d_sub >= min_depth_m) & (d_sub <= max_depth_m)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        zc = d_sub[valid]
        rays_valid = rays_base[:, valid]

        if abs(roll_rad) > 1e-4 or abs(pitch_rad) > 1e-4:
            cr = math.cos(roll_rad)
            sr = math.sin(roll_rad)
            cp = math.cos(pitch_rad)
            sp = math.sin(pitch_rad)
            R_level = np.array(
                [
                    [cp, sp * sr, sp * cr],
                    [0.0, cr, -sr],
                    [-sp, cp * sr, cp * cr],
                ],
                dtype=np.float32,
            )
            d_level = R_level @ rays_valid
            t_level = R_level @ t_cam_base
        else:
            d_level = rays_valid
            t_level = t_cam_base

        pts_level = (t_level + d_level * zc[None, :]).T
        return np.ascontiguousarray(pts_level, dtype=np.float32)

    def update_from_depth_image(
        self,
        world_map: WorldTerrainMap,
        depth_m: np.ndarray,
        robot_pose: Union[Tuple[float, ...], np.ndarray],
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        timestamp: float = 0.0,
        stride: int = 1,
    ) -> ElevationUpdateResult:
        """Unproject a metric depth image and fuse directly into WorldTerrainMap.elevation."""
        pts_robot = self.unproject_depth_to_points(
            depth_m=depth_m,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            stride=stride,
        )
        return self.update_from_robot_points(
            world_map=world_map,
            points_robot=pts_robot,
            robot_pose=robot_pose,
            timestamp=timestamp,
        )
