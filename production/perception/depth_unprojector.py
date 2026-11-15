"""Vectorized Metric Depth Unprojector with SE(3) Attitude Compensation (Subphase 15.0).

Unprojects any (H, W) optical depth map D(u, v) + camera intrinsics/extrinsics + EKF
chassis attitude (roll_rad, pitch_rad) into gravity-aligned 3D terrain points
[X_level, Y_level, Z_level]^T in < 1.5 ms on CPU.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple
import numpy as np

from ai.traversability.geometry import CameraExtrinsics, CameraIntrinsics


class DepthUnprojector:
    """High-speed vectorized depth-to-3D unprojector with cached optical ray bundles."""

    def __init__(
        self,
        intrinsics: Optional[CameraIntrinsics] = None,
        extrinsics: Optional[CameraExtrinsics] = None,
        min_depth_m: float = 0.25,
        max_depth_m: float = 8.0,
    ) -> None:
        self.intrinsics = intrinsics or CameraIntrinsics(
            fx=381.3614, fy=381.3614, cx=320.0, cy=240.0, width=640, height=480
        )
        self.extrinsics = extrinsics or CameraExtrinsics(
            pitch_deg=15.0, roll_deg=0.0, yaw_deg=0.0, height_m=0.525, forward_offset_m=0.15
        )
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self._ray_cache: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def _get_cached_rays(self, h: int, w: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (h, w, stride)
        cached = self._ray_cache.get(key)
        if cached is not None:
            return cached

        scale_x = w / float(self.intrinsics.width)
        scale_y = h / float(self.intrinsics.height)
        fx_s = self.intrinsics.fx * scale_x
        fy_s = self.intrinsics.fy * scale_y
        cx_s = self.intrinsics.cx * scale_x
        cy_s = self.intrinsics.cy * scale_y

        v_idx, u_idx = np.mgrid[0:h:stride, 0:w:stride]
        u_flat = u_idx.ravel().astype(np.float32)
        v_flat = v_idx.ravel().astype(np.float32)
        rays_opt = np.stack(
            [(u_flat - cx_s) / fx_s, (v_flat - cy_s) / fy_s, np.ones_like(u_flat)],
            axis=0,
        )
        R_opt_to_robot = self.extrinsics.rotation_matrix.T.astype(np.float32)
        rays_base = np.ascontiguousarray(R_opt_to_robot @ rays_opt)
        t_cam_base = np.array(
            [[self.extrinsics.forward_offset_m], [0.0], [self.extrinsics.height_m]], dtype=np.float32
        )
        self._ray_cache[key] = (rays_base, t_cam_base)
        return rays_base, t_cam_base

    def unproject(
        self,
        depth_m: np.ndarray,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        stride: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """Unproject (H, W) metric depth map into (N, 3) gravity-leveled 3D points and return (points, latency_ms)."""
        t0 = time.perf_counter()
        depth = np.asarray(depth_m, dtype=np.float32)
        h, w = depth.shape
        step = max(1, int(stride))
        rays_base, t_cam_base = self._get_cached_rays(h, w, step)

        d_sub = depth[0:h:step, 0:w:step].ravel()
        valid = np.isfinite(d_sub) & (d_sub >= self.min_depth_m) & (d_sub <= self.max_depth_m)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), (time.perf_counter() - t0) * 1000.0

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
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return np.ascontiguousarray(pts_level, dtype=np.float32), dt_ms
