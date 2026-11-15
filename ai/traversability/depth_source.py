"""Modular depth abstractions decoupling traversability projection from physical depth hardware."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np

from ai.traversability.geometry import CameraExtrinsics, CameraIntrinsics


class DepthSource(ABC):
    """Abstract interface supplying per-pixel metric depth (in meters) for projection."""

    @abstractmethod
    def get_depth_meters(
        self,
        image_shape: tuple[int, int],
        rgb_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute or retrieve (H, W) float32 metric depth array in meters.

        Args:
            image_shape: (height, width) of the target depth map.
            rgb_image: Optional RGB frame for monocular estimation models.

        Returns:
            (H, W) float32 array where values > 0 represent metric depth along optical Z axis.
        """
        pass


class PlanarGroundDepthSource(DepthSource):
    """Analytical planar ground depth model using calibrated camera pitch and height.

    Computes ray intersection with the ground plane Z_ground = 0:
      Z_optical(u, v) = height / (sin(pitch) + ((v - cy) / fy) * cos(pitch))
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        max_depth_m: float = 12.0
    ) -> None:
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.max_depth_m = float(max_depth_m)
        self._precomputed_depth: Optional[np.ndarray] = None
        self._precompute_ground_depth()

    def _precompute_ground_depth(self) -> None:
        h, w = self.intrinsics.height, self.intrinsics.width
        v_coords, _ = np.mgrid[0:h, 0:w]

        pitch_rad = np.radians(self.extrinsics.pitch_deg)
        height = self.extrinsics.height_m
        fy = self.intrinsics.fy
        cy = self.intrinsics.cy

        # Angle of ray relative to optical axis
        ray_angle = (v_coords - cy) / fy

        # Denominator represents the vertical component of the ray pointing down
        denom = np.sin(pitch_rad) + ray_angle * np.cos(pitch_rad)

        depth = np.full((h, w), np.nan, dtype=np.float32)
        valid = denom > 1e-4
        depth[valid] = height / denom[valid]

        # Clip negative / sky pixels and distances beyond max range
        depth[depth <= 0.0] = np.nan
        depth[depth > self.max_depth_m] = np.nan

        self._precomputed_depth = depth

    def get_depth_meters(
        self,
        image_shape: tuple[int, int],
        rgb_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        h, w = image_shape
        if self._precomputed_depth is None or self._precomputed_depth.shape != (h, w):
            self.intrinsics.height = h
            self.intrinsics.width = w
            self._precompute_ground_depth()
        return self._precomputed_depth.copy()


class StereoDepthSource(DepthSource):
    """Direct depth ingest from stereo disparity processors (e.g. ZED, RealSense D435)."""

    def __init__(self, raw_depth: Optional[np.ndarray] = None) -> None:
        self._depth = raw_depth

    def set_depth(self, depth_meters: np.ndarray) -> None:
        """Update current frame's metric depth map."""
        self._depth = np.asarray(depth_meters, dtype=np.float32)

    def get_depth_meters(
        self,
        image_shape: tuple[int, int],
        rgb_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self._depth is None:
            raise RuntimeError("Stereo depth has not been supplied for current frame.")
        return self._depth.copy()


class EstimatedMonocularDepthSource(DepthSource):
    """Monocular relative depth scaled to metric ground estimation."""

    def __init__(self, scale_factor: float = 5.0) -> None:
        self.scale_factor = scale_factor

    def get_depth_meters(
        self,
        image_shape: tuple[int, int],
        rgb_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        h, w = image_shape
        # Vertical gradient proxy representing ground perspective depth if no DNN depth active
        y_indices = np.linspace(1.0, 0.1, h, dtype=np.float32)[:, None]
        depth_map = (1.0 / np.clip(y_indices, 0.05, 1.0)) * self.scale_factor
        return np.repeat(depth_map, w, axis=1)


class SimulatedDepthSource(DepthSource):
    """Ingests simulated ground truth depth buffers from Gazebo or Isaac Sim."""

    def __init__(self, depth_buffer: Optional[np.ndarray] = None) -> None:
        self._buffer = depth_buffer

    def update_buffer(self, depth_buffer: np.ndarray) -> None:
        self._buffer = np.asarray(depth_buffer, dtype=np.float32)

    def get_depth_meters(
        self,
        image_shape: tuple[int, int],
        rgb_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self._buffer is None:
            h, w = image_shape
            return np.full((h, w), 3.0, dtype=np.float32)
        return self._buffer.copy()


class AnalyticalTerrainDepthSource(DepthSource):
    """Ray-casts camera pixels against parametric 2.5D heightfields Z = f(x, y) under 3D attitude (Subphase 15.0)."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        terrain_fn: Optional[Any] = None,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        max_depth_m: float = 10.0,
    ) -> None:
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.terrain_fn = terrain_fn or (lambda x, y: np.zeros_like(x, dtype=np.float32))
        self.roll_rad = float(roll_rad)
        self.pitch_rad = float(pitch_rad)
        self.max_depth_m = float(max_depth_m)

    def get_depth_meters(
        self,
        image_shape: tuple[int, int],
        rgb_image: Optional[np.ndarray] = None,
        roll_rad: Optional[float] = None,
        pitch_rad: Optional[float] = None,
    ) -> np.ndarray:
        """Compute optical Z depth map D(u, v) by fixed-point ray-marching against Z = terrain_fn(x, y)."""
        h, w = image_shape
        r_ang = self.roll_rad if roll_rad is None else float(roll_rad)
        p_ang = self.pitch_rad if pitch_rad is None else float(pitch_rad)

        scale_x = w / float(self.intrinsics.width)
        scale_y = h / float(self.intrinsics.height)
        fx_s = self.intrinsics.fx * scale_x
        fy_s = self.intrinsics.fy * scale_y
        cx_s = self.intrinsics.cx * scale_x
        cy_s = self.intrinsics.cy * scale_y

        v_idx, u_idx = np.mgrid[0:h, 0:w]
        u_f = u_idx.ravel().astype(np.float32)
        v_f = v_idx.ravel().astype(np.float32)
        rays_opt = np.stack([(u_f - cx_s) / fx_s, (v_f - cy_s) / fy_s, np.ones_like(u_f)], axis=0)

        R_opt_to_robot = self.extrinsics.rotation_matrix.T.astype(np.float32)
        rays_base = R_opt_to_robot @ rays_opt
        t_cam_base = np.array(
            [[self.extrinsics.forward_offset_m], [0.0], [self.extrinsics.height_m]], dtype=np.float32
        )

        cr = np.float32(np.cos(r_ang))
        sr = np.float32(np.sin(r_ang))
        cp = np.float32(np.cos(p_ang))
        sp = np.float32(np.sin(p_ang))
        R_level = np.array(
            [
                [cp, sp * sr, sp * cr],
                [0.0, cr, -sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float32,
        )
        d_level = R_level @ rays_base
        t_level = R_level @ t_cam_base

        dx = d_level[0, :]
        dy = d_level[1, :]
        dz = d_level[2, :]
        tx, ty, tz = float(t_level[0, 0]), float(t_level[1, 0]), float(t_level[2, 0])

        # Vectorized 14-step bisection ray-marching for exact intersection tz + zc * dz = f(tx + zc * dx, ty + zc * dy)
        lo = np.full_like(dx, 0.10, dtype=np.float32)
        hi = np.full_like(dx, float(self.max_depth_m), dtype=np.float32)
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            xw = tx + mid * dx
            yw = ty + mid * dy
            zw = tz + mid * dz
            z_surf = np.asarray(self.terrain_fn(xw, yw), dtype=np.float32)
            above = zw >= z_surf
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)

        zc = 0.5 * (lo + hi)
        xw_f = tx + zc * dx
        yw_f = ty + zc * dy
        zw_f = tz + zc * dz
        res_err = np.abs(zw_f - np.asarray(self.terrain_fn(xw_f, yw_f), dtype=np.float32))
        zc = np.where((zc > 0.15) & (zc < self.max_depth_m - 0.05) & (res_err < 0.015), zc, np.nan)
        return zc.reshape((h, w)).astype(np.float32)
