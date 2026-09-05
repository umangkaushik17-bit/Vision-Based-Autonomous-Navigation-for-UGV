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
