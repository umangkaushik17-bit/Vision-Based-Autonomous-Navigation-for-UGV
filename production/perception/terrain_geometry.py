"""Vectorized 2.5D Terrain Geometry Estimator: Surface Normals, Slope, Roughness, Steps & Ditches (WS-1C / Subphase 15.2).

Computes geometric terrain attributes from a 2.5D elevation grid Z(x, y) in < 3.0 ms
for 200x200 grids on CPU using vectorized finite differences and 3x3 strided stencils:
  1. Unit Surface Normal: n(x, y) = [-dZ/dx, -dZ/dy, 1]^T / ||n||
  2. Terrain Slope Angle: alpha(x, y) = arccos(n_z) in degrees
  3. Boundary-Aware Finite Differences across `elevation_valid` masks (zero FOV-boundary cliff spikes)
  4. Positive Step Height Discontinuity: delta_Z+(x, y) >= max_step_height_m (0.12 m)
  5. Negative Ditch / Drop-Off Detection: delta_Z-(x, y) <= -max_ditch_depth_m (-0.12 m)
  6. Surface Roughness: local plane-residual dispersion sigma_z(x, y)
  7. Configured Hard Safety Gates (configs/terrain_safety.yaml):
     - Slopes >= critical_slope_deg (28.0 deg) -> Lethal (cost = 1.0, obstacle = True)
     - Positive steps >= max_step_height_m (0.12 m) -> Lethal high-centering hazard (cost = 1.0, obstacle = True)
     - Negative ditches <= -max_ditch_depth_m (-0.12 m) -> Lethal drop-off hazard (cost = 1.0, obstacle = True)
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from ai.spatial_fusion.world_map import WorldTerrainMap


@dataclass
class TerrainGeometryConfig:
    """Configuration thresholds matching configs/terrain_safety.yaml."""

    max_safe_slope_deg: float = 20.0
    critical_slope_deg: float = 28.0
    max_step_height_m: float = 0.12
    critical_step_height_m: float = 0.20
    ground_clearance_m: float = 0.12
    max_ditch_depth_m: float = 0.12
    critical_ditch_depth_m: float = 0.15
    w_slope: float = 0.35
    w_roughness: float = 0.20
    w_step: float = 0.25


def check_belly_clearance_grid(
    elevation_m: np.ndarray,
    resolution_m: float = 0.10,
    ground_clearance_m: float = 0.12,
    elevation_valid: Optional[np.ndarray] = None,
    dz_dx: Optional[np.ndarray] = None,
    dz_dy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate undercarriage high-centering crown height relative to surrounding wheel support plane.

    Compares the inner 3x3 chassis belly elevation against the outer 5x5 wheel support ring:
        h_crown(r, c) = Z_belly(r, c) - Z_wheels_mean(r, c) - smooth_slope_allowance
    where Z_wheels_mean is the mean over the 16 outer perimeter cells of the 5x5 footprint
    (and 7x7 outer ring for wider boulders/mounds).

    Returns:
        Tuple of (crown_height_m, lethal_belly_mask) of shape (H, W).
    """
    Z = np.ascontiguousarray(elevation_m, dtype=np.float32)
    h, w = Z.shape
    res = float(resolution_m)
    k3 = np.ones((3, 3), dtype=np.uint8)
    k5 = np.ones((5, 5), dtype=np.uint8)

    if dz_dx is None or dz_dy is None:
        inv_2res = np.float32(0.5 / res)
        dz_dx = np.zeros_like(Z)
        dz_dy = np.zeros_like(Z)
        if h >= 3:
            dz_dx[1:-1, :] = (Z[:-2, :] - Z[2:, :]) * inv_2res
        if w >= 3:
            dz_dy[:, 1:-1] = (Z[:, :-2] - Z[:, 2:]) * inv_2res

    # Inner 3x3 chassis belly peak and center elevation
    z_belly = cv2.dilate(Z, k3)
    s3 = np.float32(9.0) * cv2.blur(Z, (3, 3))
    s5 = np.float32(25.0) * cv2.blur(Z, (5, 5))
    s7 = np.float32(49.0) * cv2.blur(Z, (7, 7))

    # Outer 16-cell 5x5 wheel ring mean and outer 24-cell 7x7 wheel perimeter ring mean
    z_wheels_5x5 = (s5 - s3) * np.float32(1.0 / 16.0)
    z_wheels_7x7 = (s7 - s5) * np.float32(1.0 / 24.0)
    z_wheels = np.minimum(z_wheels_5x5, z_wheels_7x7)

    # Subtract linear plane slope rise across 1-cell belly half-width so smooth ramps have zero crown
    smooth_belly_rise = np.float32(res) * (np.abs(dz_dx) + np.abs(dz_dy))
    crown_height_m = np.maximum(np.float32(0.0), (z_belly - z_wheels) - smooth_belly_rise)

    if elevation_valid is not None:
        v_mask = np.asarray(elevation_valid, dtype=np.uint8)
        if not np.all(v_mask):
            # Require full 7x7 wheel-ring neighborhood validity to prevent FOV boundary false positives
            k7 = np.ones((7, 7), dtype=np.uint8)
            v_inner = cv2.erode(v_mask, k7) > 0
            crown_height_m = np.where(v_inner, crown_height_m, np.float32(0.0))

    lethal_belly_mask = crown_height_m >= np.float32(ground_clearance_m)
    return crown_height_m, lethal_belly_mask


@dataclass
class TerrainGeometryResult:
    """Computed 2.5D geometric layers and execution timing."""

    dz_dx: np.ndarray               # (H, W) float32 gradient dZ/dx
    dz_dy: np.ndarray               # (H, W) float32 gradient dZ/dy
    nz: np.ndarray                  # (H, W) float32 unit normal Z
    slope_deg: np.ndarray           # (H, W) float32 slope angle in degrees
    step_height_m: np.ndarray       # (H, W) float32 local 3x3 vertical step discontinuity
    roughness_m: np.ndarray         # (H, W) float32 local 3x3 residual roughness
    lethal_slope_mask: np.ndarray   # (H, W) bool
    lethal_step_mask: np.ndarray    # (H, W) bool
    compute_ms: float
    negative_step_mask: Optional[np.ndarray] = None  # (H, W) bool, True where ditch drop <= -max_ditch_depth_m
    crown_height_m: Optional[np.ndarray] = None      # (H, W) float32 undercarriage crown height
    lethal_belly_mask: Optional[np.ndarray] = None   # (H, W) bool, True where crown_height_m >= ground_clearance_m
    _ditch_depth_cached: Optional[np.ndarray] = None
    _normals_cached: Optional[np.ndarray] = None

    @property
    def ditch_depth_m(self) -> np.ndarray:
        """Lazily compute (H, W) signed negative drop depth (<= 0.0 m) when requested."""
        if self._ditch_depth_cached is None:
            if self.negative_step_mask is not None and np.any(self.negative_step_mask):
                self._ditch_depth_cached = np.where(
                    self.negative_step_mask, -self.step_height_m, np.float32(0.0)
                )
            else:
                self._ditch_depth_cached = np.zeros_like(self.step_height_m)
        return self._ditch_depth_cached

    @property
    def normals(self) -> np.ndarray:
        """Lazily stack (H, W, 3) unit surface normal tensor when requested."""
        if self._normals_cached is None:
            nx = -self.dz_dx * self.nz
            ny = -self.dz_dy * self.nz
            self._normals_cached = np.stack([nx, ny, self.nz], axis=-1)
        return self._normals_cached


class VectorizedTerrainGeometryEstimator:
    """High-speed vectorized 2.5D terrain normal, slope, roughness, step, and ditch estimator."""

    def __init__(self, config: Optional[TerrainGeometryConfig] = None) -> None:
        self.config = config or TerrainGeometryConfig()
        self._pad_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._vpad_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._k3 = np.ones((3, 3), dtype=np.uint8)
        self._k5 = np.ones((5, 5), dtype=np.uint8)

    def check_belly_clearance_grid(
        self,
        elevation_m: np.ndarray,
        resolution_m: float = 0.10,
        elevation_valid: Optional[np.ndarray] = None,
        dz_dx: Optional[np.ndarray] = None,
        dz_dy: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate undercarriage high-centering crown height relative to surrounding wheel support plane."""
        return check_belly_clearance_grid(
            elevation_m=elevation_m,
            resolution_m=resolution_m,
            ground_clearance_m=self.config.ground_clearance_m,
            elevation_valid=elevation_valid,
            dz_dx=dz_dx,
            dz_dy=dz_dy,
        )

    def estimate_from_elevation(
        self,
        elevation_m: np.ndarray,
        resolution_m: float = 0.10,
        elevation_valid: Optional[np.ndarray] = None,
    ) -> TerrainGeometryResult:
        """Compute surface normals, slope, step height, negative ditches, and roughness from Z(r, c).

        When `elevation_valid` is provided with unobserved cells, finite differences and
        morphological extrema are strictly restricted to observed `elevation_valid` neighbors,
        preventing artificial slope/step/ditch spikes at the FOV boundary.
        """
        t0 = time.perf_counter()
        Z = np.ascontiguousarray(elevation_m, dtype=np.float32)
        h, w = Z.shape
        res = float(resolution_m)
        inv_res = np.float32(1.0 / res)
        inv_2res = np.float32(0.5 / res)

        has_partial_valid = False
        valid_mask: Optional[np.ndarray] = None
        if elevation_valid is not None:
            valid_mask = np.asarray(elevation_valid, dtype=bool)
            if not np.all(valid_mask):
                has_partial_valid = True

        # Preallocated padded elevation array
        shape_key = (h, w)
        Z_pad = self._pad_cache.get(shape_key)
        if Z_pad is None:
            Z_pad = np.empty((h + 2, w + 2), dtype=np.float32)
            self._pad_cache[shape_key] = Z_pad

        Z_pad[1 : h + 1, 1 : w + 1] = Z
        Z_pad[0, 1 : w + 1] = Z[0, :]
        Z_pad[h + 1, 1 : w + 1] = Z[-1, :]
        Z_pad[:, 0] = Z_pad[:, 1]
        Z_pad[:, w + 1] = Z_pad[:, w]

        n01 = Z_pad[0:h, 1 : w + 1]
        n10 = Z_pad[1 : h + 1, 0:w]
        n11 = Z_pad[1 : h + 1, 1 : w + 1]
        n12 = Z_pad[1 : h + 1, 2 : w + 2]
        n21 = Z_pad[2 : h + 2, 1 : w + 1]

        if not has_partial_valid:
            # Fast full-grid path (all cells valid)
            dz_dx = (n01 - n21) * inv_2res
            if h >= 2:
                dz_dx[0, :] = (Z[0, :] - Z[1, :]) * inv_res
                dz_dx[-1, :] = (Z[-2, :] - Z[-1, :]) * inv_res

            dz_dy = (n10 - n12) * inv_2res
            if w >= 2:
                dz_dy[:, 0] = (Z[:, 0] - Z[:, 1]) * inv_res
                dz_dy[:, -1] = (Z[:, -2] - Z[:, -1]) * inv_res

            grad_sq = dz_dx * dz_dx + dz_dy * dz_dy
            nz = np.float32(1.0) / np.sqrt(grad_sq + np.float32(1.0))
            slope_deg = np.arccos(nz) * np.float32(57.2957795)

            z_max_3x3 = cv2.dilate(Z, self._k3)
            z_min_3x3 = cv2.erode(Z, self._k3)
            raw_span = z_max_3x3 - z_min_3x3
            smooth_plane_span = np.float32(2.0 * res) * (np.abs(dz_dx) + np.abs(dz_dy))
            step_mag = np.where(
                grad_sq > np.float32(0.50),
                raw_span,
                np.maximum(np.float32(0.0), raw_span - smooth_plane_span),
            )

            roughness_m = np.abs((n01 + n10 + n12 + n21) - np.float32(4.0) * n11) * np.float32(0.25)
            z_ambient = None
        else:
            # Boundary-aware path: restrict gradients, curvature, and extrema to elevation_valid cells
            if not np.any(valid_mask):
                zeros_f = np.zeros((h, w), dtype=np.float32)
                ones_f = np.ones((h, w), dtype=np.float32)
                zeros_b = np.zeros((h, w), dtype=bool)
                return TerrainGeometryResult(
                    dz_dx=zeros_f,
                    dz_dy=zeros_f.copy(),
                    nz=ones_f,
                    slope_deg=zeros_f.copy(),
                    step_height_m=zeros_f.copy(),
                    roughness_m=zeros_f.copy(),
                    lethal_slope_mask=zeros_b,
                    lethal_step_mask=zeros_b.copy(),
                    negative_step_mask=zeros_b.copy(),
                    compute_ms=(time.perf_counter() - t0) * 1000.0,
                )

            V_pad = self._vpad_cache.get(shape_key)
            if V_pad is None:
                V_pad = np.zeros((h + 2, w + 2), dtype=bool)
                self._vpad_cache[shape_key] = V_pad
            else:
                V_pad.fill(False)
            V_pad[1 : h + 1, 1 : w + 1] = valid_mask

            v01 = V_pad[0:h, 1 : w + 1]
            v10 = V_pad[1 : h + 1, 0:w]
            v11 = V_pad[1 : h + 1, 1 : w + 1]
            v12 = V_pad[1 : h + 1, 2 : w + 2]
            v21 = V_pad[2 : h + 2, 1 : w + 1]

            n01_eff = np.where(v01, n01, n11)
            n21_eff = np.where(v21, n21, n11)
            both_x = v01 & v21 & v11
            one_x = (v01 ^ v21) & v11
            scale_x = np.where(both_x, inv_2res, np.where(one_x, inv_res, np.float32(0.0)))
            dz_dx = (n01_eff - n21_eff) * scale_x

            n10_eff = np.where(v10, n10, n11)
            n12_eff = np.where(v12, n12, n11)
            both_y = v10 & v12 & v11
            one_y = (v10 ^ v12) & v11
            scale_y = np.where(both_y, inv_2res, np.where(one_y, inv_res, np.float32(0.0)))
            dz_dy = (n10_eff - n12_eff) * scale_y

            grad_sq = dz_dx * dz_dx + dz_dy * dz_dy
            nz = np.where(v11, np.float32(1.0) / np.sqrt(grad_sq + np.float32(1.0)), np.float32(1.0))
            slope_deg = np.where(
                v11,
                np.arccos(np.clip(nz, -1.0, 1.0)) * np.float32(57.2957795),
                np.float32(0.0),
            )

            # Mask unobserved cells with sentinel values before 3x3 max/min morphology
            z_for_max = np.where(v11, Z, np.float32(-1e6))
            z_for_min = np.where(v11, Z, np.float32(1e6))
            z_max_3x3 = np.where(v11, cv2.dilate(z_for_max, self._k3), np.float32(0.0))
            z_min_3x3 = np.where(v11, cv2.erode(z_for_min, self._k3), np.float32(0.0))

            raw_span = np.where(v11, np.maximum(np.float32(0.0), z_max_3x3 - z_min_3x3), np.float32(0.0))
            smooth_plane_span = np.float32(2.0 * res) * (np.abs(dz_dx) + np.abs(dz_dy))
            step_mag = np.where(
                v11,
                np.where(
                    grad_sq > np.float32(0.50),
                    raw_span,
                    np.maximum(np.float32(0.0), raw_span - smooth_plane_span),
                ),
                np.float32(0.0),
            )

            # Normalized box blur over valid cells for ambient surface
            v_f = v11.astype(np.float32)
            z_sum_amb = cv2.blur(np.where(v11, Z, np.float32(0.0)), (31, 31))
            v_sum_amb = np.maximum(np.float32(1e-5), cv2.blur(v_f, (31, 31)))
            z_ambient = np.where(v11, z_sum_amb / v_sum_amb, np.float32(0.0))

            lap_x = np.where(both_x, (n01 + n21) - np.float32(2.0) * n11, np.float32(0.0))
            lap_y = np.where(both_y, (n10 + n12) - np.float32(2.0) * n11, np.float32(0.0))
            roughness_m = np.abs(lap_x + lap_y) * np.float32(0.25)

        # 5. Distinguish Positive Steps (delta_Z+) vs Negative Ditches (delta_Z-)
        step_height_m = step_mag
        lethal_slope = slope_deg >= self.config.critical_slope_deg
        ditch_cand = (step_height_m >= self.config.max_ditch_depth_m) & (z_min_3x3 < np.float32(-0.02))
        if np.any(ditch_cand):
            if z_ambient is None:
                z_ambient = cv2.blur(Z, (31, 31))
            negative_step = ditch_cand & (
                z_ambient > np.float32(0.56) * z_max_3x3 + np.float32(0.44) * z_min_3x3 + np.float32(0.005)
            )
            lethal_step = (step_height_m >= self.config.max_step_height_m) & (~negative_step)
        else:
            negative_step = ditch_cand
            lethal_step = step_height_m >= self.config.max_step_height_m

        compute_ms = (time.perf_counter() - t0) * 1000.0
        return TerrainGeometryResult(
            dz_dx=dz_dx,
            dz_dy=dz_dy,
            nz=nz,
            slope_deg=slope_deg,
            step_height_m=step_height_m,
            roughness_m=roughness_m,
            lethal_slope_mask=lethal_slope,
            lethal_step_mask=lethal_step,
            negative_step_mask=negative_step,
            compute_ms=compute_ms,
        )

    def update_world_map_from_elevation(
        self,
        world_map: WorldTerrainMap,
        elevation_m: Optional[np.ndarray] = None,
        elevation_valid: Optional[np.ndarray] = None,
    ) -> TerrainGeometryResult:
        """Populate WorldTerrainMap 2.5D geometric layers and enforce lethal slope/step/ditch cost gates."""
        if elevation_m is None:
            elev_input = world_map.elevation
            valid_input = elevation_valid if elevation_valid is not None else (
                world_map.elevation_valid if np.any(world_map.elevation_valid) else None
            )
        else:
            elev_input = np.asarray(elevation_m, dtype=np.float32)
            world_map.elevation = elev_input.copy()
            if elevation_valid is not None:
                valid_input = np.asarray(elevation_valid, dtype=bool)
                world_map.elevation_valid = valid_input.copy()
            elif np.any(world_map.elevation_valid):
                valid_input = world_map.elevation_valid
            else:
                valid_input = None
                world_map.elevation_valid.fill(True)

        res_geom = self.estimate_from_elevation(
            elev_input,
            resolution_m=world_map.resolution_m,
            elevation_valid=valid_input,
        )
        crown_h, lethal_belly = self.check_belly_clearance_grid(
            elev_input,
            resolution_m=world_map.resolution_m,
            elevation_valid=valid_input,
            dz_dx=res_geom.dz_dx,
            dz_dy=res_geom.dz_dy,
        )
        res_geom.crown_height_m = crown_h
        res_geom.lethal_belly_mask = lethal_belly

        world_map.surface_normal = res_geom.normals
        world_map.slope = res_geom.slope_deg
        world_map.roughness = res_geom.roughness_m
        world_map.height_var = np.maximum(res_geom.step_height_m, crown_h)

        # Continuous terrain cost contribution on non-lethal cells
        slope_norm = np.clip(res_geom.slope_deg / max(1.0, self.config.max_safe_slope_deg), 0.0, 2.0)
        step_norm = np.clip(world_map.height_var / max(1e-3, self.config.max_step_height_m), 0.0, 2.0)
        rough_norm = np.clip(res_geom.roughness_m / 0.05, 0.0, 2.0)

        geom_cost = (
            self.config.w_slope * slope_norm
            + self.config.w_step * step_norm
            + self.config.w_roughness * rough_norm
        ).astype(np.float32)

        # Combine with existing semantic/base cost without diluting hazards
        world_map.cost = np.clip(np.maximum(world_map.cost, geom_cost), 0.0, 1.0)

        # Hard Lethal Gates: Steep slopes (>= 28 deg), positive steps (>= 0.12 m),
        # negative ditches (<= -0.12 m), and undercarriage high-centering crowns (>= 0.12 m)
        lethal_geom = res_geom.lethal_slope_mask | res_geom.lethal_step_mask | lethal_belly
        if res_geom.negative_step_mask is not None:
            lethal_geom = lethal_geom | res_geom.negative_step_mask

        if np.any(lethal_geom):
            world_map.obstacle[lethal_geom] = True
            world_map.cost[lethal_geom] = 1.0
            world_map.hazard[lethal_geom] = 1.0
            world_map.traversability[lethal_geom] = 0.0

        return res_geom
