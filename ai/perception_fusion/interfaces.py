"""Geometric primitives and data contracts for 3D terrain analysis and perception fusion.

Synthesizes concepts from MoveBase3D (SVD plane fitting, slope, flatness) and
UGV Nav4D (multi-layered surface terrain properties) into clean camera-first abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class TerrainCell:
    """Quantitative geometric characterization of a localized ground patch."""

    center_x: float  # Longitudinal coordinate relative to robot (meters)
    center_y: float  # Lateral coordinate relative to robot (meters)
    center_z: float  # Elevation coordinate relative to robot ground contact (meters)
    normal: np.ndarray  # Unit surface normal [nx, ny, nz], shape (3,)
    slope_deg: float  # Surface inclination relative to gravity/horizontal (degrees)
    flatness: float  # Surface roughness/coplanarity metric (0.0 = perfectly flat)
    height_variation: float  # Delta between maximum and minimum elevation in patch (meters)
    point_count: int  # Number of 3D points contributing to estimate (sparsity indicator)
    is_valid: bool = True

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        center_xy: Optional[Tuple[float, float]] = None
    ) -> TerrainCell:
        """Estimate terrain surface properties using SVD / Principal Component Analysis (MoveBase3D concept).

        Args:
            points: 3D point array (N, 3) representing ground samples within this cell.
            center_xy: Optional (x, y) cell center coordinate.

        Returns:
            TerrainCell populated with derived slope, flatness, and normal.
        """
        if points is None or len(points) < 3:
            cx, cy = center_xy if center_xy else (0.0, 0.0)
            return cls(
                center_x=cx,
                center_y=cy,
                center_z=0.0,
                normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                slope_deg=0.0,
                flatness=0.0,
                height_variation=0.0,
                point_count=0 if points is None else len(points),
                is_valid=False
            )

        pts = np.asarray(points, dtype=np.float32)
        centroid = np.mean(pts, axis=0)

        # Covariance matrix for SVD plane fitting
        centered = pts - centroid
        cov = (centered.T @ centered) / max(len(pts) - 1, 1)

        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Normal vector is eigenvector associated with smallest eigenvalue
        normal = eigenvectors[:, 0]
        # Ensure normal points upward (nz > 0)
        if normal[2] < 0.0:
            normal = -normal

        # Slope: angle between surface normal and vertical Z axis [0, 0, 1]
        nz = np.clip(normal[2], -1.0, 1.0)
        slope_rad = np.arccos(nz)
        slope_deg = float(np.degrees(slope_rad))

        # Flatness / roughness: residual variance normalized by total variance
        total_var = float(np.sum(eigenvalues))
        flatness = float(eigenvalues[0] / total_var) if total_var > 1e-7 else 0.0

        # Height variation
        z_min = float(np.min(pts[:, 2]))
        z_max = float(np.max(pts[:, 2]))
        height_var = z_max - z_min

        cx = float(center_xy[0]) if center_xy else float(centroid[0])
        cy = float(center_xy[1]) if center_xy else float(centroid[1])

        return cls(
            center_x=cx,
            center_y=cy,
            center_z=float(centroid[2]),
            normal=normal.astype(np.float32),
            slope_deg=slope_deg,
            flatness=flatness,
            height_variation=height_var,
            point_count=len(pts),
            is_valid=True
        )


@dataclass
class FusedTraversabilityMap:
    """Composite costmap combining learned traversability, terrain slope/geometry, and obstacle hazards."""

    cost_grid: np.ndarray  # (H_grid, W_grid) float32 in [0.0, 1.0], 0.0 = FREE, 1.0 = LETHAL
    confidence_grid: np.ndarray  # (H_grid, W_grid) float32 in [0.0, 1.0]
    unknown_mask: np.ndarray  # (H_grid, W_grid) bool
    resolution_m: float
    x_range_m: Tuple[float, float]
    y_range_m: Tuple[float, float]
    metadata: dict = field(default_factory=dict)
