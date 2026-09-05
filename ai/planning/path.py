"""Geometric path representations, interpolation, and trajectory curve metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class PathPoint:
    """Discrete waypoint along a planned geometric path."""
    x: float
    y: float
    yaw: float = 0.0
    cost: float = 0.0
    distance_from_start: float = 0.0
    curvature: float = 0.0
    clearance_m: float = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Path:
    """Canonical continuous or piecewise-linear path."""
    points: List[PathPoint] = field(default_factory=list)
    frame_id: str = "base_link"
    timestamp: float = 0.0
    is_smoothed: bool = False

    def __len__(self) -> int:
        return len(self.points)

    def is_empty(self) -> bool:
        return len(self.points) == 0

    @property
    def length(self) -> float:
        """Total cumulative geometric arc length in meters."""
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].distance_from_start

    @property
    def total_length_m(self) -> float:
        """Alias for cumulative path length in meters."""
        return self.length

    @property
    def waypoints_xy(self) -> np.ndarray:
        """Return (N, 2) numpy array of (x, y) coordinates."""
        if not self.points:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([[p.x, p.y] for p in self.points], dtype=np.float32)

    @classmethod
    def from_xy_array(
        cls,
        xy_coords: np.ndarray,
        frame_id: str = "base_link",
        timestamp: float = 0.0
    ) -> Path:
        """Construct a Path from an (N, 2) numpy array of positions."""
        if len(xy_coords) == 0:
            return cls(points=[], frame_id=frame_id, timestamp=timestamp)

        pts: List[PathPoint] = []
        cum_dist = 0.0
        n = len(xy_coords)

        for i in range(n):
            x, y = float(xy_coords[i, 0]), float(xy_coords[i, 1])
            if i > 0:
                dx = x - pts[-1].x
                dy = y - pts[-1].y
                cum_dist += float(np.hypot(dx, dy))

            # Estimate yaw
            if i < n - 1:
                yaw = float(np.arctan2(xy_coords[i+1, 1] - y, xy_coords[i+1, 0] - x))
            elif i > 0:
                yaw = pts[-1].yaw
            else:
                yaw = 0.0

            pts.append(PathPoint(
                x=x, y=y, yaw=yaw, distance_from_start=cum_dist
            ))

        path = cls(points=pts, frame_id=frame_id, timestamp=timestamp)
        path.compute_curvature()
        return path

    def compute_curvature(self) -> None:
        """Compute path curvature at each point using finite differences."""
        n = len(self.points)
        if n < 3:
            for p in self.points:
                p.curvature = 0.0
            return

        for i in range(1, n - 1):
            p_prev = self.points[i - 1]
            p_curr = self.points[i]
            p_next = self.points[i + 1]

            # Vector 1 and Vector 2
            v1 = np.array([p_curr.x - p_prev.x, p_curr.y - p_prev.y])
            v2 = np.array([p_next.x - p_curr.x, p_next.y - p_curr.y])

            d1 = np.linalg.norm(v1)
            d2 = np.linalg.norm(v2)

            if d1 < 1e-4 or d2 < 1e-4:
                p_curr.curvature = 0.0
                continue

            # Heading change
            cos_angle = np.clip(np.dot(v1, v2) / (d1 * d2), -1.0, 1.0)
            angle = float(np.arccos(cos_angle))
            ds = 0.5 * (d1 + d2)
            p_curr.curvature = float(angle / max(ds, 1e-4))

        self.points[0].curvature = self.points[1].curvature if n > 1 else 0.0
        self.points[-1].curvature = self.points[-2].curvature if n > 2 else 0.0

    def interpolate(self, step_m: float = 0.05) -> Path:
        """Resample the path with uniform spatial resolution."""
        if len(self.points) < 2:
            return self

        xy = self.waypoints_xy
        dists = np.array([p.distance_from_start for p in self.points])
        total_len = dists[-1]

        if total_len < step_m:
            return self

        num_samples = int(np.ceil(total_len / step_m)) + 1
        interp_dists = np.linspace(0.0, total_len, num_samples)

        interp_x = np.interp(interp_dists, dists, xy[:, 0])
        interp_y = np.interp(interp_dists, dists, xy[:, 1])

        new_xy = np.column_stack([interp_x, interp_y])
        return Path.from_xy_array(new_xy, frame_id=self.frame_id, timestamp=self.timestamp)
