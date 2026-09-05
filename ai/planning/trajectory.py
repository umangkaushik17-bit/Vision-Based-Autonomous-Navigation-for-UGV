"""Time-parameterized trajectory representations and state profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from ai.planning.base import RobotState
from ai.planning.path import Path, PathPoint


@dataclass
class TrajectoryPoint:
    """Discrete state along a time-parameterized motion profile."""
    t: float
    x: float
    y: float
    yaw: float = 0.0
    v: float = 0.0
    omega: float = 0.0
    a: float = 0.0
    alpha: float = 0.0
    cost: float = 0.0
    clearance_m: float = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_robot_state(self) -> RobotState:
        return RobotState(
            x=self.x, y=self.y, yaw=self.yaw,
            v=self.v, omega=self.omega, timestamp=self.t
        )


@dataclass
class Trajectory:
    """Sequence of time-parameterized trajectory states."""
    points: List[TrajectoryPoint] = field(default_factory=list)
    is_valid: bool = True
    total_cost: float = 0.0
    min_clearance_m: float = 0.0
    frame_id: str = "base_link"

    def __len__(self) -> int:
        return len(self.points)

    def is_empty(self) -> bool:
        return len(self.points) == 0

    @property
    def duration(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].t - self.points[0].t

    @property
    def end_state(self) -> Optional[RobotState]:
        if not self.points:
            return None
        return self.points[-1].to_robot_state()

    @property
    def trajectory_xy(self) -> np.ndarray:
        if not self.points:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([[p.x, p.y] for p in self.points], dtype=np.float32)

    def to_path(self) -> Path:
        """Convert trajectory into a geometric Path."""
        path_pts: List[PathPoint] = []
        cum_dist = 0.0
        for i, pt in enumerate(self.points):
            if i > 0:
                dx = pt.x - self.points[i-1].x
                dy = pt.y - self.points[i-1].y
                cum_dist += float(np.hypot(dx, dy))
            path_pts.append(PathPoint(
                x=pt.x, y=pt.y, yaw=pt.yaw, cost=pt.cost,
                distance_from_start=cum_dist, clearance_m=pt.clearance_m
            ))
        path = Path(points=path_pts, frame_id=self.frame_id, timestamp=self.points[0].t if self.points else 0.0)
        path.compute_curvature()
        return path
