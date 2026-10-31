"""Waypoint data model and spatial arrival condition checking for UGV missions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from ai.planning.base import GoalPose, RobotState


@dataclass
class Waypoint:
    """Represents a discrete mission waypoint in coordinate space."""

    id: str
    x: float
    y: float
    yaw: Optional[float] = None
    xy_tolerance_m: float = 0.35
    yaw_tolerance_rad: Optional[float] = None
    frame_id: str = "odom"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> Tuple[float, float]:
        """Return 2D coordinates (x, y)."""
        return (self.x, self.y)

    @property
    def pose(self) -> Tuple[float, float, Optional[float]]:
        """Return full pose (x, y, yaw)."""
        return (self.x, self.y, self.yaw)

    def distance_to(
        self,
        pose_or_state: Union[RobotState, Tuple[float, float], Tuple[float, float, float]],
    ) -> float:
        """Compute Euclidean distance from a robot pose or state to this waypoint."""
        if isinstance(pose_or_state, RobotState):
            rx, ry = pose_or_state.x, pose_or_state.y
        elif len(pose_or_state) >= 2:
            rx, ry = float(pose_or_state[0]), float(pose_or_state[1])
        else:
            raise ValueError(f"Invalid pose_or_state argument: {pose_or_state}")

        return float(math.hypot(rx - self.x, ry - self.y))

    def heading_error(
        self,
        pose_or_state: Union[RobotState, Tuple[float, float, float]],
    ) -> Optional[float]:
        """Compute wrapped absolute heading error if a target yaw is configured."""
        if self.yaw is None:
            return None

        if isinstance(pose_or_state, RobotState):
            ryaw = pose_or_state.yaw
        elif len(pose_or_state) >= 3:
            ryaw = float(pose_or_state[2])
        else:
            return None

        dyaw = abs((ryaw - self.yaw + math.pi) % (2.0 * math.pi) - math.pi)
        return float(dyaw)

    def is_reached(
        self,
        pose_or_state: Union[RobotState, Tuple[float, float], Tuple[float, float, float]],
    ) -> bool:
        """Check whether the given robot pose satisfies the waypoint arrival criteria.
        
        A waypoint is reached ONLY if:
        1. 2D Euclidean distance is <= xy_tolerance_m
        2. Heading error is <= yaw_tolerance_rad (if both target yaw and tolerance are configured).
        """
        dist = self.distance_to(pose_or_state)
        if dist > self.xy_tolerance_m:
            return False

        if self.yaw is not None and self.yaw_tolerance_rad is not None:
            herr = self.heading_error(pose_or_state)
            if herr is not None and herr > self.yaw_tolerance_rad:
                return False

        return True

    def validate(self) -> Tuple[bool, str]:
        """Validate internal consistency and numerical soundness of the waypoint."""
        if not self.id or not isinstance(self.id, str):
            return False, "Waypoint ID must be a non-empty string"

        if not math.isfinite(self.x) or not math.isfinite(self.y):
            return False, f"Waypoint {self.id} coordinates must be finite: ({self.x}, {self.y})"

        if self.yaw is not None and not math.isfinite(self.yaw):
            return False, f"Waypoint {self.id} yaw must be finite if specified: {self.yaw}"

        if not math.isfinite(self.xy_tolerance_m) or self.xy_tolerance_m <= 0.0:
            return False, f"Waypoint {self.id} xy_tolerance_m must be positive: {self.xy_tolerance_m}"

        if self.yaw_tolerance_rad is not None:
            if not math.isfinite(self.yaw_tolerance_rad) or self.yaw_tolerance_rad <= 0.0:
                return False, f"Waypoint {self.id} yaw_tolerance_rad must be positive: {self.yaw_tolerance_rad}"

        if not self.frame_id or not isinstance(self.frame_id, str):
            return False, f"Waypoint {self.id} frame_id must be a non-empty string"

        return True, "ok"

    def to_dict(self) -> Dict[str, Any]:
        """Convert waypoint to a serializable dictionary."""
        d: Dict[str, Any] = {
            "id": self.id,
            "x": float(self.x),
            "y": float(self.y),
            "frame_id": self.frame_id,
            "xy_tolerance_m": float(self.xy_tolerance_m),
        }
        if self.yaw is not None:
            d["yaw"] = float(self.yaw)
        if self.yaw_tolerance_rad is not None:
            d["yaw_tolerance_rad"] = float(self.yaw_tolerance_rad)
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_tolerance_m: float = 0.35) -> Waypoint:
        """Construct Waypoint instance from a dictionary."""
        wp_id = str(data.get("id", data.get("name", "WP")))
        x = float(data["x"])
        y = float(data["y"])
        yaw = float(data["yaw"]) if "yaw" in data and data["yaw"] is not None else None
        xy_tol = float(data.get("xy_tolerance_m", data.get("tolerance_m", default_tolerance_m)))
        yaw_tol = float(data["yaw_tolerance_rad"]) if "yaw_tolerance_rad" in data and data["yaw_tolerance_rad"] is not None else None
        frame_id = str(data.get("frame_id", "odom"))
        metadata = dict(data.get("metadata", {}))

        return cls(
            id=wp_id,
            x=x,
            y=y,
            yaw=yaw,
            xy_tolerance_m=xy_tol,
            yaw_tolerance_rad=yaw_tol,
            frame_id=frame_id,
            metadata=metadata,
        )

    def to_goal_pose(self) -> GoalPose:
        """Convert Waypoint to canonical Phase 12 GoalPose."""
        return GoalPose(
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            frame_id=self.frame_id,
            timestamp=0.0,
            xy_tolerance_m=self.xy_tolerance_m,
            yaw_tolerance_rad=self.yaw_tolerance_rad or 0.30,
        )
