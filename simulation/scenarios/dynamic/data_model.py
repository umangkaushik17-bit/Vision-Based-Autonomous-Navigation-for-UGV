"""Data models for dynamic SIL obstacle scenarios and ground-truth specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ai.planning.dynamic.data_model import polygon_intersection_sat as _sat_core


def polygon_intersection_sat(poly1: Any, poly2: Any) -> bool:
    """Evaluate 2D convex polygon collision using the Separating Axis Theorem (SAT)."""
    p1 = np.asarray(poly1, dtype=np.float64)
    p2 = np.asarray(poly2, dtype=np.float64)
    return _sat_core(p1, p2)


class MotionModelType(str, Enum):
    """Supported kinematic motion models for simulated dynamic obstacles."""
    CONSTANT_VELOCITY = "CONSTANT_VELOCITY"
    CONSTANT_TURN_RATE = "CONSTANT_TURN_RATE"
    ACCELERATED = "ACCELERATED"
    WAYPOINT = "WAYPOINT"
    STATIONARY = "STATIONARY"


@dataclass
class GroundTruthObstacleState:
    """True physical state of a dynamic obstacle at simulation time t."""
    obstacle_id: str
    timestamp_s: float
    x: float
    y: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    length_m: float = 0.8
    width_m: float = 0.6
    is_active: bool = True

    @property
    def speed(self) -> float:
        """Scalar forward speed in m/s."""
        return math.hypot(self.vx, self.vy)

    def get_obb_corners(self) -> np.ndarray:
        """Compute the 4 corners of the 2D oriented bounding box in world frame.

        Returns:
            np.ndarray of shape (4, 2) in counter-clockwise order.
        """
        hl = self.length_m / 2.0
        hw = self.width_m / 2.0
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)

        # Local body corners: front-left, rear-left, rear-right, front-right
        local = np.array([
            [hl, hw],
            [-hl, hw],
            [-hl, -hw],
            [hl, -hw],
        ], dtype=np.float64)

        R = np.array([
            [cos_y, -sin_y],
            [sin_y, cos_y],
        ], dtype=np.float64)

        return (R @ local.T).T + np.array([self.x, self.y], dtype=np.float64)


@dataclass
class DynamicObstacleSpec:
    """Specification of an obstacle script in a dynamic scenario."""
    obstacle_id: str
    motion_model: MotionModelType = MotionModelType.CONSTANT_VELOCITY
    size: Tuple[float, float] = (0.8, 0.6)  # (length, width) in meters
    initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, yaw)
    initial_velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in m/s
    acceleration: Tuple[float, float] = (0.0, 0.0)  # (ax, ay) in m/s^2
    yaw_rate_rad_s: float = 0.0
    waypoints: List[Tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)
    spawn_time_s: float = 0.0
    despawn_time_s: float = float("inf")
    measurement_noise_std: float = 0.0
    dropout_probability: float = 0.0
    dropout_windows: List[Tuple[float, float]] = field(default_factory=list)  # [(t_start, t_end)]
    stale_timestamp_offset_s: float = 0.0
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if isinstance(self.motion_model, str):
            self.motion_model = MotionModelType(self.motion_model)
        if self.size[0] <= 0.0 or self.size[1] <= 0.0:
            raise ValueError(f"Obstacle size must be positive, got {self.size}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if not (0.0 <= self.dropout_probability <= 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1], got {self.dropout_probability}")
        if self.measurement_noise_std < 0.0:
            raise ValueError(f"Measurement noise std must be non-negative, got {self.measurement_noise_std}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obstacle_id": self.obstacle_id,
            "motion_model": self.motion_model.value,
            "size": list(self.size),
            "initial_pose": list(self.initial_pose),
            "initial_velocity": list(self.initial_velocity),
            "acceleration": list(self.acceleration),
            "yaw_rate_rad_s": self.yaw_rate_rad_s,
            "waypoints": [list(w) for w in self.waypoints],
            "spawn_time_s": self.spawn_time_s,
            "despawn_time_s": self.despawn_time_s,
            "measurement_noise_std": self.measurement_noise_std,
            "dropout_probability": self.dropout_probability,
            "dropout_windows": [list(w) for w in self.dropout_windows],
            "stale_timestamp_offset_s": self.stale_timestamp_offset_s,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DynamicObstacleSpec:
        return cls(
            obstacle_id=data["obstacle_id"],
            motion_model=MotionModelType(data.get("motion_model", "CONSTANT_VELOCITY")),
            size=tuple(data.get("size", (0.8, 0.6))),
            initial_pose=tuple(data.get("initial_pose", (0.0, 0.0, 0.0))),
            initial_velocity=tuple(data.get("initial_velocity", (0.0, 0.0))),
            acceleration=tuple(data.get("acceleration", (0.0, 0.0))),
            yaw_rate_rad_s=float(data.get("yaw_rate_rad_s", 0.0)),
            waypoints=[tuple(w) for w in data.get("waypoints", [])],
            spawn_time_s=float(data.get("spawn_time_s", 0.0)),
            despawn_time_s=float(data.get("despawn_time_s", float("inf"))),
            measurement_noise_std=float(data.get("measurement_noise_std", 0.0)),
            dropout_probability=float(data.get("dropout_probability", 0.0)),
            dropout_windows=[tuple(w) for w in data.get("dropout_windows", [])],
            stale_timestamp_offset_s=float(data.get("stale_timestamp_offset_s", 0.0)),
            confidence=float(data.get("confidence", 0.95)),
        )


@dataclass
class DynamicScenarioConfig:
    """Complete specification for a deterministic dynamic SIL scenario."""
    scenario_id: str
    name: str
    description: str = ""
    world_bounds_m: Tuple[Tuple[float, float], Tuple[float, float]] = ((-5.0, 20.0), (-8.0, 8.0))
    start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, yaw)
    goal_pose: Tuple[float, float, float] = (5.0, 0.0, 0.0)   # (x, y, yaw)
    obstacles: List[DynamicObstacleSpec] = field(default_factory=list)
    static_obstacles: List[Dict[str, Any]] = field(default_factory=list)
    control_dt_s: float = 0.10
    max_duration_s: float = 25.0
    random_seed: int = 42
    expected_outcome: str = "SUCCESS"
    goal_tolerance_m: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "world_bounds_m": [list(b) for b in self.world_bounds_m],
            "start_pose": list(self.start_pose),
            "goal_pose": list(self.goal_pose),
            "goal_tolerance_m": self.goal_tolerance_m,
            "obstacles": [o.to_dict() for o in self.obstacles],
            "static_obstacles": self.static_obstacles,
            "control_dt_s": self.control_dt_s,
            "max_duration_s": self.max_duration_s,
            "random_seed": self.random_seed,
            "expected_outcome": self.expected_outcome,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DynamicScenarioConfig:
        return cls(
            scenario_id=data["scenario_id"],
            name=data["name"],
            description=data.get("description", ""),
            world_bounds_m=tuple(tuple(b) for b in data.get("world_bounds_m", ((-5.0, 20.0), (-8.0, 8.0)))),
            start_pose=tuple(data.get("start_pose", (0.0, 0.0, 0.0))),
            goal_pose=tuple(data.get("goal_pose", (5.0, 0.0, 0.0))),
            obstacles=[DynamicObstacleSpec.from_dict(o) for o in data.get("obstacles", [])],
            static_obstacles=data.get("static_obstacles", []),
            control_dt_s=float(data.get("control_dt_s", 0.10)),
            max_duration_s=float(data.get("max_duration_s", 25.0)),
            random_seed=int(data.get("random_seed", 42)),
            expected_outcome=data.get("expected_outcome", "SUCCESS"),
            goal_tolerance_m=float(data.get("goal_tolerance_m", 0.25)),
        )

    def save_to_file(self, filepath: str) -> None:
        """Save scenario configuration to JSON file."""
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> DynamicScenarioConfig:
        """Load scenario configuration from JSON file."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
