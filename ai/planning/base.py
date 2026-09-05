"""Core abstractions, typed representations, and status definitions for UGV motion planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import yaml

DEFAULT_PLANNING_CONFIG_PATH = Path("configs/planning.yaml")


class PlannerStatus(str, Enum):
    """Execution and diagnostic status for global/local planners and controllers."""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    FOLLOWING = "FOLLOWING"
    SUCCESS = "SUCCESS"
    NO_PATH = "NO_PATH"
    INVALID_START = "INVALID_START"
    INVALID_GOAL = "INVALID_GOAL"
    START_BLOCKED = "START_BLOCKED"
    GOAL_BLOCKED = "GOAL_BLOCKED"
    GOAL_REACHED = "GOAL_REACHED"
    NO_SAFE_TRAJECTORY = "LOCAL_NO_SAFE_TRAJECTORY"
    LOCAL_NO_SAFE_TRAJECTORY = "LOCAL_NO_SAFE_TRAJECTORY"
    MAP_UNAVAILABLE = "MAP_UNAVAILABLE"
    MAP_STALE = "MAP_STALE"
    LOCALIZATION_INVALID = "LOCALIZATION_INVALID"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    TIMEOUT = "TIMEOUT"
    INVALID_MAP = "INVALID_MAP"


@dataclass
class RobotState:
    """Robot kinematic state in world/odom or local map coordinates."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    omega: float = 0.0
    timestamp: float = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def pose(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.yaw)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.v, self.omega], dtype=np.float32)


@dataclass
class GoalPose:
    """Target navigation goal in world/odom coordinates (Phase 12 specification)."""
    x: float
    y: float
    yaw: Optional[float] = None
    frame_id: str = "odom"
    timestamp: float = 0.0
    xy_tolerance_m: float = 0.20
    yaw_tolerance_rad: float = 0.30

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def is_reached(self, state: Union[RobotState, Tuple[float, float, float]]) -> bool:
        """Check if robot state is within position and heading tolerances."""
        if isinstance(state, RobotState):
            rx, ry, ryaw = state.x, state.y, state.yaw
        else:
            rx, ry, ryaw = state[0], state[1], state[2]

        dist = float(np.hypot(rx - self.x, ry - self.y))
        if dist > self.xy_tolerance_m:
            return False
        if self.yaw is not None:
            dyaw = abs((ryaw - self.yaw + np.pi) % (2.0 * np.pi) - np.pi)
            if dyaw > self.yaw_tolerance_rad:
                return False
        return True


# Backward-compatible alias for GoalPose
@dataclass
class GoalState(GoalPose):
    """Target navigation goal with position and heading tolerances (legacy alias)."""
    pass


@dataclass
class VehicleKinematics:
    """Vehicle kinematic constraints, simulation dimensions, and safety margins."""
    length_m: float = 0.65
    width_m: float = 0.45
    wheelbase_m: float = 0.40
    track_width_m: float = 0.48
    wheel_radius_m: float = 0.125
    inscribed_radius_m: float = 0.25
    circumscribed_radius_m: float = 0.395
    inflation_radius_m: float = 0.45
    collision_margin_m: float = 0.05
    stop_margin_m: float = 0.30
    goal_margin_m: float = 0.15
    max_linear_velocity_mps: float = 1.00
    min_linear_velocity_mps: float = -0.20
    max_angular_velocity_radps: float = 1.00
    max_linear_acceleration_mps2: float = 1.50
    max_angular_acceleration_radps2: float = 2.50
    degraded_max_linear_velocity_mps: float = 0.35


@dataclass
class PlanningConfig:
    """Configuration bundle loaded from planning.yaml."""
    vehicle: VehicleKinematics = field(default_factory=VehicleKinematics)
    cost_weights: Dict[str, float] = field(default_factory=dict)
    unknown_policy: Dict[str, Any] = field(default_factory=dict)
    tolerances: Dict[str, float] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)
    astar: Dict[str, Any] = field(default_factory=dict)
    pf_rrt_star: Dict[str, Any] = field(default_factory=dict)
    hybrid_astar: Dict[str, Any] = field(default_factory=dict)
    dwa: Dict[str, Any] = field(default_factory=dict)
    pure_pursuit: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Union[str, Path] = DEFAULT_PLANNING_CONFIG_PATH) -> PlanningConfig:
        p = Path(config_path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        v_cfg = cfg.get("vehicle", {})
        vehicle = VehicleKinematics(
            length_m=float(v_cfg.get("length_m", 0.65)),
            width_m=float(v_cfg.get("width_m", 0.45)),
            wheelbase_m=float(v_cfg.get("wheelbase_m", 0.40)),
            track_width_m=float(v_cfg.get("track_width_m", 0.48)),
            wheel_radius_m=float(v_cfg.get("wheel_radius_m", 0.125)),
            inscribed_radius_m=float(v_cfg.get("inscribed_radius_m", 0.25)),
            circumscribed_radius_m=float(v_cfg.get("circumscribed_radius_m", 0.395)),
            inflation_radius_m=float(v_cfg.get("inflation_radius_m", 0.45)),
            collision_margin_m=float(v_cfg.get("collision_margin_m", 0.05)),
            stop_margin_m=float(v_cfg.get("stop_margin_m", 0.30)),
            goal_margin_m=float(v_cfg.get("goal_margin_m", 0.15)),
            max_linear_velocity_mps=float(v_cfg.get("max_linear_velocity_mps", 1.00)),
            min_linear_velocity_mps=float(v_cfg.get("min_linear_velocity_mps", -0.20)),
            max_angular_velocity_radps=float(v_cfg.get("max_angular_velocity_radps", 1.00)),
            max_linear_acceleration_mps2=float(v_cfg.get("max_linear_acceleration_mps2", 1.50)),
            max_angular_acceleration_radps2=float(v_cfg.get("max_angular_acceleration_radps2", 2.50)),
            degraded_max_linear_velocity_mps=float(v_cfg.get("degraded_max_linear_velocity_mps", 0.35)),
        )

        return cls(
            vehicle=vehicle,
            cost_weights=cfg.get("cost_weights", {}),
            unknown_policy=cfg.get("unknown_policy", {"unknown_is_lethal": False, "unknown_cost_penalty": 0.80}),
            tolerances=cfg.get("tolerances", {}),
            safety=cfg.get("safety", {}),
            astar=cfg.get("astar", {}),
            pf_rrt_star=cfg.get("pf_rrt_star", {}),
            hybrid_astar=cfg.get("hybrid_astar", {}),
            dwa=cfg.get("dwa", {}),
            pure_pursuit=cfg.get("pure_pursuit", {})
        )
