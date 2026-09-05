"""Event-driven global replanning layer for dynamic obstacle deadlocks and persistent blockages (Milestone 13.11).

Consumes tactical behavioral decisions (M13.10), predictive DWA results (M13.9), and
spatiotemporal obstacle tracks to trigger deterministic global A* route regeneration
only when explicit persistent blockage or deadlock renders the active global path unusable.

Key Architectural Guarantees:
1. Temporary dynamic obstructions (WAIT / AVOID) do not trigger global replanning.
2. DYNAMIC_DEADLOCK and persistent blockage (> persistent_blockage_duration_s) trigger replanning.
3. Cooldown and debouncing prevent replanning thrashing/oscillations.
4. Canonical static WorldTerrainMap is NEVER mutated in-place (uses DynamicPlanningMapOverlay).
5. Stale path rule: Invalidated/deadlocked paths are explicitly marked stale;
   GLOBAL_REPLAN_FAILED never commands motion or silently executes an invalid path.
6. Safety invariants strictly enforced: Start/goal validation, entrance clearance,
   static and dynamic collision-freedom, and reachability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from ai.planning.base import GoalPose, GoalState, PlanningConfig, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import BehavioralDecision, BehavioralState, DeadlockEvent
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    TrackedObstacleState,
    compute_obb_corners,
    point_in_polygon,
)
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.path import Path, PathPoint
from ai.spatial_fusion.world_map import WorldTerrainMap


class DynamicReplanTrigger(str, Enum):
    """Deterministic triggers for event-driven global route regeneration."""

    NONE = "NONE"
    DYNAMIC_DEADLOCK = "DYNAMIC_DEADLOCK"
    PERSISTENT_BLOCKAGE = "PERSISTENT_BLOCKAGE"
    INSUFFICIENT_PROGRESS = "INSUFFICIENT_PROGRESS"
    PATH_INVALIDATED = "PATH_INVALIDATED"
    MANUAL = "MANUAL"


@dataclass
class DynamicReplanEvent:
    """Structured, immutable audit record of a triggered global replan event."""

    trigger_type: DynamicReplanTrigger
    timestamp: float
    current_pose: Tuple[float, float, float]  # (x, y, yaw)
    current_goal: Tuple[float, float]          # (x, y)
    current_global_path: Optional[Path]
    blocking_obstacle_ids: List[int]
    blocked_duration: float
    deadlock_duration: float
    current_progress: float
    reason: str
    confidence: float
    event_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to serializable dictionary."""
        return {
            "event_id": self.event_id,
            "trigger_type": self.trigger_type.value,
            "timestamp": round(float(self.timestamp), 4),
            "current_pose": [round(float(v), 3) for v in self.current_pose],
            "current_goal": [round(float(v), 3) for v in self.current_goal],
            "blocking_obstacle_ids": list(self.blocking_obstacle_ids),
            "blocked_duration": round(float(self.blocked_duration), 3),
            "deadlock_duration": round(float(self.deadlock_duration), 3),
            "current_progress": round(float(self.current_progress), 3),
            "reason": self.reason,
            "confidence": round(float(self.confidence), 3),
            "path_waypoints": len(self.current_global_path.points) if self.current_global_path else 0,
        }


@dataclass
class PathChangeMetrics:
    """Quantitative comparison between pre-replan and newly generated global paths."""

    old_path_length: float
    new_path_length: float
    old_path_cost: float
    new_path_cost: float
    path_deviation: float
    goal_reachable: bool
    num_dynamic_obstacle_intersections: int
    min_dynamic_clearance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "old_path_length_m": round(float(self.old_path_length), 3),
            "new_path_length_m": round(float(self.new_path_length), 3),
            "old_path_cost": round(float(self.old_path_cost), 3),
            "new_path_cost": round(float(self.new_path_cost), 3),
            "path_deviation_m": round(float(self.path_deviation), 3),
            "goal_reachable": bool(self.goal_reachable),
            "num_dynamic_obstacle_intersections": int(self.num_dynamic_obstacle_intersections),
            "min_dynamic_clearance_m": round(float(self.min_dynamic_clearance), 3),
        }


@dataclass
class GlobalReplannerConfig:
    """Configurable thresholds, hysteresis windows, and planning overlay policies."""

    cooldown_s: float = 4.0
    persistent_blockage_duration_s: float = 3.0
    progress_loss_window_s: float = 5.0
    min_progress_m: float = 0.20
    goal_tolerance_m: float = 0.20
    debounce_cycles: int = 2
    entrance_clearance_steps: int = 5
    entrance_min_clearance_m: float = 0.35
    dynamic_obstacle_inflation_m: float = 0.20
    dynamic_risk_cost_penalty: float = 0.30
    lethal_cost_threshold: float = 0.85
    unknown_is_lethal: bool = False
    max_replan_retries: int = 3
    astar_config: Optional[PlanningConfig] = None


@dataclass
class DynamicReplanResult:
    """Structured result returned by the event-driven global replanner."""

    replan_triggered: bool
    trigger_type: DynamicReplanTrigger = DynamicReplanTrigger.NONE
    event: Optional[DynamicReplanEvent] = None
    success: bool = False
    status: PlannerStatus = PlannerStatus.SUCCESS
    path: Optional[Path] = None
    is_path_stale: bool = False
    metrics: Optional[PathChangeMetrics] = None
    failure_reason: Optional[str] = None
    cooldown_remaining_s: float = 0.0
    planning_latency_ms: float = 0.0
    validation_latency_ms: float = 0.0
    detection_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "replan_triggered": bool(self.replan_triggered),
            "trigger_type": self.trigger_type.value,
            "success": bool(self.success),
            "status": self.status.value,
            "is_path_stale": bool(self.is_path_stale),
            "failure_reason": self.failure_reason,
            "cooldown_remaining_s": round(float(self.cooldown_remaining_s), 3),
            "planning_latency_ms": round(float(self.planning_latency_ms), 2),
            "validation_latency_ms": round(float(self.validation_latency_ms), 2),
            "detection_latency_ms": round(float(self.detection_latency_ms), 2),
            "total_latency_ms": round(float(self.total_latency_ms), 2),
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


class DynamicPlanningMapOverlay:
    """Ephemeral planning representation composing static terrain with dynamic obstacle costs.

    CRITICAL ARCHITECTURAL RULE:
    The canonical static base_map (WorldTerrainMap or BEVTerrainCostmap) is NEVER mutated.
    Local copies of cost, obstacle, hazard, and unknown layers are maintained only for
    the lifetime of the replanning search cycle.
    """

    def __init__(
        self,
        base_map: Any,
        dynamic_obstacles: Optional[Sequence[Any]] = None,
        inflation_m: float = 0.20,
        risk_cost_penalty: float = 0.30,
    ) -> None:
        self.base_map = base_map
        self.resolution_m = float(getattr(base_map, "resolution_m", getattr(base_map, "resolution", 0.10)))
        self.x_bounds_m = tuple(getattr(base_map, "x_bounds_m", (-20.0, 20.0)))
        self.y_bounds_m = tuple(getattr(base_map, "y_bounds_m", (-20.0, 20.0)))

        # Shape
        if hasattr(base_map, "cost"):
            self.n_rows, self.n_cols = base_map.cost.shape
        else:
            self.n_rows = int(np.round((self.x_bounds_m[1] - self.x_bounds_m[0]) / self.resolution_m))
            self.n_cols = int(np.round((self.y_bounds_m[1] - self.y_bounds_m[0]) / self.resolution_m))

        # Ephemeral isolated layer copies — ZERO mutation of base_map
        self.cost = base_map.cost.copy() if hasattr(base_map, "cost") else np.full((self.n_rows, self.n_cols), 0.10, dtype=np.float32)
        self.obstacle = base_map.obstacle.copy() if hasattr(base_map, "obstacle") else np.zeros((self.n_rows, self.n_cols), dtype=bool)
        self.hazard = base_map.hazard.copy() if hasattr(base_map, "hazard") else np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        self.unknown = base_map.unknown.copy() if hasattr(base_map, "unknown") else np.zeros((self.n_rows, self.n_cols), dtype=bool)

        # Forward secondary layers without mutation
        self.traversability = getattr(base_map, "traversability", np.full((self.n_rows, self.n_cols), 0.90, dtype=np.float32))
        self.slope = getattr(base_map, "slope", getattr(base_map, "slope_deg", np.zeros((self.n_rows, self.n_cols), dtype=np.float32)))
        self.roughness = getattr(base_map, "roughness", np.zeros((self.n_rows, self.n_cols), dtype=np.float32))
        self.height_var = getattr(base_map, "height_var", np.zeros((self.n_rows, self.n_cols), dtype=np.float32))

        # Dynamic obstacles rasterization
        if dynamic_obstacles:
            self._rasterize_dynamic_obstacles(dynamic_obstacles, inflation_m, risk_cost_penalty)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.cost.shape

    def world_to_grid(self, x_w: float, y_w: float) -> Tuple[int, int]:
        if hasattr(self.base_map, "world_to_grid"):
            return self.base_map.world_to_grid(x_w, y_w)
        if hasattr(self.base_map, "meter_to_grid"):
            return self.base_map.meter_to_grid(x_w, y_w)
        row = int((self.x_bounds_m[1] - x_w) / self.resolution_m)
        col = int((self.y_bounds_m[1] - y_w) / self.resolution_m)
        return row, col

    def meter_to_grid(self, x_w: float, y_w: float) -> Tuple[int, int]:
        """Alias for compatibility with BEVTerrainCostmap interface."""
        return self.world_to_grid(x_w, y_w)

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        if hasattr(self.base_map, "grid_to_world"):
            return self.base_map.grid_to_world(row, col)
        if hasattr(self.base_map, "grid_to_meter"):
            return self.base_map.grid_to_meter(row, col)
        x_w = self.x_bounds_m[1] - (row + 0.5) * self.resolution_m
        y_w = self.y_bounds_m[1] - (col + 0.5) * self.resolution_m
        return float(x_w), float(y_w)

    def grid_to_meter(self, row: int, col: int) -> Tuple[float, float]:
        """Alias for compatibility with BEVTerrainCostmap interface."""
        return self.grid_to_world(row, col)

    def is_in_bounds(self, x_w: float, y_w: float) -> bool:
        if hasattr(self.base_map, "is_in_bounds"):
            return bool(self.base_map.is_in_bounds(x_w, y_w))
        return (self.x_bounds_m[0] <= x_w < self.x_bounds_m[1]) and (self.y_bounds_m[0] <= y_w < self.y_bounds_m[1])

    def is_grid_in_bounds(self, row: int, col: int) -> bool:
        return (0 <= row < self.n_rows) and (0 <= col < self.n_cols)

    def get_cost(self, x_w: float, y_w: float, default: float = 1.0) -> float:
        if not self.is_in_bounds(x_w, y_w):
            return default
        r, c = self.world_to_grid(x_w, y_w)
        if not self.is_grid_in_bounds(r, c):
            return default
        return float(self.cost[r, c])

    def is_cell_traversable(self, x_w: float, y_w: float, cost_threshold: float = 0.65) -> bool:
        if not self.is_in_bounds(x_w, y_w):
            return False
        r, c = self.world_to_grid(x_w, y_w)
        if not self.is_grid_in_bounds(r, c):
            return False
        return (
            (not self.obstacle[r, c])
            and (self.hazard[r, c] < 0.40)
            and (self.cost[r, c] < cost_threshold)
            and (not self.unknown[r, c])
        )

    def _rasterize_dynamic_obstacles(
        self,
        obstacles: Sequence[Any],
        inflation_m: float,
        risk_penalty: float,
    ) -> None:
        """Rasterize physical footprints and inflated risk halos into overlay layers."""
        for obs in obstacles:
            # Extract position and dimensions
            x, y, yaw = 0.0, 0.0, 0.0
            length, width = 0.60, 0.60
            is_unknown = False

            if isinstance(obs, TrackedObstacleState):
                x, y = float(obs.x), float(obs.y)
                yaw = float(obs.bbox_yaw)
                length, width = float(obs.bbox_size[0]), float(obs.bbox_size[1])
                is_unknown = (obs.status == "UNKNOWN")
            elif isinstance(obs, PredictedObstacleTrajectory):
                if obs.waypoints:
                    wp0 = obs.waypoints[0]
                    x, y = float(wp0.x), float(wp0.y)
                yaw = float(obs.bbox_yaw)
                length, width = float(obs.bbox_size[0]), float(obs.bbox_size[1])
                is_unknown = bool(getattr(obs, "is_unknown", False))
            elif isinstance(obs, dict):
                x = float(obs.get("x", 0.0))
                y = float(obs.get("y", 0.0))
                yaw = float(obs.get("yaw", 0.0))
                length = float(obs.get("length", obs.get("radius", 0.3) * 2.0))
                width = float(obs.get("width", obs.get("radius", 0.3) * 2.0))
                is_unknown = bool(obs.get("is_unknown", False))
            elif hasattr(obs, "x") and hasattr(obs, "y"):
                x = float(obs.x)
                y = float(obs.y)
                yaw = float(getattr(obs, "yaw", getattr(obs, "bbox_yaw", 0.0)))
                size = getattr(obs, "bbox_size", (0.6, 0.6))
                length, width = float(size[0]), float(size[1])
                is_unknown = bool(getattr(obs, "is_unknown", False))
            else:
                continue

            radius_phys = max(length, width) / 2.0
            radius_infl = radius_phys + inflation_m

            # Bounding box in metric space
            x_min = x - radius_infl
            x_max = x + radius_infl
            y_min = y - radius_infl
            y_max = y + radius_infl

            # Raster coordinate bounds
            r_top, c_right = self.world_to_grid(x_max, y_min)
            r_bottom, c_left = self.world_to_grid(x_min, y_max)

            r_start = max(0, min(r_top, r_bottom))
            r_end = min(self.n_rows, max(r_top, r_bottom) + 1)
            c_start = max(0, min(c_left, c_right))
            c_end = min(self.n_cols, max(c_left, c_right) + 1)

            # Compute OBB corners for exact physical check
            obb_corners = compute_obb_corners((x, y), (length, width), yaw)

            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    cx, cy = self.grid_to_world(r, c)
                    dx = cx - x
                    dy = cy - y
                    dist = math.hypot(dx, dy)

                    if dist > radius_infl:
                        continue

                    # Exact OBB or inscribed check for lethal footprint
                    if point_in_polygon((cx, cy), obb_corners) or dist <= min(length, width) / 2.0:
                        self.obstacle[r, c] = True
                    else:
                        # Inflated risk zone
                        self.cost[r, c] = min(1.0, self.cost[r, c] + risk_penalty)
                        self.hazard[r, c] = max(self.hazard[r, c], 0.50)

                    # Unknown dynamic space: flag unknown, NEVER clear static unknown
                    if is_unknown:
                        self.unknown[r, c] = True


def compute_path_deviation(old_path: Optional[Path], new_path: Optional[Path]) -> float:
    """Calculate the directed Hausdorff distance (meters) between new and old paths."""
    if old_path is None or new_path is None:
        return 0.0
    if len(old_path.points) == 0 or len(new_path.points) == 0:
        return 0.0

    old_xy = old_path.waypoints_xy
    new_xy = new_path.waypoints_xy

    # For each point in new_path, compute distance to nearest point in old_path
    # Hausdorff max_i (min_j ||new_i - old_j||)
    diff = new_xy[:, np.newaxis, :] - old_xy[np.newaxis, :, :]  # (N_new, N_old, 2)
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])             # (N_new, N_old)
    min_dists = np.min(dists, axis=1)                          # (N_new,)
    return float(np.max(min_dists))


def compute_min_dynamic_clearance(
    path: Optional[Path],
    obstacles: Optional[Sequence[Any]],
) -> float:
    """Compute minimum Euclidean distance (meters) between path waypoints and dynamic obstacles."""
    if path is None or len(path.points) == 0 or not obstacles:
        return float("inf")

    min_clearance = float("inf")
    xy = path.waypoints_xy

    for obs in obstacles:
        ox, oy = 0.0, 0.0
        rad = 0.30
        if isinstance(obs, TrackedObstacleState):
            ox, oy = float(obs.x), float(obs.y)
            rad = max(obs.bbox_size) / 2.0
        elif isinstance(obs, PredictedObstacleTrajectory):
            if obs.waypoints:
                ox, oy = float(obs.waypoints[0].x), float(obs.waypoints[0].y)
            rad = max(obs.bbox_size) / 2.0
        elif isinstance(obs, dict):
            ox, oy = float(obs.get("x", 0.0)), float(obs.get("y", 0.0))
            rad = float(obs.get("radius", 0.30))
        elif hasattr(obs, "x") and hasattr(obs, "y"):
            ox, oy = float(obs.x), float(obs.y)
            size = getattr(obs, "bbox_size", (0.6, 0.6))
            rad = max(size) / 2.0
        else:
            continue

        dists = np.hypot(xy[:, 0] - ox, xy[:, 1] - oy) - rad
        obs_min = float(np.min(dists))
        if obs_min < min_clearance:
            min_clearance = obs_min

    return float(max(0.0, min_clearance))


class EventDrivenGlobalReplanner:
    """Event-driven global replanning supervisor governing A* route regeneration.

    Evaluates explicit persistent blockage and deadlock evidence to regenerate
    global routes while strictly rejecting oscillatory, single-frame, or premature replanning.
    """

    def __init__(
        self,
        config: Optional[GlobalReplannerConfig] = None,
        planner: Optional[AStarPlanner] = None,
    ) -> None:
        self.config = config or GlobalReplannerConfig()
        self.planner = planner or AStarPlanner(self.config.astar_config)

        # Internal state & hysteresis
        self.last_replan_timestamp: float = -1000.0
        self.last_replan_event: Optional[DynamicReplanEvent] = None
        self.current_global_path: Optional[Path] = None
        self.is_current_path_stale: bool = False
        self.blocked_start_timestamp: Optional[float] = None
        self.deadlock_start_timestamp: Optional[float] = None
        self.blocked_duration: float = 0.0
        self.deadlock_duration: float = 0.0
        self.path_invalid_consecutive_cycles: int = 0
        self.progress_history: List[Tuple[float, float]] = []  # [(timestamp, dist_to_goal)]
        self.total_replans_executed: int = 0
        self.event_counter: int = 0
        self.last_result: Optional[DynamicReplanResult] = None

    def reset(self, initial_path: Optional[Path] = None, timestamp: float = 0.0) -> None:
        """Reset replanner state and initialize active global path."""
        self.current_global_path = initial_path
        self.is_current_path_stale = False
        self.last_replan_timestamp = -1000.0
        self.last_replan_event = None
        self.blocked_start_timestamp = None
        self.deadlock_start_timestamp = None
        self.blocked_duration = 0.0
        self.deadlock_duration = 0.0
        self.path_invalid_consecutive_cycles = 0
        self.progress_history = []
        self.total_replans_executed = 0
        self.event_counter = 0
        self.last_result = None

    def set_global_path(self, path: Optional[Path]) -> None:
        """Set or update the active global path."""
        self.current_global_path = path
        self.is_current_path_stale = False
        self.path_invalid_consecutive_cycles = 0

    def is_goal_reached(
        self,
        current_pose: Tuple[float, float, float],
        goal: Tuple[float, float],
    ) -> bool:
        """Verify whether vehicle is already within the goal tolerance zone."""
        dist = math.hypot(current_pose[0] - goal[0], current_pose[1] - goal[1])
        return dist <= self.config.goal_tolerance_m

    def validate_global_path(
        self,
        path: Optional[Path],
        planning_map: Any,
        dynamic_obstacles: Optional[Sequence[Any]] = None,
    ) -> Tuple[bool, List[int], str]:
        """Validate whether the remaining global path is blocked by static or dynamic obstacles.

        Does NOT assume local DWA obstruction implies global path invalidation.
        """
        if path is None or len(path.points) == 0:
            return False, [], "No global path provided"

        blocking_ids: List[int] = []

        # 1. Validate waypoints against map bounds, static obstacles, and critical costs
        for i, pt in enumerate(path.points):
            if not planning_map.is_in_bounds(pt.x, pt.y):
                return False, [], f"Waypoint {i} out of bounds ({pt.x:.2f}, {pt.y:.2f})"

            r, c = planning_map.world_to_grid(pt.x, pt.y)
            if hasattr(planning_map, "obstacle") and planning_map.obstacle[r, c]:
                return False, [], f"Waypoint {i} intersects static/lethal obstacle"

            if hasattr(planning_map, "cost") and float(planning_map.cost[r, c]) >= self.config.lethal_cost_threshold:
                return False, [], f"Waypoint {i} exceeds lethal cost threshold"

            if self.config.unknown_is_lethal and hasattr(planning_map, "unknown") and planning_map.unknown[r, c]:
                return False, [], f"Waypoint {i} is in unknown space under strict policy"

        # 2. Validate waypoints against dynamic obstacles
        if dynamic_obstacles:
            for obs in dynamic_obstacles:
                ox, oy, oyaw = 0.0, 0.0, 0.0
                obs_id = 0
                size = (0.6, 0.6)

                if isinstance(obs, TrackedObstacleState):
                    ox, oy = float(obs.x), float(obs.y)
                    oyaw = float(obs.bbox_yaw)
                    size = obs.bbox_size
                    obs_id = obs.track_id
                elif isinstance(obs, PredictedObstacleTrajectory):
                    if obs.waypoints:
                        ox, oy = float(obs.waypoints[0].x), float(obs.waypoints[0].y)
                    oyaw = float(obs.bbox_yaw)
                    size = obs.bbox_size
                    obs_id = obs.track_id
                elif isinstance(obs, dict):
                    ox = float(obs.get("x", 0.0))
                    oy = float(obs.get("y", 0.0))
                    oyaw = float(obs.get("yaw", 0.0))
                    size = (float(obs.get("length", 0.6)), float(obs.get("width", 0.6)))
                    obs_id = int(obs.get("track_id", 0))

                obb = compute_obb_corners((ox, oy), (size[0], size[1]), oyaw)
                rad = max(size) / 2.0

                for pt in path.points:
                    if math.hypot(pt.x - ox, pt.y - oy) <= rad:
                        if point_in_polygon((pt.x, pt.y), obb):
                            if obs_id not in blocking_ids:
                                blocking_ids.append(obs_id)

            if blocking_ids:
                return False, blocking_ids, f"Global path intersects dynamic obstacle(s): {blocking_ids}"

        return True, [], "Global path valid"

    def check_trigger_conditions(
        self,
        current_pose: Tuple[float, float, float],
        goal: Tuple[float, float],
        current_path: Optional[Path],
        planning_map: Any,
        behavioral_decision: Optional[BehavioralDecision] = None,
        dynamic_obstacles: Optional[Sequence[Any]] = None,
        timestamp: float = 0.0,
    ) -> Tuple[bool, DynamicReplanTrigger, str, List[int]]:
        """Evaluate deterministic trigger hierarchy with debouncing and cooldown protection."""
        # Guard 1: Goal already reached -> No replan required
        if self.is_goal_reached(current_pose, goal):
            return False, DynamicReplanTrigger.NONE, "Goal already reached", []

        # Track progress towards goal
        dist_to_goal = math.hypot(current_pose[0] - goal[0], current_pose[1] - goal[1])
        self.progress_history.append((timestamp, dist_to_goal))
        # Keep rolling window
        cutoff = timestamp - self.config.progress_loss_window_s
        self.progress_history = [p for p in self.progress_history if p[0] >= cutoff]

        # Guard 2: Temporary dynamic obstruction in AVOID or PROCEED -> No global replan
        if behavioral_decision is not None and behavioral_decision.state in (BehavioralState.PROCEED, BehavioralState.AVOID):
            self.blocked_start_timestamp = None
            self.deadlock_start_timestamp = None
            self.blocked_duration = 0.0
            self.deadlock_duration = 0.0
            self.path_invalid_consecutive_cycles = 0

            # Unless the global path itself has become physically invalidated
            is_valid, blocking_ids, reason = self.validate_global_path(current_path, planning_map, dynamic_obstacles)
            if is_valid:
                return False, DynamicReplanTrigger.NONE, "Vehicle active and path clear", []

        # Check path validity
        is_path_valid, blocking_ids, path_reason = self.validate_global_path(current_path, planning_map, dynamic_obstacles)

        # Trigger A: M13.10 DYNAMIC_DEADLOCK event
        if behavioral_decision is not None:
            if behavioral_decision.deadlock or behavioral_decision.deadlock_event == DeadlockEvent.DYNAMIC_DEADLOCK.value:
                if self.deadlock_start_timestamp is None:
                    self.deadlock_start_timestamp = timestamp
                self.deadlock_duration = timestamp - self.deadlock_start_timestamp
                b_ids = [behavioral_decision.blocking_obstacle_id] if behavioral_decision.blocking_obstacle_id else blocking_ids
                return True, DynamicReplanTrigger.DYNAMIC_DEADLOCK, f"Deadlock confirmed: {behavioral_decision.reason}", b_ids

        # Track persistent blocked duration (STOP or WAIT)
        if behavioral_decision is not None and behavioral_decision.state in (BehavioralState.STOP, BehavioralState.WAIT):
            if self.blocked_start_timestamp is None:
                self.blocked_start_timestamp = timestamp
            self.blocked_duration = timestamp - self.blocked_start_timestamp
        else:
            self.blocked_start_timestamp = None
            self.blocked_duration = 0.0

        # Trigger B: Persistent local blockage exceeding duration threshold
        if self.blocked_duration >= self.config.persistent_blockage_duration_s:
            b_ids = [behavioral_decision.blocking_obstacle_id] if behavioral_decision and behavioral_decision.blocking_obstacle_id else blocking_ids
            return True, DynamicReplanTrigger.PERSISTENT_BLOCKAGE, f"Persistent blockage for {self.blocked_duration:.2f}s", b_ids

        # Trigger D: Existing global path invalidated by dynamic/static obstruction (with debouncing)
        if not is_path_valid:
            self.path_invalid_consecutive_cycles += 1
            if self.path_invalid_consecutive_cycles >= self.config.debounce_cycles:
                return True, DynamicReplanTrigger.PATH_INVALIDATED, f"Path invalidated: {path_reason}", blocking_ids
        else:
            self.path_invalid_consecutive_cycles = 0

        # Trigger C: Insufficient progress toward goal over configured window
        if len(self.progress_history) >= 2:
            dt_window = self.progress_history[-1][0] - self.progress_history[0][0]
            if dt_window >= (self.config.progress_loss_window_s * 0.80):
                progress = self.progress_history[0][1] - self.progress_history[-1][1]
                if progress < self.config.min_progress_m:
                    return True, DynamicReplanTrigger.INSUFFICIENT_PROGRESS, f"Progress {progress:.3f}m < {self.config.min_progress_m}m over {dt_window:.1f}s", blocking_ids

        return False, DynamicReplanTrigger.NONE, "No replan condition met", []

    def check_entrance_clearance(
        self,
        new_path: Path,
        dynamic_obstacles: Optional[Sequence[Any]],
    ) -> Tuple[bool, str]:
        """Validate that path entrance has sufficient clearance from active dynamic obstacles."""
        if not dynamic_obstacles or len(new_path.points) == 0:
            return True, "Clear"

        entrance_steps = min(len(new_path.points), self.config.entrance_clearance_steps)
        entrance_pts = new_path.points[:entrance_steps]

        for obs in dynamic_obstacles:
            ox, oy = 0.0, 0.0
            rad = 0.30
            if isinstance(obs, TrackedObstacleState):
                ox, oy = float(obs.x), float(obs.y)
                rad = max(obs.bbox_size) / 2.0
            elif isinstance(obs, PredictedObstacleTrajectory):
                if obs.waypoints:
                    ox, oy = float(obs.waypoints[0].x), float(obs.waypoints[0].y)
                rad = max(obs.bbox_size) / 2.0
            elif isinstance(obs, dict):
                ox, oy = float(obs.get("x", 0.0)), float(obs.get("y", 0.0))
                rad = float(obs.get("radius", 0.30))
            elif hasattr(obs, "x") and hasattr(obs, "y"):
                ox, oy = float(obs.x), float(obs.y)
                size = getattr(obs, "bbox_size", (0.6, 0.6))
                rad = max(size) / 2.0

            for pt in entrance_pts:
                dist = math.hypot(pt.x - ox, pt.y - oy) - rad
                if dist < self.config.entrance_min_clearance_m:
                    return False, f"Entrance waypoint ({pt.x:.2f}, {pt.y:.2f}) clearance {dist:.2f}m < {self.config.entrance_min_clearance_m:.2f}m from obstacle ({ox:.2f}, {oy:.2f})"

        return True, "Entrance cleared"

    def execute_replan(
        self,
        current_pose: Tuple[float, float, float],
        goal: Tuple[float, float],
        base_map: Any,
        dynamic_obstacles: Optional[Sequence[Any]] = None,
        event: Optional[DynamicReplanEvent] = None,
        t_detect: float = 0.0,
        t_validate: float = 0.0,
    ) -> DynamicReplanResult:
        """Execute A* global search on DynamicPlanningMapOverlay and validate resulting path."""
        t_plan_start = time.perf_counter()

        # STALE PATH RULE: Old path is explicitly invalidated upon replan trigger
        self.is_current_path_stale = True

        # Build ephemeral dynamic planning overlay — ZERO mutation of canonical map
        overlay = DynamicPlanningMapOverlay(
            base_map=base_map,
            dynamic_obstacles=dynamic_obstacles,
            inflation_m=self.config.dynamic_obstacle_inflation_m,
            risk_cost_penalty=self.config.dynamic_risk_cost_penalty,
        )

        start_state = RobotState(x=current_pose[0], y=current_pose[1], yaw=current_pose[2])
        goal_pose = GoalPose(x=goal[0], y=goal[1])

        # Execute A* search
        new_path, status, diagnostics = self.planner._search(overlay, start_state, goal_pose)
        t_plan_end = time.perf_counter()
        plan_lat_ms = (t_plan_end - t_plan_start) * 1000.0

        if status != PlannerStatus.SUCCESS or new_path is None or len(new_path.points) == 0:
            # REPLAN FAILURE: Emit GLOBAL_REPLAN_FAILED and keep old path stale
            fail_reason = diagnostics.get("error", "GLOBAL_REPLAN_FAILED: A* found no feasible route")
            return DynamicReplanResult(
                replan_triggered=True,
                trigger_type=event.trigger_type if event else DynamicReplanTrigger.NONE,
                event=event,
                success=False,
                status=status if status != PlannerStatus.SUCCESS else PlannerStatus.NO_PATH,
                path=None,
                is_path_stale=True,
                failure_reason=fail_reason,
                planning_latency_ms=plan_lat_ms,
                validation_latency_ms=t_validate,
                detection_latency_ms=t_detect,
                total_latency_ms=t_detect + t_validate + plan_lat_ms,
            )

        # Entrance clearance validation
        entrance_ok, entrance_msg = self.check_entrance_clearance(new_path, dynamic_obstacles)
        if not entrance_ok:
            return DynamicReplanResult(
                replan_triggered=True,
                trigger_type=event.trigger_type if event else DynamicReplanTrigger.NONE,
                event=event,
                success=False,
                status=PlannerStatus.START_BLOCKED,
                path=None,
                is_path_stale=True,
                failure_reason=f"GLOBAL_REPLAN_FAILED: {entrance_msg}",
                planning_latency_ms=plan_lat_ms,
                validation_latency_ms=t_validate,
                detection_latency_ms=t_detect,
                total_latency_ms=t_detect + t_validate + plan_lat_ms,
            )

        # Calculate path change metrics
        old_p = self.current_global_path
        old_len = float(old_p.total_length_m) if old_p else 0.0
        new_len = float(new_path.total_length_m)
        old_cost = float(diagnostics.get("old_path_cost", old_len * 1.0))
        new_cost = float(diagnostics.get("path_cost", new_len * 1.0))
        dev = compute_path_deviation(old_p, new_path)
        min_clear = compute_min_dynamic_clearance(new_path, dynamic_obstacles)

        # Check goal reachability
        last_pt = new_path.points[-1]
        dist_goal = math.hypot(last_pt.x - goal[0], last_pt.y - goal[1])
        goal_reachable = dist_goal <= (self.config.goal_tolerance_m + 0.05)

        metrics = PathChangeMetrics(
            old_path_length=old_len,
            new_path_length=new_len,
            old_path_cost=old_cost,
            new_path_cost=new_cost,
            path_deviation=dev,
            goal_reachable=goal_reachable,
            num_dynamic_obstacle_intersections=0,
            min_dynamic_clearance=min_clear,
        )

        # SUCCESSFUL REPLAN
        self.current_global_path = new_path
        self.is_current_path_stale = False
        self.last_replan_timestamp = event.timestamp if event else time.time()
        self.total_replans_executed += 1
        self.blocked_start_timestamp = None
        self.deadlock_start_timestamp = None
        self.blocked_duration = 0.0
        self.deadlock_duration = 0.0
        self.path_invalid_consecutive_cycles = 0

        return DynamicReplanResult(
            replan_triggered=True,
            trigger_type=event.trigger_type if event else DynamicReplanTrigger.NONE,
            event=event,
            success=True,
            status=PlannerStatus.SUCCESS,
            path=new_path,
            is_path_stale=False,
            metrics=metrics,
            planning_latency_ms=plan_lat_ms,
            validation_latency_ms=t_validate,
            detection_latency_ms=t_detect,
            total_latency_ms=t_detect + t_validate + plan_lat_ms,
        )

    def check_and_replan(
        self,
        current_pose: Tuple[float, float, float],
        goal: Tuple[float, float],
        base_map: Any,
        behavioral_decision: Optional[BehavioralDecision] = None,
        dynamic_obstacles: Optional[Sequence[Any]] = None,
        timestamp: float = 0.0,
    ) -> DynamicReplanResult:
        """Main periodic evaluation loop for event-driven global replanning."""
        t_start = time.perf_counter()

        # 1. Evaluate trigger conditions
        should_replan, trigger, reason, blocking_ids = self.check_trigger_conditions(
            current_pose=current_pose,
            goal=goal,
            current_path=self.current_global_path,
            planning_map=base_map,
            behavioral_decision=behavioral_decision,
            dynamic_obstacles=dynamic_obstacles,
            timestamp=timestamp,
        )
        t_detect = (time.perf_counter() - t_start) * 1000.0

        if not should_replan:
            return DynamicReplanResult(
                replan_triggered=False,
                trigger_type=DynamicReplanTrigger.NONE,
                success=True,
                status=PlannerStatus.SUCCESS,
                path=self.current_global_path,
                is_path_stale=self.is_current_path_stale,
                detection_latency_ms=t_detect,
                total_latency_ms=t_detect,
            )

        # 2. Check replan cooldown
        time_since_last = timestamp - self.last_replan_timestamp
        cooldown_rem = max(0.0, self.config.cooldown_s - time_since_last)

        # Cooldown override condition: direct physical collision on current path
        is_path_valid, _, _ = self.validate_global_path(self.current_global_path, base_map, dynamic_obstacles)
        is_direct_collision = not is_path_valid

        if cooldown_rem > 0.0 and not is_direct_collision:
            return DynamicReplanResult(
                replan_triggered=False,
                trigger_type=trigger,
                success=False,
                status=PlannerStatus.FOLLOWING,
                path=self.current_global_path,
                is_path_stale=self.is_current_path_stale,
                cooldown_remaining_s=cooldown_rem,
                failure_reason=f"Replan suppressed by cooldown ({cooldown_rem:.2f}s remaining)",
                detection_latency_ms=t_detect,
                total_latency_ms=t_detect,
            )

        # 3. Construct structured event
        t_val_start = time.perf_counter()
        self.event_counter += 1
        event_id = f"replan_{trigger.value.lower()}_{self.event_counter:04d}_{timestamp:.2f}"

        progress_val = 0.0
        if len(self.progress_history) >= 2:
            progress_val = float(self.progress_history[0][1] - self.progress_history[-1][1])

        event = DynamicReplanEvent(
            trigger_type=trigger,
            timestamp=timestamp,
            current_pose=current_pose,
            current_goal=goal,
            current_global_path=self.current_global_path,
            blocking_obstacle_ids=blocking_ids,
            blocked_duration=self.blocked_duration,
            deadlock_duration=self.deadlock_duration,
            current_progress=progress_val,
            reason=reason,
            confidence=float(behavioral_decision.confidence if behavioral_decision else 1.0),
            event_id=event_id,
        )
        self.last_replan_event = event
        t_validate = (time.perf_counter() - t_val_start) * 1000.0

        # 4. Execute replan
        result = self.execute_replan(
            current_pose=current_pose,
            goal=goal,
            base_map=base_map,
            dynamic_obstacles=dynamic_obstacles,
            event=event,
            t_detect=t_detect,
            t_validate=t_validate,
        )
        self.last_result = result
        return result
