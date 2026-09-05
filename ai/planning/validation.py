"""Validation engine ensuring candidate paths comply with geometric and safety rules."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import GoalState, PlannerStatus, RobotState, VehicleKinematics
from ai.planning.collision import FootprintCollisionChecker
from ai.planning.path import Path


class PathValidator:
    """Validates global and local paths against map boundaries, clearances, and lethal hazards."""

    def __init__(
        self,
        vehicle: VehicleKinematics = VehicleKinematics(),
        max_safe_cost: float = 0.65,
        critical_slope_deg: float = 28.0,
        critical_step_m: float = 0.20
    ) -> None:
        self.checker = FootprintCollisionChecker(vehicle)
        self.max_safe_cost = max_safe_cost
        self.critical_slope_deg = critical_slope_deg
        self.critical_step_m = critical_step_m

    def validate_start_goal(
        self,
        start: RobotState,
        goal: GoalState,
        costmap: BEVTerrainCostmap
    ) -> Tuple[bool, PlannerStatus, str]:
        """Verify validity of start and goal poses before planning."""
        # 1. Start bounds and passability
        if not costmap.is_in_bounds(start.x, start.y):
            return False, PlannerStatus.INVALID_START, f"Start ({start.x:.2f}, {start.y:.2f}) is out of map bounds."

        r_s, c_s = costmap.meter_to_grid(start.x, start.y)
        if costmap.obstacle[r_s, c_s] or costmap.cost[r_s, c_s] > self.max_safe_cost:
            return False, PlannerStatus.INVALID_START, f"Start ({start.x:.2f}, {start.y:.2f}) is on an obstacle or lethal cell."

        # 2. Goal bounds and passability
        if not costmap.is_in_bounds(goal.x, goal.y):
            return False, PlannerStatus.INVALID_GOAL, f"Goal ({goal.x:.2f}, {goal.y:.2f}) is out of map bounds."

        r_g, c_g = costmap.meter_to_grid(goal.x, goal.y)
        if costmap.obstacle[r_g, c_g] or costmap.cost[r_g, c_g] > self.max_safe_cost:
            return False, PlannerStatus.INVALID_GOAL, f"Goal ({goal.x:.2f}, {goal.y:.2f}) is on an obstacle or lethal cell."

        return True, PlannerStatus.SUCCESS, "Start and goal are valid."

    def validate_path(
        self,
        path: Path,
        costmap: BEVTerrainCostmap
    ) -> Tuple[bool, str, Dict[str, float]]:
        """Validate entire path point-by-point against safety thresholds."""
        if len(path) == 0:
            return False, "Path is empty.", {}

        interp_path = path.interpolate(step_m=0.05)
        min_clearance = float("inf")
        max_slope = 0.0
        max_step = 0.0

        for i, pt in enumerate(interp_path.points):
            if not costmap.is_in_bounds(pt.x, pt.y):
                return False, f"Path point {i} ({pt.x:.2f}, {pt.y:.2f}) is out of bounds.", {}

            r, c = costmap.meter_to_grid(pt.x, pt.y)

            # Check obstacle
            if costmap.obstacle[r, c]:
                return False, f"Path point {i} intersects hard obstacle at cell ({r}, {c}).", {}

            # Check cost threshold
            if costmap.cost[r, c] > self.max_safe_cost:
                return False, f"Path point {i} crosses unsafe cell cost {costmap.cost[r, c]:.2f} > {self.max_safe_cost}.", {}

            # Check critical rollover slope
            slope = costmap.slope_deg[r, c]
            max_slope = max(max_slope, slope)
            if slope >= self.critical_slope_deg:
                return False, f"Path point {i} exceeds critical slope {slope:.1f}° >= {self.critical_slope_deg}°.", {}

            # Check critical step height
            step_h = costmap.height_var[r, c]
            max_step = max(max_step, step_h)
            if step_h >= self.critical_step_m:
                return False, f"Path point {i} exceeds critical step {step_h:.2f}m >= {self.critical_step_m}m.", {}

            clearance = self.checker.get_clearance(pt.x, pt.y, costmap)
            min_clearance = min(min_clearance, clearance)
            pt.clearance_m = clearance

        diagnostics = {
            "min_clearance_m": round(float(min_clearance), 3),
            "max_slope_deg": round(float(max_slope), 1),
            "max_step_m": round(float(max_step), 3)
        }
        return True, "Path successfully validated.", diagnostics
