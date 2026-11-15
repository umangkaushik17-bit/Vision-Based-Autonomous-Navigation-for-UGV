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

    def _coords(self, costmap: Any, x: float, y: float) -> Tuple[int, int]:
        if hasattr(costmap, "world_to_grid"):
            return costmap.world_to_grid(x, y)
        return costmap.meter_to_grid(x, y)

    def validate_start_goal(
        self,
        start: RobotState,
        goal: GoalState,
        costmap: Any
    ) -> Tuple[bool, PlannerStatus, str]:
        """Verify validity of start and goal poses before planning."""
        # 1. Start bounds and passability
        if not costmap.is_in_bounds(start.x, start.y):
            return False, PlannerStatus.INVALID_START, f"Start ({start.x:.2f}, {start.y:.2f}) is out of map bounds."

        r_s, c_s = self._coords(costmap, start.x, start.y)
        if (hasattr(costmap, "obstacle") and costmap.obstacle[r_s, c_s]) or (hasattr(costmap, "cost") and costmap.cost[r_s, c_s] > self.max_safe_cost):
            return False, PlannerStatus.INVALID_START, f"Start ({start.x:.2f}, {start.y:.2f}) is on an obstacle or lethal cell."

        # 2. Goal bounds and passability
        if not costmap.is_in_bounds(goal.x, goal.y):
            return False, PlannerStatus.INVALID_GOAL, f"Goal ({goal.x:.2f}, {goal.y:.2f}) is out of map bounds."

        r_g, c_g = self._coords(costmap, goal.x, goal.y)
        if (hasattr(costmap, "obstacle") and costmap.obstacle[r_g, c_g]) or (hasattr(costmap, "cost") and costmap.cost[r_g, c_g] > self.max_safe_cost):
            return False, PlannerStatus.INVALID_GOAL, f"Goal ({goal.x:.2f}, {goal.y:.2f}) is on an obstacle or lethal cell."

        return True, PlannerStatus.SUCCESS, "Start and goal are valid."

    def validate_path(
        self,
        path: Path,
        costmap: Any
    ) -> Tuple[bool, str, Dict[str, float]]:
        """Validate entire path against safety thresholds using vectorized NumPy evaluation."""
        if len(path) == 0:
            return False, "Path is empty.", {}

        interp_path = path.interpolate(step_m=0.05)
        xy = interp_path.waypoints_xy
        if len(xy) == 0:
            return False, "Path is empty.", {}

        xs = xy[:, 0]
        ys = xy[:, 1]

        res = float(getattr(costmap, "resolution_m", getattr(costmap, "resolution", 0.10)))
        shape = getattr(costmap, "shape", (costmap.cost.shape[0], costmap.cost.shape[1]))
        n_rows, n_cols = int(shape[0]), int(shape[1])

        if hasattr(costmap, "x_range_m"):
            x_min, x_max = float(costmap.x_range_m[0]), float(costmap.x_range_m[1])
            y_min, y_max = float(costmap.y_range_m[0]), float(costmap.y_range_m[1])
        elif hasattr(costmap, "x_bounds_m"):
            x_min, x_max = float(costmap.x_bounds_m[0]), float(costmap.x_bounds_m[1])
            y_min, y_max = float(costmap.y_bounds_m[0]), float(costmap.y_bounds_m[1])
        else:
            x_min, x_max = -1e9, 1e9
            y_min, y_max = -1e9, 1e9

        out_bounds = (xs < x_min) | (xs >= x_max) | (ys < y_min) | (ys >= y_max)
        if np.any(out_bounds):
            i = int(np.argmax(out_bounds))
            return False, f"Path point {i} ({xs[i]:.2f}, {ys[i]:.2f}) is out of bounds.", {}

        if hasattr(costmap, "world_to_grid_vec"):
            rows, cols = costmap.world_to_grid_vec(xs, ys)
        else:
            rows = ((x_max - xs) / res).astype(np.int32)
            cols = ((y_max - ys) / res).astype(np.int32)

        rows = np.clip(rows, 0, n_rows - 1)
        cols = np.clip(cols, 0, n_cols - 1)

        if hasattr(costmap, "obstacle"):
            obs_hits = costmap.obstacle[rows, cols]
            if np.any(obs_hits):
                i = int(np.argmax(obs_hits))
                return False, f"Path point {i} intersects hard obstacle at cell ({rows[i]}, {cols[i]}).", {}

        if hasattr(costmap, "cost"):
            cost_vals = costmap.cost[rows, cols]
            unsafe_cost = cost_vals > self.max_safe_cost
            if np.any(unsafe_cost):
                i = int(np.argmax(unsafe_cost))
                return False, f"Path point {i} crosses unsafe cell cost {cost_vals[i]:.2f} > {self.max_safe_cost}.", {}

        if hasattr(costmap, "slope_deg"):
            slope_vals = costmap.slope_deg[rows, cols]
        elif hasattr(costmap, "slope"):
            slope_vals = costmap.slope[rows, cols]
        else:
            slope_vals = None

        max_slope = 0.0
        if slope_vals is not None:
            max_slope = float(np.max(slope_vals))
            if max_slope >= self.critical_slope_deg:
                i = int(np.argmax(slope_vals >= self.critical_slope_deg))
                return False, f"Path point {i} exceeds critical slope {slope_vals[i]:.1f}° >= {self.critical_slope_deg}°.", {}

        max_step = 0.0
        if hasattr(costmap, "height_var"):
            step_vals = costmap.height_var[rows, cols]
            max_step = float(np.max(step_vals))
            if max_step >= self.critical_step_m:
                i = int(np.argmax(step_vals >= self.critical_step_m))
                return False, f"Path point {i} exceeds critical step {step_vals[i]:.2f}m >= {self.critical_step_m}m.", {}

        # Fast Euclidean distance transform clearance lookup
        max_search_radius_m = 2.0
        has_obs = hasattr(costmap, "obstacle") and costmap.obstacle.any()
        has_haz = hasattr(costmap, "hazard") and (costmap.hazard >= 0.70).any()
        if not has_obs and not has_haz:
            min_clearance = max_search_radius_m
        else:
            from scipy.ndimage import distance_transform_edt
            obs_mask = costmap.obstacle if hasattr(costmap, "obstacle") else np.zeros((n_rows, n_cols), dtype=bool)
            if has_haz:
                obs_mask = obs_mask | (costmap.hazard >= 0.70)
            edt = distance_transform_edt(~obs_mask) * res
            pt_clearances = np.minimum(max_search_radius_m, edt[rows, cols])
            min_clearance = float(np.min(pt_clearances))

        diagnostics = {
            "min_clearance_m": round(float(min_clearance), 3),
            "max_slope_deg": round(float(max_slope), 1),
            "max_step_m": round(float(max_step), 3)
        }
        return True, "Path successfully validated.", diagnostics
