"""Unified UGV Navigation Planner uniting A* global search, DWA local planning, and Safety Controller."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from ai.planning.base import GoalPose, GoalState, PlanningConfig, PlannerStatus, RobotState
from ai.planning.collision import FootprintCollisionChecker
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path
from ai.planning.safety import SafetyController
from ai.planning.trajectory import Trajectory


class UGVNavigationPlanner:
    """Integrated navigation planning system for outdoor 4WD UGV."""

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        self.config = config or PlanningConfig()
        self.a_star = AStarPlanner(self.config)
        self.dwa = DWAPlanner(self.config)
        self.safety = SafetyController(self.config)
        self.collision_checker = FootprintCollisionChecker(self.config.vehicle)

        safety_cfg = getattr(self.config, "safety", {})
        self.replan_dev_threshold = float(safety_cfg.get("replan_cross_track_threshold_m", 0.50))
        self.replan_blocked_lookahead = float(safety_cfg.get("replan_blocked_lookahead_m", 2.00))

        # Runtime planner state
        self.current_goal: Optional[GoalPose] = None
        self.global_path: Optional[Path] = None
        self.local_trajectory: Optional[Trajectory] = None
        self.status: PlannerStatus = PlannerStatus.IDLE
        self.replan_count: int = 0
        self.emergency_stop_count: int = 0
        self.last_sim_time: float = 0.0

    def set_goal(self, goal: Union[GoalPose, GoalState, Tuple[float, float]]) -> None:
        """Set or update target navigation goal."""
        if isinstance(goal, (GoalPose, GoalState)):
            self.current_goal = goal
        else:
            tol_xy = float(self.config.tolerances.get("goal_xy_tolerance_m", 0.20))
            self.current_goal = GoalPose(x=float(goal[0]), y=float(goal[1]), xy_tolerance_m=tol_xy)
        # Invalidate current global path to trigger fresh A* search
        self.global_path = None
        self.status = PlannerStatus.PLANNING

    def _should_replan(self, current_state: RobotState, terrain_map: Any) -> Tuple[bool, str]:
        """Determine if a global replan is required based on path invalidation or deviation."""
        if self.global_path is None or len(self.global_path) == 0:
            return True, "No active global path"

        # 1. Check path deviation (cross-track error against path segments)
        xy = self.global_path.waypoints_xy
        if len(xy) == 1:
            min_dist = float(np.hypot(xy[0, 0] - current_state.x, xy[0, 1] - current_state.y))
        else:
            min_dist = min(
                DWAPlanner._point_to_segment_dist(
                    current_state.x, current_state.y,
                    float(xy[i, 0]), float(xy[i, 1]),
                    float(xy[i + 1, 0]), float(xy[i + 1, 1])
                )
                for i in range(len(xy) - 1)
            )
        if min_dist > self.replan_dev_threshold:
            return True, f"Cross-track deviation {min_dist:.2f}m > threshold {self.replan_dev_threshold:.2f}m"

        # 2. Check if path segment immediately ahead has become blocked by obstacle/hazard
        dists = np.hypot(xy[:, 0] - current_state.x, xy[:, 1] - current_state.y)
        closest_idx = int(np.argmin(dists))
        lookahead_idx = min(len(xy) - 1, closest_idx + 5)
        for idx in range(closest_idx, lookahead_idx + 1):
            wx, wy = float(xy[idx, 0]), float(xy[idx, 1])
            if self.collision_checker.is_point_collision(wx, wy, terrain_map):
                return True, f"Global path waypoint at index {idx} ({wx:.1f}, {wy:.1f}) is obstructed"

        return False, "Nominal tracking"

    def plan_step(
        self,
        current_state: RobotState,
        terrain_map: Any,
        goal: Optional[Union[GoalPose, GoalState, Tuple[float, float]]] = None,
        tracking_state: Union[int, str] = 0,
        map_timestamp_s: Optional[float] = None,
        loc_timestamp_s: Optional[float] = None,
        tf_available: bool = True,
        sim_time_s: Optional[float] = None,
        dt_s: float = 0.10,
    ) -> Tuple[float, float, Optional[Path], Optional[Trajectory], PlannerStatus, Dict[str, Any]]:
        """Execute one complete planning and control generation cycle.

        Returns:
            (linear_x, angular_z, global_path, local_trajectory, status, diagnostics)
        """
        curr_time = sim_time_s if sim_time_s is not None else current_state.timestamp
        if goal is not None:
            if self.current_goal is None or (self.current_goal.x != goal[0] if isinstance(goal, tuple) else self.current_goal.x != goal.x):
                self.set_goal(goal)

        diagnostics: Dict[str, Any] = {
            "timestamp": curr_time,
            "status": self.status.value,
            "replan_count": self.replan_count,
            "emergency_stop_count": self.emergency_stop_count,
        }

        # 1. Check Goal Reached (Section 46)
        if self.current_goal is not None and self.current_goal.is_reached(current_state):
            self.status = PlannerStatus.GOAL_REACHED
            cmd_v, cmd_w, status, arb_diag = self.safety.arbitrate_command(
                requested_v=0.0, requested_w=0.0, dt_s=dt_s,
                current_time_s=curr_time, planner_status=self.status
            )
            return 0.0, 0.0, self.global_path, None, self.status, {**diagnostics, **arb_diag}

        # 2. Safety Guards Evaluation (Sections 32-36)
        is_safe, guard_status, guard_reason, is_degraded = self.safety.evaluate_safety_guards(
            current_time_s=curr_time,
            current_state=current_state,
            terrain_map=terrain_map,
            tracking_state=tracking_state,
            map_timestamp_s=map_timestamp_s,
            loc_timestamp_s=loc_timestamp_s,
            tf_available=tf_available
        )

        if not is_safe:
            self.status = guard_status
            self.emergency_stop_count += 1
            cmd_v, cmd_w, status, arb_diag = self.safety.arbitrate_command(
                requested_v=0.0, requested_w=0.0, dt_s=dt_s,
                current_time_s=curr_time, planner_status=self.status,
                is_emergency=True
            )
            diagnostics["safety_fault_reason"] = guard_reason
            return 0.0, 0.0, self.global_path, None, self.status, {**diagnostics, **arb_diag}

        # 3. Check for Global Goal Availability
        if self.current_goal is None:
            self.status = PlannerStatus.IDLE
            return 0.0, 0.0, None, None, self.status, {**diagnostics, "info": "No active navigation goal"}

        # 4. Check if Global Replan Needed (Sections 47-49)
        needs_replan, replan_reason = self._should_replan(current_state, terrain_map)
        if needs_replan:
            self.status = PlannerStatus.PLANNING
            path, astar_status, astar_diag = self.a_star._search(terrain_map, current_state, self.current_goal)
            diagnostics["astar_diag"] = astar_diag
            diagnostics["replan_reason"] = replan_reason

            if astar_status != PlannerStatus.SUCCESS or path is None:
                self.status = astar_status
                cmd_v, cmd_w, status, arb_diag = self.safety.arbitrate_command(
                    requested_v=0.0, requested_w=0.0, dt_s=dt_s,
                    current_time_s=curr_time, planner_status=self.status
                )
                return 0.0, 0.0, None, None, self.status, {**diagnostics, **arb_diag}

            self.global_path = path
            self.replan_count += 1
            self.status = PlannerStatus.FOLLOWING

        # 5. Local Trajectory Generation via DWA (Sections 25-31)
        assert self.global_path is not None
        local_traj, desired_v, desired_w, dwa_status, dwa_diag = self.dwa.compute_trajectory(
            current_state=current_state,
            global_path=self.global_path,
            costmap=terrain_map,
            is_degraded=is_degraded
        )
        self.local_trajectory = local_traj
        diagnostics["dwa_diag"] = dwa_diag

        if dwa_status != PlannerStatus.SUCCESS:
            self.status = dwa_status
            self.emergency_stop_count += 1
            cmd_v, cmd_w, status, arb_diag = self.safety.arbitrate_command(
                requested_v=0.0, requested_w=0.0, dt_s=dt_s,
                current_time_s=curr_time, planner_status=self.status
            )
            return 0.0, 0.0, self.global_path, local_traj, self.status, {**diagnostics, **arb_diag}

        # 6. Safety Arbitration & Clamping (Sections 37-40, 70)
        final_v, final_w, cmd_status, arb_diag = self.safety.arbitrate_command(
            requested_v=desired_v,
            requested_w=desired_w,
            dt_s=dt_s,
            current_time_s=curr_time,
            planner_status=PlannerStatus.FOLLOWING,
            is_degraded=is_degraded
        )

        self.status = cmd_status
        diagnostics.update(arb_diag)
        return final_v, final_w, self.global_path, self.local_trajectory, self.status, diagnostics
