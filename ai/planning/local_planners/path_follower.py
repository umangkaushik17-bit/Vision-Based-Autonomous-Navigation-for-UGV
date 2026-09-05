"""Pure Pursuit geometric path-following controller with safety lookahead."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.collision import FootprintCollisionChecker
from ai.planning.local_planners.base import LocalPlanner
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory, TrajectoryPoint


class PurePursuitTracker(LocalPlanner):
    """Geometric Pure Pursuit path tracker with obstacle lookahead validation."""

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        super().__init__(name="Pure Pursuit", config=config)
        self.collision_checker = FootprintCollisionChecker(self.config.vehicle)

        pp_cfg = self.config.pure_pursuit
        self.lookahead_ratio = float(pp_cfg.get("lookahead_ratio", 0.40))
        self.min_lookahead = float(pp_cfg.get("min_lookahead_m", 0.30))
        self.max_lookahead = float(pp_cfg.get("max_lookahead_m", 1.20))
        self.max_v = self.config.vehicle.max_linear_velocity_mps
        self.max_omega = self.config.vehicle.max_angular_velocity_radps

    def _find_lookahead_point(self, state: RobotState, global_path: Path, lookahead_dist: float) -> Tuple[float, float, int]:
        """Find coordinates of the lookahead point along the path."""
        xy = global_path.waypoints_xy
        dists = np.hypot(xy[:, 0] - state.x, xy[:, 1] - state.y)
        closest_idx = int(np.argmin(dists))

        target_idx = closest_idx
        while target_idx < len(xy) - 1 and dists[target_idx] < lookahead_dist:
            target_idx += 1

        return (float(xy[target_idx, 0]), float(xy[target_idx, 1]), target_idx)

    def compute_trajectory(
        self,
        current_state: RobotState,
        global_path: Path,
        costmap: BEVTerrainCostmap
    ) -> Tuple[Optional[Trajectory], float, float, PlannerStatus, Dict[str, Any]]:
        """Compute steering curvature and trajectory toward lookahead waypoint."""
        if len(global_path) < 2:
            return None, 0.0, 0.0, PlannerStatus.NO_PATH, {"reason": "Path is too short"}

        # Dynamic lookahead based on speed
        l_dist = max(self.min_lookahead, min(self.max_lookahead, self.lookahead_ratio * max(current_state.v, 0.20)))
        lx, ly, target_idx = self._find_lookahead_point(current_state, global_path, l_dist)

        # Collision check lookahead segment
        p_curr = (current_state.x, current_state.y)
        p_look = (lx, ly)
        if self.collision_checker.is_segment_collision(p_curr, p_look, costmap):
            # Emergency stop: path ahead is blocked
            zero_traj = Trajectory(
                points=[TrajectoryPoint(
                    t=current_state.timestamp + 0.1,
                    x=current_state.x, y=current_state.y, yaw=current_state.yaw,
                    v=0.0, omega=0.0
                )],
                is_valid=False
            )
            return zero_traj, 0.0, 0.0, PlannerStatus.NO_SAFE_TRAJECTORY, {
                "emergency_stop_triggered": True,
                "reason": "Lookahead segment intersects obstacle"
            }

        # Transform lookahead point to robot coordinate frame
        dx = lx - current_state.x
        dy = ly - current_state.y
        cos_yaw = math.cos(current_state.yaw)
        sin_yaw = math.sin(current_state.yaw)

        # Local coordinates (forward = x_l, lateral = y_l)
        x_local = cos_yaw * dx + sin_yaw * dy
        y_local = -sin_yaw * dx + cos_yaw * dy

        actual_l = max(math.hypot(dx, dy), 1e-4)
        curvature = (2.0 * y_local) / (actual_l**2)

        # Speed scaling: slow down on high curvature
        curv_scale = 1.0 / (1.0 + 1.5 * abs(curvature))
        target_v = min(self.max_v, max(0.20, self.max_v * curv_scale))
        target_omega = np.clip(target_v * curvature, -self.max_omega, self.max_omega)

        # Simulate brief forward rollout (1.0s)
        dt = 0.10
        steps = 10
        sim_x, sim_y, sim_yaw = current_state.x, current_state.y, current_state.yaw
        points: List[TrajectoryPoint] = []
        for step in range(steps):
            t = current_state.timestamp + (step + 1) * dt
            sim_yaw += target_omega * dt
            sim_x += target_v * math.cos(sim_yaw) * dt
            sim_y += target_v * math.sin(sim_yaw) * dt
            clr = self.collision_checker.get_clearance(sim_x, sim_y, costmap)
            points.append(TrajectoryPoint(t=t, x=sim_x, y=sim_y, yaw=sim_yaw, v=target_v, omega=target_omega, clearance_m=clr))

        traj = Trajectory(points=points, is_valid=True, min_clearance_m=float(np.min([p.clearance_m for p in points])))

        diagnostics = {
            "lookahead_dist_m": round(l_dist, 3),
            "target_waypoint_idx": target_idx,
            "curvature": round(curvature, 4),
            "desired_linear_v": round(target_v, 3),
            "desired_angular_w": round(target_omega, 3),
            "emergency_stop_triggered": False
        }
        return traj, target_v, target_omega, PlannerStatus.SUCCESS, diagnostics
