"""Visualizer for global paths, candidate local trajectories, and robot footprints on BEV costmaps."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import GoalState, RobotState
from ai.planning.collision import FootprintCollisionChecker
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory


class PlanningVisualizer:
    """Renders 2D planning diagnostic HUDs overlaying paths and footprints on BEV costmaps."""

    def __init__(self) -> None:
        self.checker = FootprintCollisionChecker()

    def render_plan(
        self,
        costmap: BEVTerrainCostmap,
        start: RobotState,
        goal: GoalState,
        global_path: Optional[Path] = None,
        local_trajectory: Optional[Trajectory] = None,
        candidate_trajectories: Optional[List[Trajectory]] = None,
        title: str = "UGV Motion Plan"
    ) -> np.ndarray:
        """Render a high-resolution 2D diagnostic visualization (RGB)."""
        h_grid, w_grid = costmap.shape
        scale = 8  # Magnify 8x for clear visualization
        img_h = h_grid * scale
        img_w = w_grid * scale

        # Base cost layer normalized to uint8 [0, 255]
        cost_uint8 = np.clip(costmap.cost * 255.0, 0, 255).astype(np.uint8)
        color_cost = cv2.applyColorMap(cost_uint8, cv2.COLORMAP_TURBO)
        color_cost = cv2.resize(color_cost, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # Mark lethal obstacles in dark crimson
        obs_resized = cv2.resize(costmap.obstacle.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        color_cost[obs_resized > 0] = [30, 30, 180]  # BGR

        # Mark unknown territory in textured dark gray
        unk_resized = cv2.resize(costmap.unknown.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        color_cost[unk_resized > 0] = [60, 60, 60]

        def meter_to_pixel(x_m: float, y_m: float) -> Tuple[int, int]:
            r, c = costmap.meter_to_grid(x_m, y_m)
            px = int(np.clip(c * scale + scale // 2, 0, img_w - 1))
            py = int(np.clip(r * scale + scale // 2, 0, img_h - 1))
            return (px, py)

        # 1. Candidate Trajectories (faint white/cyan arcs)
        if candidate_trajectories:
            for cand in candidate_trajectories:
                if cand.points:
                    cand_pts = [meter_to_pixel(pt.x, pt.y) for pt in cand.points]
                    for k in range(len(cand_pts) - 1):
                        cv2.line(color_cost, cand_pts[k], cand_pts[k+1], (180, 180, 180), 1)

        # 2. Global Path (vibrant bright cyan)
        if global_path and len(global_path) > 1:
            g_pts = [meter_to_pixel(pt.x, pt.y) for pt in global_path.points]
            for k in range(len(g_pts) - 1):
                cv2.line(color_cost, g_pts[k], g_pts[k+1], (255, 255, 0), 2)  # Cyan in BGR
            for pt in g_pts:
                cv2.circle(color_cost, pt, 2, (255, 255, 0), -1)

        # 3. Selected Local Trajectory (thick magenta)
        if local_trajectory and len(local_trajectory) > 1:
            l_pts = [meter_to_pixel(pt.x, pt.y) for pt in local_trajectory.points]
            for k in range(len(l_pts) - 1):
                cv2.line(color_cost, l_pts[k], l_pts[k+1], (255, 0, 255), 3)  # Magenta in BGR

        # 4. Robot Oriented Footprint (Yellow rectangle)
        corners = self.checker.get_footprint_corners(start.x, start.y, start.yaw)
        corner_pxs = [meter_to_pixel(corners[i, 0], corners[i, 1]) for i in range(4)]
        pts_arr = np.array(corner_pxs, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(color_cost, [pts_arr], isClosed=True, color=(0, 255, 255), thickness=2)

        # Heading vector
        hx = start.x + 0.40 * np.cos(start.yaw)
        hy = start.y + 0.40 * np.sin(start.yaw)
        p_start = meter_to_pixel(start.x, start.y)
        p_head = meter_to_pixel(hx, hy)
        cv2.arrowedLine(color_cost, p_start, p_head, (0, 255, 255), 2, tipLength=0.3)

        # 5. Goal Marker (Green target ring)
        p_goal = meter_to_pixel(goal.x, goal.y)
        cv2.circle(color_cost, p_goal, 8, (0, 255, 0), 2)
        cv2.circle(color_cost, p_goal, 3, (0, 255, 0), -1)

        # 6. Diagnostics Banner
        header = np.zeros((40, img_w, 3), dtype=np.uint8)
        header[:] = (25, 25, 25)
        cv2.putText(header, f"{title} | Start: ({start.x:.1f}, {start.y:.1f}) -> Goal: ({goal.x:.1f}, {goal.y:.1f})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        combined = np.vstack([header, color_cost])
        return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
