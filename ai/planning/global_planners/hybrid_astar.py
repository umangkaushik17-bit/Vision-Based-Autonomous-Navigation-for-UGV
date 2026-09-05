"""Kinematic 3D State-Space (x, y, theta) Hybrid A* global planner."""

from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import GoalState, PlanningConfig, PlannerStatus, RobotState
from ai.planning.collision import FootprintCollisionChecker
from ai.planning.costs import TerrainAwareCostEvaluator
from ai.planning.global_planners.base import GlobalPlanner
from ai.planning.path import Path


class HybridNode:
    """Search node in continuous metric pose (x, y, yaw)."""
    def __init__(
        self,
        x: float,
        y: float,
        yaw: float,
        g_cost: float = 0.0,
        parent: Optional[HybridNode] = None,
        steer_cmd: float = 0.0
    ) -> None:
        self.x = x
        self.y = y
        self.yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
        self.g_cost = g_cost
        self.parent = parent
        self.steer_cmd = steer_cmd


class HybridAStarPlanner(GlobalPlanner):
    """Orientation-aware Hybrid A* operating in SE(2) state space."""

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        super().__init__(name="Hybrid A*", config=config)
        self.cost_evaluator = TerrainAwareCostEvaluator(self.config)
        self.collision_checker = FootprintCollisionChecker(self.config.vehicle)

        cfg = self.config.hybrid_astar
        self.num_heading_bins = int(cfg.get("heading_bins", 16))
        self.step_size = float(cfg.get("step_size_m", 0.25))
        self.min_turning_radius = float(cfg.get("min_turning_radius_m", 0.40))
        self.max_nodes = int(cfg.get("max_nodes", 6000))

        # Motion primitives: (steering_angle, forward_step)
        # Steer angles: straight (0), left turn (+max_curv), right turn (-max_curv)
        max_steer = math.atan(self.config.vehicle.wheelbase_m / self.min_turning_radius)
        self.steer_angles = [0.0, max_steer, -max_steer, 0.5 * max_steer, -0.5 * max_steer]

    def _state_to_grid_index(
        self,
        node: HybridNode,
        costmap: BEVTerrainCostmap
    ) -> Tuple[int, int, int]:
        """Convert continuous (x, y, yaw) to discrete 3D grid cell."""
        r, c = costmap.meter_to_grid(node.x, node.y)
        # Normalize yaw to [0, 2*pi)
        norm_yaw = (node.yaw + 2.0 * np.pi) % (2.0 * np.pi)
        heading_bin = int(norm_yaw / (2.0 * np.pi / self.num_heading_bins)) % self.num_heading_bins
        return (r, c, heading_bin)

    def _heuristic(self, node: HybridNode, goal: GoalState) -> float:
        """Euclidean distance + heading alignment heuristic."""
        dist = math.hypot(goal.x - node.x, goal.y - node.y)
        if goal.yaw is not None:
            dyaw = abs((goal.yaw - node.yaw + np.pi) % (2.0 * np.pi) - np.pi)
            return float(dist + 0.20 * dyaw)
        return float(dist)

    def _propagate_primitive(self, current: HybridNode, steer: float) -> HybridNode:
        """Propagate kinematic bicycle model forward by step_size."""
        L = self.config.vehicle.wheelbase_m
        ds = self.step_size
        d_theta = (ds / L) * math.tan(steer)
        new_yaw = current.yaw + d_theta
        new_x = current.x + ds * math.cos(0.5 * (current.yaw + new_yaw))
        new_y = current.y + ds * math.sin(0.5 * (current.yaw + new_yaw))
        return HybridNode(new_x, new_y, new_yaw, parent=current, steer_cmd=steer)

    def _search(
        self,
        costmap: BEVTerrainCostmap,
        start: RobotState,
        goal: GoalState
    ) -> Tuple[Optional[Path], PlannerStatus, Dict[str, Any]]:
        """Run Hybrid A* graph search in SE(2)."""
        start_node = HybridNode(start.x, start.y, start.yaw, g_cost=0.0)
        start_idx = self._state_to_grid_index(start_node, costmap)

        open_heap: List[Tuple[float, float, int, HybridNode]] = []
        h_start = self._heuristic(start_node, goal)
        heapq.heappush(open_heap, (h_start, 0.0, 0, start_node))

        closed_set: Set[Tuple[int, int, int]] = set()
        g_costs: Dict[Tuple[int, int, int], float] = {start_idx: 0.0}

        nodes_explored = 0
        counter = 0
        best_node: Optional[HybridNode] = None

        while open_heap and nodes_explored < self.max_nodes:
            f, g, _, curr = heapq.heappop(open_heap)
            curr_idx = self._state_to_grid_index(curr, costmap)

            if curr_idx in closed_set:
                continue
            closed_set.add(curr_idx)
            nodes_explored += 1

            # Goal tolerance check
            dist_to_goal = math.hypot(goal.x - curr.x, goal.y - curr.y)
            if dist_to_goal <= goal.xy_tolerance_m:
                if goal.yaw is None or abs((goal.yaw - curr.yaw + np.pi) % (2.0 * np.pi) - np.pi) <= goal.yaw_tolerance_rad:
                    best_node = curr
                    break

            # Analytic straight connection if close to goal
            if dist_to_goal <= 1.0 and not self.collision_checker.is_segment_collision((curr.x, curr.y), (goal.x, goal.y), costmap):
                goal_yaw = math.atan2(goal.y - curr.y, goal.x - curr.x)
                if goal.yaw is None or abs((goal.yaw - goal_yaw + np.pi) % (2.0 * np.pi) - np.pi) <= goal.yaw_tolerance_rad:
                    best_node = HybridNode(goal.x, goal.y, goal_yaw, g_cost=g + dist_to_goal, parent=curr)
                    break

            # Expand kinematic motion primitives
            for steer in self.steer_angles:
                nxt = self._propagate_primitive(curr, steer)

                # Collision check along footprint
                if self.collision_checker.is_footprint_collision(nxt.x, nxt.y, nxt.yaw, costmap):
                    continue

                r, c = costmap.meter_to_grid(nxt.x, nxt.y)
                c_terrain = self.cost_evaluator.evaluate_cell_cost(r, c, costmap)
                if np.isinf(c_terrain):
                    continue

                step_cost = self.step_size * (1.0 + c_terrain) + 0.05 * abs(steer)
                tentative_g = g + step_cost
                nxt_idx = self._state_to_grid_index(nxt, costmap)

                if tentative_g < g_costs.get(nxt_idx, float("inf")):
                    g_costs[nxt_idx] = tentative_g
                    nxt.g_cost = tentative_g
                    h = self._heuristic(nxt, goal)
                    counter += 1
                    heapq.heappush(open_heap, (tentative_g + h, tentative_g, counter, nxt))

        if best_node is None:
            return None, PlannerStatus.NO_PATH, {"nodes_explored": nodes_explored}

        # Reconstruct path
        path_pts = []
        node_ptr: Optional[HybridNode] = best_node
        while node_ptr is not None:
            path_pts.append([node_ptr.x, node_ptr.y])
            node_ptr = node_ptr.parent
        path_pts.reverse()

        path = Path.from_xy_array(np.array(path_pts, dtype=np.float32))
        diagnostics = {
            "nodes_explored": nodes_explored,
            "raw_waypoints": len(path),
            "final_heading_rad": round(float(best_node.yaw), 3)
        }
        return path, PlannerStatus.SUCCESS, diagnostics
