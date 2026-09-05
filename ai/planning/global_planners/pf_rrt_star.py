"""MoveBase3D-inspired terrain-aware Potential Field RRT* (PF-RRT*) global planner."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import GoalState, PlanningConfig, PlannerStatus, RobotState
from ai.planning.collision import FootprintCollisionChecker
from ai.planning.costs import TerrainAwareCostEvaluator
from ai.planning.global_planners.base import GlobalPlanner
from ai.planning.path import Path


class RRTNode:
    """A search tree node in 2D metric space."""
    def __init__(self, x: float, y: float, cost: float = 0.0, parent: Optional[RRTNode] = None) -> None:
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)


class PFRRTStarPlanner(GlobalPlanner):
    """Terrain-aware PF-RRT* global planner implementing biased sampling and asymptotic rewiring."""

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        super().__init__(name="PF-RRT*", config=config)
        self.cost_evaluator = TerrainAwareCostEvaluator(self.config)
        self.collision_checker = FootprintCollisionChecker(self.config.vehicle)

        cfg = self.config.pf_rrt_star
        self.max_iterations = int(cfg.get("max_iterations", 1500))
        self.step_size = float(cfg.get("step_size_m", 0.30))
        self.goal_bias = float(cfg.get("goal_bias", 0.15))
        self.terrain_bias = float(cfg.get("terrain_bias", 0.35))
        self.gamma = float(cfg.get("search_radius_gamma", 1.50))
        self.max_search_radius = float(cfg.get("max_search_radius_m", 1.20))
        self.random_seed = cfg.get("random_seed", 42)

    def _sample_point(
        self,
        costmap: BEVTerrainCostmap,
        goal: GoalState,
        rng: np.random.Generator
    ) -> Tuple[float, float]:
        """Draw a metric sample point with goal and terrain cost bias."""
        p = rng.random()
        x_min, x_max = costmap.x_range_m
        y_min, y_max = costmap.y_range_m

        # 1. Goal bias
        if p < self.goal_bias:
            return (goal.x, goal.y)

        # 2. Terrain-aware rejection sampling
        if p < self.goal_bias + self.terrain_bias:
            for _ in range(5):
                rx = float(rng.uniform(x_min, x_max))
                ry = float(rng.uniform(y_min, y_max))
                if costmap.is_in_bounds(rx, ry):
                    r, c = costmap.meter_to_grid(rx, ry)
                    if not costmap.obstacle[r, c]:
                        acceptance_prob = 1.0 - min(costmap.cost[r, c], 0.90)
                        if rng.random() < acceptance_prob:
                            return (rx, ry)

        # 3. Uniform exploration sample
        return (float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)))

    def _steer(self, from_node: RRTNode, to_pt: Tuple[float, float]) -> RRTNode:
        """Step from tree node toward sample point by at most step_size."""
        dx = to_pt[0] - from_node.x
        dy = to_pt[1] - from_node.y
        dist = math.hypot(dx, dy)
        if dist <= self.step_size:
            return RRTNode(to_pt[0], to_pt[1], parent=from_node)
        scale = self.step_size / max(dist, 1e-6)
        nx = from_node.x + dx * scale
        ny = from_node.y + dy * scale
        return RRTNode(nx, ny, parent=from_node)

    def _evaluate_edge_cost(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        costmap: BEVTerrainCostmap
    ) -> Optional[float]:
        """Verify edge feasibility and compute integrated terrain cost."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            return 0.0

        steps = max(2, int(math.ceil(dist / 0.05)))
        accum_cost = 0.0

        for alpha in np.linspace(0.0, 1.0, steps):
            x = p1[0] + alpha * dx
            y = p1[1] + alpha * dy
            if not costmap.is_in_bounds(x, y):
                return None
            r, c = costmap.meter_to_grid(x, y)
            if costmap.obstacle[r, c] or costmap.cost[r, c] >= 0.85:
                return None

            c_cell = self.cost_evaluator.evaluate_cell_cost(r, c, costmap)
            if np.isinf(c_cell):
                return None
            accum_cost += c_cell

        avg_c = accum_cost / steps
        return float(dist * (1.0 + avg_c))

    def _search(
        self,
        costmap: BEVTerrainCostmap,
        start: RobotState,
        goal: GoalState
    ) -> Tuple[Optional[Path], PlannerStatus, Dict[str, Any]]:
        """Run PF-RRT* tree expansion and rewiring."""
        rng = np.random.default_rng(self.random_seed)
        root = RRTNode(start.x, start.y, cost=0.0, parent=None)
        tree: List[RRTNode] = [root]

        best_goal_node: Optional[RRTNode] = None
        best_goal_cost = float("inf")
        nodes_explored = 0

        for iteration in range(self.max_iterations):
            nodes_explored += 1
            rand_pt = self._sample_point(costmap, goal, rng)

            # Nearest neighbor
            dists = [math.hypot(n.x - rand_pt[0], n.y - rand_pt[1]) for n in tree]
            nearest_idx = int(np.argmin(dists))
            nearest_node = tree[nearest_idx]

            # Steer
            new_node = self._steer(nearest_node, rand_pt)

            # Check edge collision
            edge_cost = self._evaluate_edge_cost(nearest_node.position, new_node.position, costmap)
            if edge_cost is None:
                continue

            new_node.cost = nearest_node.cost + edge_cost
            new_node.parent = nearest_node

            # RRT* Rewiring Neighborhood
            n_nodes = len(tree)
            radius = min(
                self.max_search_radius,
                self.gamma * math.sqrt(math.log(max(n_nodes, 2)) / n_nodes)
            )

            # Find near neighbors
            near_indices = [
                i for i, n in enumerate(tree)
                if math.hypot(n.x - new_node.x, n.y - new_node.y) <= radius
            ]

            # Choose parent with minimum cumulative cost
            min_cost = new_node.cost
            best_parent = nearest_node
            for idx in near_indices:
                cand = tree[idx]
                c_edge = self._evaluate_edge_cost(cand.position, new_node.position, costmap)
                if c_edge is not None and cand.cost + c_edge < min_cost:
                    min_cost = cand.cost + c_edge
                    best_parent = cand

            new_node.cost = min_cost
            new_node.parent = best_parent
            tree.append(new_node)

            # Rewire existing tree neighbors through new_node
            for idx in near_indices:
                cand = tree[idx]
                c_edge = self._evaluate_edge_cost(new_node.position, cand.position, costmap)
                if c_edge is not None and new_node.cost + c_edge < cand.cost:
                    cand.cost = new_node.cost + c_edge
                    cand.parent = new_node

            # Check if close to goal
            dist_to_goal = math.hypot(new_node.x - goal.x, new_node.y - goal.y)
            if dist_to_goal <= goal.xy_tolerance_m:
                c_goal_edge = self._evaluate_edge_cost(new_node.position, (goal.x, goal.y), costmap)
                if c_goal_edge is not None:
                    total_cand_cost = new_node.cost + c_goal_edge
                    if total_cand_cost < best_goal_cost:
                        goal_node = RRTNode(goal.x, goal.y, cost=total_cand_cost, parent=new_node)
                        best_goal_node = goal_node
                        best_goal_cost = total_cand_cost

        if best_goal_node is None:
            return None, PlannerStatus.NO_PATH, {
                "nodes_explored": nodes_explored,
                "tree_size": len(tree)
            }

        # Reconstruct path
        path_pts = []
        curr: Optional[RRTNode] = best_goal_node
        while curr is not None:
            path_pts.append([curr.x, curr.y])
            curr = curr.parent
        path_pts.reverse()

        path = Path.from_xy_array(np.array(path_pts, dtype=np.float32))
        diagnostics = {
            "nodes_explored": nodes_explored,
            "tree_size": len(tree),
            "final_cost": round(best_goal_cost, 3),
            "raw_waypoints": len(path)
        }
        return path, PlannerStatus.SUCCESS, diagnostics
