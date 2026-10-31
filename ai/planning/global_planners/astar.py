"""8-connected grid A* path planner with octile heuristic, terrain weighting, and footprint-validated smoothing."""

from __future__ import annotations

import heapq
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from ai.planning.base import GoalPose, GoalState, PlanningConfig, PlannerStatus, RobotState
from ai.planning.collision import FootprintCollisionChecker
from ai.planning.costs import TerrainAwareCostEvaluator
from ai.planning.global_planners.base import GlobalPlanner
from ai.planning.path import Path


class AStarPlanner(GlobalPlanner):
    """Terrain-aware A* graph search operating on WorldTerrainMap and BEVTerrainCostmap."""

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        super().__init__(name="A*", config=config)
        self.cost_evaluator = TerrainAwareCostEvaluator(self.config)
        self.collision_checker = FootprintCollisionChecker(self.config.vehicle)

        astar_cfg = self.config.astar
        self.heuristic_type = str(astar_cfg.get("heuristic_type", "octile")).lower()
        self.heuristic_weight = float(astar_cfg.get("heuristic_weight", 1.00))
        self.enable_smoothing = bool(astar_cfg.get("enable_smoothing", True))
        self.max_smoothing_iters = int(astar_cfg.get("max_smoothing_iterations", 100))

        # 8-connected neighborhood: (dr, dc, step_dist_multiplier)
        sqrt2 = math.sqrt(2.0)
        self.motions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, sqrt2), (-1, 1, sqrt2), (1, -1, sqrt2), (1, 1, sqrt2)
        ]

    def _grid_coords(self, map_obj: Any, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert metric coordinate to (row, col) across WorldTerrainMap and BEVTerrainCostmap."""
        if hasattr(map_obj, "world_to_grid"):
            return map_obj.world_to_grid(x_m, y_m)
        return map_obj.meter_to_grid(x_m, y_m)

    def _metric_coords(self, map_obj: Any, r: int, c: int) -> Tuple[float, float]:
        """Convert grid index (row, col) to metric coordinates (x, y)."""
        if hasattr(map_obj, "grid_to_world"):
            return map_obj.grid_to_world(r, c)
        return map_obj.grid_to_meter(r, c)

    def _get_resolution(self, map_obj: Any) -> float:
        """Extract metric cell resolution."""
        if hasattr(map_obj, "resolution_m"):
            return float(map_obj.resolution_m)
        return float(getattr(map_obj, "resolution", 0.10))

    def _heuristic(self, r: int, c: int, r_goal: int, c_goal: int, res: float) -> float:
        """Admissible heuristic: Octile distance for 8-connected grids."""
        dr = abs(r_goal - r) * res
        dc = abs(c_goal - c) * res

        if self.heuristic_type == "euclidean":
            h = math.hypot(dr, dc)
        else:
            # Octile distance: (dr + dc) + (sqrt(2) - 2) * min(dr, dc)
            # Never overestimates 8-connected grid movement
            h = (dr + dc) + (math.sqrt(2.0) - 2.0) * min(dr, dc)

        return float(h * self.heuristic_weight)

    def _prune_collinear(self, path_xy: np.ndarray) -> np.ndarray:
        """Remove intermediate points along straight collinear grid segments."""
        if len(path_xy) <= 2:
            return path_xy

        pruned = [path_xy[0]]
        for i in range(1, len(path_xy) - 1):
            p_prev = pruned[-1]
            p_curr = path_xy[i]
            p_next = path_xy[i + 1]

            d1 = p_curr - p_prev
            d2 = p_next - p_curr
            cross = abs(d1[0] * d2[1] - d1[1] * d2[0])
            if cross > 1e-4:
                pruned.append(p_curr)

        pruned.append(path_xy[-1])
        return np.array(pruned, dtype=np.float32)

    def _smooth_path(self, path_xy: np.ndarray, costmap: Any) -> np.ndarray:
        """Shortcut redundant waypoints via line-of-sight footprint collision checks."""
        if len(path_xy) <= 2:
            return path_xy

        # 1. Collinear pruning first
        pruned_xy = self._prune_collinear(path_xy)
        if len(pruned_xy) <= 2:
            return pruned_xy

        # 2. Line-of-sight shortcutting with footprint validation
        smoothed = [pruned_xy[0]]
        curr_idx = 0
        n = len(pruned_xy)
        iters = 0

        while curr_idx < n - 1 and iters < self.max_smoothing_iters:
            iters += 1
            next_idx = n - 1
            # Look ahead as far as possible along remaining waypoints
            while next_idx > curr_idx + 1:
                p1 = (float(pruned_xy[curr_idx, 0]), float(pruned_xy[curr_idx, 1]))
                p2 = (float(pruned_xy[next_idx, 0]), float(pruned_xy[next_idx, 1]))
                # Check segment with footprint clearance
                if not self.collision_checker.is_segment_collision(
                    p1, p2, costmap, step_m=0.08, check_footprint=True
                ):
                    # Valid collision-free shortcut found
                    break
                next_idx -= 1

            smoothed.append(pruned_xy[next_idx])
            curr_idx = next_idx

        return np.array(smoothed, dtype=np.float32)

    def _search(
        self,
        costmap: Any,
        start: Union[RobotState, GoalPose, Tuple[float, float]],
        goal: Union[GoalPose, GoalState, Tuple[float, float]],
    ) -> Tuple[Optional[Path], PlannerStatus, Dict[str, Any]]:
        """Execute A* search on WorldTerrainMap or BEVTerrainCostmap."""
        t_start = time.perf_counter()

        # Map validation
        if costmap is None:
            return None, PlannerStatus.MAP_UNAVAILABLE, {"error": "Map is None"}

        start_x = float(start.x if hasattr(start, "x") else start[0])
        start_y = float(start.y if hasattr(start, "y") else start[1])
        goal_x = float(goal.x if hasattr(goal, "x") else goal[0])
        goal_y = float(goal.y if hasattr(goal, "y") else goal[1])

        # Bounds validation
        if not costmap.is_in_bounds(start_x, start_y):
            return None, PlannerStatus.INVALID_START, {"error": "Start out of map bounds"}

        if not costmap.is_in_bounds(goal_x, goal_y):
            return None, PlannerStatus.INVALID_GOAL, {"error": "Goal out of map bounds"}

        res = self._get_resolution(costmap)
        shape = getattr(costmap, "shape", (costmap.cost.shape[0], costmap.cost.shape[1]))

        # Footprint inflation for chassis obstacle & critical hazard clearance (circumscribed radius ensures full rotation clearance)
        infl_radius = self.config.vehicle.circumscribed_radius_m + self.config.vehicle.collision_margin_m
        infl_cells = max(4, int(np.ceil(infl_radius / res)))
        infl_obstacle: Optional[np.ndarray] = None
        if hasattr(costmap, "obstacle") and costmap.obstacle.any():
            cached_infl = getattr(costmap, "_cached_infl_obstacle", None)
            cached_cells = getattr(costmap, "_cached_infl_cells", None)
            if cached_infl is not None and cached_cells == infl_cells:
                infl_obstacle = cached_infl
            else:
                from scipy.ndimage import binary_dilation
                infl_obstacle = binary_dilation(costmap.obstacle, iterations=infl_cells)
                try:
                    costmap._cached_infl_obstacle = infl_obstacle
                    costmap._cached_infl_cells = infl_cells
                except Exception:
                    pass

        infl_hazard: Optional[np.ndarray] = None
        if hasattr(costmap, "hazard") and (costmap.hazard >= self.cost_evaluator.critical_hazard_threshold).any():
            from scipy.ndimage import binary_dilation
            crit_mask = costmap.hazard >= self.cost_evaluator.critical_hazard_threshold
            infl_hazard = binary_dilation(crit_mask, iterations=infl_cells)

        r_start, c_start = self._grid_coords(costmap, start_x, start_y)
        r_goal, c_goal = self._grid_coords(costmap, goal_x, goal_y)

        # Start / Goal collision validation
        if self.collision_checker.is_point_collision(start_x, start_y, costmap):
            return None, PlannerStatus.START_BLOCKED, {"error": "Start point in collision/hazard"}

        goal_yaw = float(goal.yaw) if hasattr(goal, "yaw") and goal.yaw is not None else 0.0
        if (
            self.collision_checker.is_point_collision(goal_x, goal_y, costmap)
            or self.collision_checker.is_footprint_collision(goal_x, goal_y, goal_yaw, costmap)
            or (infl_obstacle is not None and 0 <= r_goal < shape[0] and 0 <= c_goal < shape[1] and infl_obstacle[r_goal, c_goal])
        ):
            free_pt = self._find_nearest_free_point(
                goal_x, goal_y, costmap, search_radius_m=1.5, check_footprint=True, yaw=goal_yaw,
                infl_obstacle=infl_obstacle, infl_hazard=infl_hazard
            )
            if free_pt is not None:
                goal_x, goal_y = free_pt
                r_goal, c_goal = self._grid_coords(costmap, goal_x, goal_y)
            else:
                return None, PlannerStatus.GOAL_BLOCKED, {"error": "Goal point in collision/hazard"}

        if (r_start, c_start) == (r_goal, c_goal):
            pts = np.array([[start_x, start_y], [goal_x, goal_y]], dtype=np.float32)
            p = Path.from_xy_array(pts)
            lat_ms = (time.perf_counter() - t_start) * 1000.0
            return p, PlannerStatus.SUCCESS, {
                "nodes_explored": 1,
                "planning_latency_ms": lat_ms,
                "path_length_m": float(p.total_length_m),
                "path_cost": 0.0
            }

        # Corridor bounding box to prevent expanding irrelevant distant cells
        bbox_margin_m = 12.0
        bbox_margin_cells = max(30, int(np.ceil(bbox_margin_m / res)))
        r_min = max(0, min(r_start, r_goal) - bbox_margin_cells)
        r_max = min(shape[0] - 1, max(r_start, r_goal) + bbox_margin_cells)
        c_min = max(0, min(c_start, c_goal) - bbox_margin_cells)
        c_max = min(shape[1] - 1, max(c_start, c_goal) + bbox_margin_cells)

        # Priority queue stores tuples: (f_score, g_score, r, c)
        start_h = self._heuristic(r_start, c_start, r_goal, c_goal, res)
        open_set: List[Tuple[float, float, int, int]] = []
        heapq.heappush(open_set, (start_h, 0.0, r_start, c_start))

        g_costs: Dict[Tuple[int, int], float] = {(r_start, c_start): 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()

        nodes_explored = 0
        goal_reached_coord: Optional[Tuple[int, int]] = None

        while open_set:
            f, g, curr_r, curr_c = heapq.heappop(open_set)

            if (curr_r, curr_c) in closed_set:
                continue
            closed_set.add((curr_r, curr_c))
            nodes_explored += 1

            # Goal check
            if (curr_r, curr_c) == (r_goal, c_goal):
                goal_reached_coord = (curr_r, curr_c)
                break

            # Explore 8 neighbors
            for dr, dc, step_mult in self.motions:
                nr, nc = curr_r + dr, curr_c + dc

                if not (r_min <= nr <= r_max and c_min <= nc <= c_max):
                    continue
                if (nr, nc) in closed_set:
                    continue

                # Chassis clearance check: do not plan paths too close to lethal obstacles/hazards
                if infl_obstacle is not None and infl_obstacle[nr, nc] and (nr, nc) != (r_goal, c_goal):
                    continue
                if infl_hazard is not None and infl_hazard[nr, nc] and (nr, nc) != (r_goal, c_goal):
                    continue

                # Cell terrain cost and hard safety check
                cell_terrain = self.cost_evaluator.evaluate_cell_cost(nr, nc, costmap)
                if np.isinf(cell_terrain):
                    continue

                step_cost = res * step_mult * (1.0 + cell_terrain)
                tentative_g = g + step_cost

                if tentative_g < g_costs.get((nr, nc), float("inf")):
                    g_costs[(nr, nc)] = tentative_g
                    came_from[(nr, nc)] = (curr_r, curr_c)
                    h = self._heuristic(nr, nc, r_goal, c_goal, res)
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, nr, nc))

        lat_ms = (time.perf_counter() - t_start) * 1000.0

        if goal_reached_coord is None:
            return None, PlannerStatus.NO_PATH, {
                "nodes_explored": nodes_explored,
                "planning_latency_ms": lat_ms,
            }

        # Reconstruct path
        path_cells: List[Tuple[int, int]] = [goal_reached_coord]
        curr = goal_reached_coord
        while curr in came_from:
            curr = came_from[curr]
            path_cells.append(curr)
        path_cells.reverse()

        # Convert to metric coordinates
        raw_coords = []
        for r, c in path_cells:
            xm, ym = self._metric_coords(costmap, r, c)
            raw_coords.append([xm, ym])

        raw_coords_arr = np.array(raw_coords, dtype=np.float32)
        # Ensure exact start and goal endpoints
        raw_coords_arr[0] = [start_x, start_y]
        raw_coords_arr[-1] = [goal_x, goal_y]

        if self.enable_smoothing:
            smoothed_coords = self._smooth_path(raw_coords_arr, costmap)
            final_path = Path.from_xy_array(smoothed_coords).interpolate(step_m=0.10)
            final_path.is_smoothed = True
        else:
            final_path = Path.from_xy_array(raw_coords_arr)

        total_cost = g_costs.get(goal_reached_coord, 0.0)

        diagnostics = {
            "nodes_explored": nodes_explored,
            "planning_latency_ms": lat_ms,
            "raw_waypoints": len(raw_coords_arr),
            "smoothed_waypoints": len(final_path),
            "is_smoothed": final_path.is_smoothed,
            "path_length_m": float(final_path.total_length_m),
            "path_cost": float(total_cost),
        }
        return final_path, PlannerStatus.SUCCESS, diagnostics

    def _find_nearest_free_point(
        self,
        gx: float,
        gy: float,
        costmap: Any,
        search_radius_m: float = 1.5,
        check_footprint: bool = False,
        yaw: float = 0.0,
        infl_obstacle: Optional[np.ndarray] = None,
        infl_hazard: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, float]]:
        """Find the closest collision-free grid point to (gx, gy) within search_radius_m."""
        res = self._get_resolution(costmap)
        max_steps = max(1, int(math.ceil(search_radius_m / res)))
        best_pt: Optional[Tuple[float, float]] = None
        best_d = float("inf")
        shape = getattr(costmap, "shape", (costmap.cost.shape[0], costmap.cost.shape[1]))
        for step_x in range(-max_steps, max_steps + 1):
            for step_y in range(-max_steps, max_steps + 1):
                d = math.hypot(step_x * res, step_y * res)
                if d > search_radius_m or d >= best_d:
                    continue
                cand_x = gx + step_x * res
                cand_y = gy + step_y * res
                if not costmap.is_in_bounds(cand_x, cand_y):
                    continue
                r_cand, c_cand = self._grid_coords(costmap, cand_x, cand_y)
                if not (0 <= r_cand < shape[0] and 0 <= c_cand < shape[1]):
                    continue
                if infl_obstacle is not None and infl_obstacle[r_cand, c_cand]:
                    continue
                if infl_hazard is not None and infl_hazard[r_cand, c_cand]:
                    continue
                if self.collision_checker.is_point_collision(cand_x, cand_y, costmap):
                    continue
                if check_footprint and self.collision_checker.is_footprint_collision(cand_x, cand_y, yaw, costmap):
                    continue
                best_d = d
                best_pt = (cand_x, cand_y)
        return best_pt

