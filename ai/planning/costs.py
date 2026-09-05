"""Terrain-aware cost evaluation functions for candidate paths and trajectories across WorldTerrainMap and BEVTerrainCostmap."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from ai.planning.base import PlanningConfig


class TerrainAwareCostEvaluator:
    """Calculates physical, geometric, and safety costs for planning search graphs."""

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        cfg = config or PlanningConfig()
        weights = cfg.cost_weights
        self.w_dist = float(weights.get("w_distance", 1.00))
        self.w_trav = float(weights.get("w_traversability", 0.80))
        self.w_cost = float(weights.get("w_cost", 1.50))
        self.w_hazard = float(weights.get("w_hazard", 2.50))
        self.w_slope = float(weights.get("w_slope", 1.20))
        self.w_rough = float(weights.get("w_roughness", 0.60))
        self.w_height = float(weights.get("w_height", 1.00))
        self.w_unk = float(weights.get("w_unknown", 1.00))
        self.w_curv = float(weights.get("w_curvature", 0.40))

        unknown_policy = getattr(cfg, "unknown_policy", {})
        self.unknown_is_lethal = bool(unknown_policy.get("unknown_is_lethal", False))
        self.unknown_cost_penalty = float(unknown_policy.get("unknown_cost_penalty", 0.80))

        tolerances = getattr(cfg, "tolerances", {})
        self.lethal_cost_threshold = float(tolerances.get("lethal_cost_threshold", 0.85))
        self.critical_hazard_threshold = float(tolerances.get("critical_hazard_threshold", 0.70))

        self.max_safe_slope_deg = 20.0
        self.max_safe_step_m = 0.12

    def _grid_coords(self, map_obj: Any, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert metric coordinate to (row, col) across WorldTerrainMap and BEVTerrainCostmap."""
        if hasattr(map_obj, "world_to_grid"):
            return map_obj.world_to_grid(x_m, y_m)
        return map_obj.meter_to_grid(x_m, y_m)

    def evaluate_cell_cost(self, r: int, c: int, costmap: Any) -> float:
        """Calculate the intrinsic terrain cost multiplier for a single grid cell.

        Enforces hard safety precedence: lethal obstacles, critical hazards (>= 0.70),
        and unknown cells (under conservative policy) return infinity.
        """
        shape = getattr(costmap, "shape", (costmap.cost.shape[0], costmap.cost.shape[1]))
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            return float("inf")

        # 1. Hard Obstacle Precedence
        if hasattr(costmap, "obstacle") and costmap.obstacle[r, c]:
            return float("inf")

        # 2. Critical Hazard Precedence (Phase 12: hazard >= 0.70 is lethal)
        if hasattr(costmap, "hazard") and float(costmap.hazard[r, c]) >= self.critical_hazard_threshold:
            return float("inf")

        # 3. Lethal Cost Precedence
        base_c = float(costmap.cost[r, c]) if hasattr(costmap, "cost") else 0.0
        if base_c >= self.lethal_cost_threshold:
            return float("inf")

        # 4. Unknown Space Policy (Phase 12: UNKNOWN != SAFE)
        is_unknown = bool(costmap.unknown[r, c]) if hasattr(costmap, "unknown") else False
        if is_unknown and self.unknown_is_lethal:
            return float("inf")

        # Continuous weighted terrain cost
        trav = float(costmap.traversability[r, c]) if hasattr(costmap, "traversability") else 0.5
        haz = float(costmap.hazard[r, c]) if hasattr(costmap, "hazard") else 0.0

        if hasattr(costmap, "slope_deg"):
            slope = float(costmap.slope_deg[r, c])
        elif hasattr(costmap, "slope"):
            slope = float(costmap.slope[r, c])
        else:
            slope = 0.0

        rough = float(costmap.roughness[r, c]) if hasattr(costmap, "roughness") else 0.0
        h_var = float(costmap.height_var[r, c]) if hasattr(costmap, "height_var") else 0.0
        unk_val = self.unknown_cost_penalty if is_unknown else 0.0

        cell_cost = (
            self.w_cost * base_c +
            self.w_trav * (1.0 - trav) +
            self.w_hazard * haz +
            self.w_slope * min(slope / self.max_safe_slope_deg, 2.0) +
            self.w_rough * rough +
            self.w_height * min(h_var / self.max_safe_step_m, 2.0) +
            self.w_unk * unk_val
        )
        return float(cell_cost)

    def evaluate_step_cost(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        costmap: Any,
        delta_yaw: float = 0.0
    ) -> float:
        """Compute the edge traversal cost between two metric coordinates."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = float(np.hypot(dx, dy))

        if not costmap.is_in_bounds(p2[0], p2[1]):
            return float("inf")

        r2, c2 = self._grid_coords(costmap, p2[0], p2[1])
        c_terrain = self.evaluate_cell_cost(r2, c2, costmap)

        if np.isinf(c_terrain):
            return float("inf")

        motion_cost = dist * (self.w_dist + c_terrain)
        curv_cost = self.w_curv * abs(delta_yaw)

        return float(motion_cost + curv_cost)

