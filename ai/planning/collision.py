"""Collision and clearance checking using vehicle footprint, WorldTerrainMap, and BEVTerrainCostmap."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import numpy as np

from ai.planning.base import VehicleKinematics


class FootprintCollisionChecker:
    """Evaluates spatial collision and safety margins for a moving UGV across WorldTerrainMap or BEVTerrainCostmap."""

    def __init__(self, vehicle: Optional[VehicleKinematics] = None) -> None:
        self.vehicle = vehicle or VehicleKinematics()
        self.length = float(self.vehicle.length_m)
        self.width = float(self.vehicle.width_m)
        self.r_inscribed = min(self.length, self.width) / 2.0
        self.r_infl = float(self.vehicle.inflation_radius_m)
        self.collision_margin = float(self.vehicle.collision_margin_m)


    def _grid_coords(self, map_obj: Any, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert metric coordinate to (row, col) across WorldTerrainMap and BEVTerrainCostmap."""
        if hasattr(map_obj, "world_to_grid"):
            return map_obj.world_to_grid(x_m, y_m)
        return map_obj.meter_to_grid(x_m, y_m)

    def _get_resolution(self, map_obj: Any) -> float:
        """Extract metric cell resolution."""
        if hasattr(map_obj, "resolution_m"):
            return float(map_obj.resolution_m)
        return float(getattr(map_obj, "resolution", 0.10))

    def _is_grid_in_bounds(self, map_obj: Any, r: int, c: int) -> bool:
        """Verify row and column are inside raster limits."""
        if hasattr(map_obj, "is_grid_in_bounds"):
            return map_obj.is_grid_in_bounds(r, c)
        shape = getattr(map_obj, "shape", (map_obj.cost.shape[0], map_obj.cost.shape[1]))
        return (0 <= r < shape[0]) and (0 <= c < shape[1])

    def is_point_collision(
        self,
        x_m: float,
        y_m: float,
        costmap: Any,
        max_cost: float = 0.65,
        unknown_is_lethal: bool = False,
        critical_hazard_threshold: float = 0.70,
    ) -> bool:
        """Check if a single metric point is in collision, lethal obstacle, critical hazard, or out-of-bounds."""
        if not costmap.is_in_bounds(x_m, y_m):
            return True

        r, c = self._grid_coords(costmap, x_m, y_m)
        if not self._is_grid_in_bounds(costmap, r, c):
            return True

        # 1. Hard Obstacle Check
        if hasattr(costmap, "obstacle") and costmap.obstacle[r, c]:
            return True

        # 2. Critical Hazard Check (Phase 4/9/12 contract: >= 0.70 is lethal)
        if hasattr(costmap, "hazard") and float(costmap.hazard[r, c]) >= critical_hazard_threshold:
            return True

        # 3. Unknown-Space Policy (Phase 12: UNKNOWN != SAFE)
        if unknown_is_lethal and hasattr(costmap, "unknown") and costmap.unknown[r, c]:
            return True

        # 4. Numerical Cost Threshold
        if hasattr(costmap, "cost") and float(costmap.cost[r, c]) > max_cost:
            return True

        return False

    def is_circle_collision(
        self,
        x_m: float,
        y_m: float,
        radius_m: float,
        costmap: Any,
        max_cost: float = 0.65,
        unknown_is_lethal: bool = False,
        critical_hazard_threshold: float = 0.70,
    ) -> bool:
        """Check if a circle around (x, y) intersects an obstacle, hazard, or lethal cost."""
        if not costmap.is_in_bounds(x_m, y_m):
            return True

        res = self._get_resolution(costmap)
        r_cells = max(1, int(np.ceil(radius_m / res)))
        r_center, c_center = self._grid_coords(costmap, x_m, y_m)

        shape = getattr(costmap, "shape", (costmap.cost.shape[0], costmap.cost.shape[1]))
        r_min = max(0, r_center - r_cells)
        r_max = min(shape[0] - 1, r_center + r_cells)
        c_min = max(0, c_center - r_cells)
        c_max = min(shape[1] - 1, c_center + r_cells)

        # Compute circular distance mask within bounding box
        dr = (np.arange(r_min, r_max + 1) - r_center) * res
        dc = (np.arange(c_min, c_max + 1) - c_center) * res
        grid_dr, grid_dc = np.meshgrid(dr, dc, indexing="ij")
        circle_mask = (grid_dr ** 2 + grid_dc ** 2) <= (radius_m ** 2)

        sub_cost = costmap.cost[r_min:r_max+1, c_min:c_max+1]
        sub_obs = costmap.obstacle[r_min:r_max+1, c_min:c_max+1] if hasattr(costmap, "obstacle") else np.zeros_like(sub_cost, dtype=bool)

        if (sub_obs & circle_mask).any() or (((sub_cost > max_cost)) & circle_mask).any():
            return True

        if hasattr(costmap, "hazard"):
            sub_hazard = costmap.hazard[r_min:r_max+1, c_min:c_max+1]
            if ((sub_hazard >= critical_hazard_threshold) & circle_mask).any():
                return True

        if unknown_is_lethal and hasattr(costmap, "unknown"):
            sub_unknown = costmap.unknown[r_min:r_max+1, c_min:c_max+1]
            if (sub_unknown & circle_mask).any():
                return True

        return False


    def get_footprint_corners(
        self,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        margin_m: float = 0.0,
    ) -> np.ndarray:
        """Calculate the 4 metric corners of the oriented bounding box (4, 2)."""
        hl = (self.length / 2.0) + margin_m
        hw = (self.width / 2.0) + margin_m

        local_corners = np.array([
            [hl, hw],
            [hl, -hw],
            [-hl, -hw],
            [-hl, hw]
        ], dtype=np.float32)

        c = np.cos(yaw_rad)
        s = np.sin(yaw_rad)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)

        rotated = (R @ local_corners.T).T
        rotated[:, 0] += x_m
        rotated[:, 1] += y_m
        return rotated

    def is_footprint_collision(
        self,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        costmap: Any,
        max_cost: float = 0.65,
        unknown_is_lethal: bool = False,
        margin_m: float = 0.0,
        critical_hazard_threshold: float = 0.70,
    ) -> bool:
        """Evaluate if the oriented rectangular vehicle footprint intersects any obstacle, hazard, or unknown cell."""
        # 1. Fast Inscribed Circle Shortcut
        eff_inscribed = self.r_inscribed + margin_m
        if self.is_circle_collision(
            x_m, y_m, eff_inscribed, costmap, max_cost, unknown_is_lethal, critical_hazard_threshold
        ):
            return True

        # 2. Check oriented footprint corners & perimeter samples
        corners = self.get_footprint_corners(x_m, y_m, yaw_rad, margin_m=margin_m)
        edge_step_m = 0.05

        for i in range(4):
            cx, cy = float(corners[i, 0]), float(corners[i, 1])
            if self.is_point_collision(cx, cy, costmap, max_cost, unknown_is_lethal, critical_hazard_threshold):
                return True

            next_i = (i + 1) % 4
            nx, ny = float(corners[next_i, 0]), float(corners[next_i, 1])
            edge_dist = float(np.hypot(nx - cx, ny - cy))
            n_sub = max(1, int(np.ceil(edge_dist / edge_step_m)))
            for k in range(1, n_sub):
                alpha = k / float(n_sub)
                px = cx + alpha * (nx - cx)
                py = cy + alpha * (ny - cy)
                if self.is_point_collision(px, py, costmap, max_cost, unknown_is_lethal, critical_hazard_threshold):
                    return True

        # 3. Check interior sample grid (2D footprint coverage)
        hl = (self.length / 2.0) + margin_m
        hw = (self.width / 2.0) + margin_m
        c = np.cos(yaw_rad)
        s = np.sin(yaw_rad)
        dx_samples = np.linspace(-hl * 0.85, hl * 0.85, 7)
        dy_samples = np.linspace(-hw * 0.80, hw * 0.80, 5)
        for dx in dx_samples:
            for dy in dy_samples:
                ix = x_m + (dx * c - dy * s)
                iy = y_m + (dx * s + dy * c)
                if self.is_point_collision(ix, iy, costmap, max_cost, unknown_is_lethal, critical_hazard_threshold):
                    return True

        return False


    def is_segment_collision(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        costmap: Any,
        step_m: float = 0.05,
        max_cost: float = 0.65,
        unknown_is_lethal: bool = False,
        check_footprint: bool = False,
        yaw_rad: Optional[float] = None,
    ) -> bool:
        """Check line segment between two metric points for collisions."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = float(np.hypot(dx, dy))

        if yaw_rad is None:
            yaw_rad = float(np.arctan2(dy, dx)) if dist > 1e-4 else 0.0

        if dist < 1e-4:
            if check_footprint:
                return self.is_footprint_collision(p1[0], p1[1], yaw_rad, costmap, max_cost, unknown_is_lethal)
            return self.is_point_collision(p1[0], p1[1], costmap, max_cost, unknown_is_lethal)

        steps = max(2, int(np.ceil(dist / step_m)))
        for alpha in np.linspace(0.0, 1.0, steps):
            x = p1[0] + alpha * dx
            y = p1[1] + alpha * dy
            if check_footprint:
                if self.is_footprint_collision(x, y, yaw_rad, costmap, max_cost, unknown_is_lethal):
                    return True
            else:
                if self.is_point_collision(x, y, costmap, max_cost, unknown_is_lethal):
                    return True
        return False

    def get_clearance(
        self,
        x_m: float,
        y_m: float,
        costmap: Any,
        max_search_radius_m: float = 2.0,
    ) -> float:
        """Approximate distance to nearest obstacle cell in meters."""
        if not costmap.is_in_bounds(x_m, y_m):
            return 0.0

        r_center, c_center = self._grid_coords(costmap, x_m, y_m)
        if not self._is_grid_in_bounds(costmap, r_center, c_center):
            return 0.0

        if hasattr(costmap, "obstacle") and costmap.obstacle[r_center, c_center]:
            return 0.0

        res = self._get_resolution(costmap)
        r_search = int(np.ceil(max_search_radius_m / res))
        shape = getattr(costmap, "shape", (costmap.cost.shape[0], costmap.cost.shape[1]))

        r_min = max(0, r_center - r_search)
        r_max = min(shape[0] - 1, r_center + r_search)
        c_min = max(0, c_center - r_search)
        c_max = min(shape[1] - 1, c_center + r_search)

        sub_obs = costmap.obstacle[r_min:r_max+1, c_min:c_max+1] if hasattr(costmap, "obstacle") else np.zeros((r_max-r_min+1, c_max-c_min+1), dtype=bool)

        # Also treat critical hazards as obstacles for clearance
        if hasattr(costmap, "hazard"):
            sub_hazard = costmap.hazard[r_min:r_max+1, c_min:c_max+1]
            sub_obs = sub_obs | (sub_hazard >= 0.70)

        obs_indices = np.argwhere(sub_obs)
        if len(obs_indices) == 0:
            return float(max_search_radius_m)

        dr = (obs_indices[:, 0] + r_min - r_center) * res
        dc = (obs_indices[:, 1] + c_min - c_center) * res
        dists = np.hypot(dr, dc)
        return float(np.min(dists))

