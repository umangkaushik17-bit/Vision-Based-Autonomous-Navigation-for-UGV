"""Quantitative evaluation metrics for candidate paths and tracking trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.collision import FootprintCollisionChecker
from ai.planning.path import Path


@dataclass
class PathCostMetrics:
    """Quantitative performance and safety metrics for a planned path."""
    path_length_m: float = 0.0
    mean_cost: float = 0.0
    total_terrain_cost: float = 0.0
    total_risk: float = 0.0
    min_clearance_m: float = 0.0
    avg_clearance_m: float = 0.0
    mean_slope_deg: float = 0.0
    max_slope_deg: float = 0.0
    unknown_exposure_ratio: float = 0.0
    mean_roughness: float = 0.0
    mean_curvature: float = 0.0
    max_curvature: float = 0.0
    smoothness_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "path_length_m": round(self.path_length_m, 3),
            "mean_cost": round(self.mean_cost, 4),
            "total_terrain_cost": round(self.total_terrain_cost, 4),
            "total_risk": round(self.total_risk, 4),
            "min_clearance_m": round(self.min_clearance_m, 3),
            "avg_clearance_m": round(self.avg_clearance_m, 3),
            "mean_slope_deg": round(self.mean_slope_deg, 2),
            "max_slope_deg": round(self.max_slope_deg, 2),
            "unknown_exposure_ratio": round(self.unknown_exposure_ratio, 4),
            "mean_roughness": round(self.mean_roughness, 4),
            "mean_curvature": round(self.mean_curvature, 4),
            "max_curvature": round(self.max_curvature, 4),
            "smoothness_score": round(self.smoothness_score, 4)
        }


class PathMetricsCalculator:
    """Computes multidimensional path quality, clearance, and safety exposure."""

    def __init__(self) -> None:
        self.checker = FootprintCollisionChecker()

    def evaluate_path(self, path: Path, costmap: BEVTerrainCostmap) -> PathCostMetrics:
        """Analyze path points against the underlying BEVTerrainCostmap."""
        if len(path) == 0:
            return PathCostMetrics()

        interp = path.interpolate(step_m=0.05)
        n = len(interp.points)

        costs: List[float] = []
        risks: List[float] = []
        clearances: List[float] = []
        slopes: List[float] = []
        unknowns: List[int] = []
        roughness_vals: List[float] = []
        curvatures: List[float] = []

        for pt in interp.points:
            if costmap.is_in_bounds(pt.x, pt.y):
                r, c = costmap.meter_to_grid(pt.x, pt.y)
                costs.append(float(costmap.cost[r, c]))
                risks.append(float(costmap.hazard[r, c]))
                slopes.append(float(costmap.slope_deg[r, c]))
                unknowns.append(1 if costmap.unknown[r, c] else 0)
                roughness_vals.append(float(costmap.roughness[r, c]))
            else:
                costs.append(1.0)
                risks.append(1.0)
                slopes.append(0.0)
                unknowns.append(1)
                roughness_vals.append(0.0)

            clearance = self.checker.get_clearance(pt.x, pt.y, costmap)
            clearances.append(clearance)
            curvatures.append(abs(pt.curvature))

        path_len = float(interp.length)
        mean_c = float(np.mean(costs)) if costs else 0.0
        tot_c = float(np.sum(costs) * 0.05)
        tot_risk = float(np.sum(risks) * 0.05)
        min_clr = float(np.min(clearances)) if clearances else 0.0
        avg_clr = float(np.mean(clearances)) if clearances else 0.0
        mean_slp = float(np.mean(slopes)) if slopes else 0.0
        max_slp = float(np.max(slopes)) if slopes else 0.0
        unk_ratio = float(np.mean(unknowns)) if unknowns else 0.0
        mean_rgh = float(np.mean(roughness_vals)) if roughness_vals else 0.0
        mean_curv = float(np.mean(curvatures)) if curvatures else 0.0
        max_curv = float(np.max(curvatures)) if curvatures else 0.0
        smoothness = float(np.sum(np.array(curvatures)**2) * 0.05)

        return PathCostMetrics(
            path_length_m=path_len,
            mean_cost=mean_c,
            total_terrain_cost=tot_c,
            total_risk=tot_risk,
            min_clearance_m=min_clr,
            avg_clearance_m=avg_clr,
            mean_slope_deg=mean_slp,
            max_slope_deg=max_slp,
            unknown_exposure_ratio=unk_ratio,
            mean_roughness=mean_rgh,
            mean_curvature=mean_curv,
            max_curvature=max_curv,
            smoothness_score=smoothness
        )
