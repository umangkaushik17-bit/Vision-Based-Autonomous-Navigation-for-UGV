"""Parametric terrain cost models combining learned traversability with physical mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CostModelWeights:
    """Weights balancing different physical and learned perceptual risk factors."""

    w_learned: float = 0.40  # Weight on vision-learned non-traversability (1 - trav_score)
    w_slope: float = 0.25    # Weight on geometric terrain slope angle
    w_flatness: float = 0.15 # Weight on surface roughness / residual variance
    w_height: float = 0.10   # Weight on local step-height variation
    w_obstacle: float = 0.10 # Weight on discrete detected obstacles
    unknown_cost: float = 0.50  # Cost assigned to unobserved / out-of-view cells

    # Physical vehicle capability constraints (engineering initial simulation baselines)
    max_safe_slope_deg: float = 20.0     # Slope above which cost escalates rapidly
    critical_slope_deg: float = 28.0     # Critical rollover slope threshold (cost forced to 1.0)
    max_step_height_m: float = 0.12      # Maximum vertical step height passable by wheels (meters)
    critical_step_height_m: float = 0.20 # Step height causing high-centering (cost forced to 1.0)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CostModelWeights:
        """Load vehicle-specific safety parameters from a YAML configuration file."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        slope_cfg = data.get("slope_thresholds", {})
        step_cfg = data.get("step_thresholds", {})
        weights_cfg = data.get("cost_weights", {})

        return cls(
            w_learned=float(weights_cfg.get("w_learned", 0.40)),
            w_slope=float(weights_cfg.get("w_slope", 0.25)),
            w_flatness=float(weights_cfg.get("w_flatness", 0.15)),
            w_height=float(weights_cfg.get("w_height", 0.10)),
            w_obstacle=float(weights_cfg.get("w_obstacle", 0.10)),
            unknown_cost=float(weights_cfg.get("unknown_cost", 0.50)),
            max_safe_slope_deg=float(slope_cfg.get("max_safe_slope_deg", 20.0)),
            critical_slope_deg=float(slope_cfg.get("critical_slope_deg", 28.0)),
            max_step_height_m=float(step_cfg.get("max_step_height_m", 0.12)),
            critical_step_height_m=float(step_cfg.get("critical_step_height_m", 0.20))
        )


class TraversabilityCostModel:
    """Evaluates driving cost from multi-factor terrain inputs.

    Cost formulation:
      C_total = w_learned * C_learned + w_slope * C_slope + w_flat * C_flat + w_height * C_height + w_obs * C_obs
    """

    def __init__(self, weights: Optional[CostModelWeights] = None) -> None:
        self.weights = weights or CostModelWeights()

    def compute_cell_cost(
        self,
        trav_score: float,
        slope_deg: float = 0.0,
        flatness: float = 0.0,
        height_var_m: float = 0.0,
        has_obstacle: bool = False,
        is_unknown: bool = False
    ) -> float:
        """Compute scalar navigation traversal cost in [0.0, 1.0] for a single grid cell.

        0.0 = completely free / flat / safe to drive
        1.0 = lethal obstacle / impassable rollover slope / step obstacle
        """
        if is_unknown:
            return float(self.weights.unknown_cost)

        # Immediate hard vehicle physical failure cutoffs
        if has_obstacle or slope_deg >= self.weights.critical_slope_deg or height_var_m >= self.weights.critical_step_height_m:
            return 1.0

        # Component costs in [0.0, 1.0]
        c_learned = 1.0 - float(np.clip(trav_score, 0.0, 1.0))
        c_slope = float(np.clip(slope_deg / self.weights.max_safe_slope_deg, 0.0, 1.0))
        c_flatness = float(np.clip(flatness / 0.10, 0.0, 1.0))  # 0.10 normalized variance cutoff
        c_height = float(np.clip(height_var_m / self.weights.max_step_height_m, 0.0, 1.0))
        c_obs = 1.0 if has_obstacle else 0.0

        total = (
            self.weights.w_learned * c_learned +
            self.weights.w_slope * c_slope +
            self.weights.w_flatness * c_flatness +
            self.weights.w_height * c_height +
            self.weights.w_obstacle * c_obs
        )

        return float(np.clip(total, 0.0, 1.0))

    def compute_grid_cost(
        self,
        bev_traversability: np.ndarray,
        unknown_mask: np.ndarray,
        slope_grid: Optional[np.ndarray] = None,
        flatness_grid: Optional[np.ndarray] = None,
        height_var_grid: Optional[np.ndarray] = None,
        obstacle_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Vectorized cost calculation over an entire 2D BEV grid."""
        shape = bev_traversability.shape
        cost = np.zeros(shape, dtype=np.float32)

        # Learned cost (1.0 - drivability)
        c_learned = 1.0 - np.clip(bev_traversability, 0.0, 1.0)

        # Geometric slope cost
        if slope_grid is not None:
            c_slope = np.clip(slope_grid / self.weights.max_safe_slope_deg, 0.0, 1.0)
            lethal_slope = slope_grid >= self.weights.critical_slope_deg
        else:
            c_slope = np.zeros(shape, dtype=np.float32)
            lethal_slope = np.zeros(shape, dtype=bool)

        # Flatness cost
        if flatness_grid is not None:
            c_flatness = np.clip(flatness_grid / 0.10, 0.0, 1.0)
        else:
            c_flatness = np.zeros(shape, dtype=np.float32)

        # Height variation cost
        if height_var_grid is not None:
            c_height = np.clip(height_var_grid / self.weights.max_step_height_m, 0.0, 1.0)
            lethal_height = height_var_grid >= self.weights.critical_step_height_m
        else:
            c_height = np.zeros(shape, dtype=np.float32)
            lethal_height = np.zeros(shape, dtype=bool)

        # Obstacle cost
        if obstacle_grid is not None:
            c_obs = np.clip(obstacle_grid, 0.0, 1.0)
            lethal_obs = obstacle_grid >= 0.8
        else:
            c_obs = np.zeros(shape, dtype=np.float32)
            lethal_obs = np.zeros(shape, dtype=bool)

        # Linear weighted combination
        cost = (
            self.weights.w_learned * c_learned +
            self.weights.w_slope * c_slope +
            self.weights.w_flatness * c_flatness +
            self.weights.w_height * c_height +
            self.weights.w_obstacle * c_obs
        )

        # Assign unknown cells penalty
        cost[unknown_mask] = self.weights.unknown_cost

        # Hard lethal cutoffs
        cost[lethal_slope | lethal_height | lethal_obs] = 1.0

        return np.clip(cost, 0.0, 1.0).astype(np.float32)
