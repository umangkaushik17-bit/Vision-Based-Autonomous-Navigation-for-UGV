"""Multi-layer costmap fusion engine enforcing weighted physical costs and hard safety overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import yaml

from ai.costmap.base import CostmapMetadata, UnknownPolicyMode
from ai.costmap.bev_costmap import BEVTerrainCostmap

DEFAULT_CONFIG_PATH = Path("configs/costmap.yaml")


class BEVCostmapFuser:
    """Fuses multi-factor terrain geometry and perception layers into a continuous costmap."""

    def __init__(
        self,
        config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
        w_traversability: float = 0.35,
        w_hazard: float = 0.25,
        w_slope: float = 0.15,
        w_roughness: float = 0.10,
        w_height: float = 0.10,
        w_unknown: float = 0.05,
        unknown_mode: Optional[UnknownPolicyMode] = None,
        critical_slope_deg: float = 28.0,
        critical_step_height_m: float = 0.20,
        max_safe_slope_deg: float = 20.0,
        max_safe_step_height_m: float = 0.12
    ) -> None:
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            weights = cfg.get("cost_weights", {})
            self.w_trav = float(weights.get("w_traversability", w_traversability))
            self.w_haz = float(weights.get("w_hazard", w_hazard))
            self.w_slope = float(weights.get("w_slope", w_slope))
            self.w_rough = float(weights.get("w_roughness", w_roughness))
            self.w_height = float(weights.get("w_height", w_height))
            self.w_unk = float(weights.get("w_unknown", w_unknown))

            unk_cfg = cfg.get("unknown_policy", {})
            if unknown_mode is not None:
                self.unknown_mode = unknown_mode
            else:
                mode_str = unk_cfg.get("mode", "CONSERVATIVE").upper()
                self.unknown_mode = UnknownPolicyMode(mode_str)
            self.conservative_cost = float(unk_cfg.get("conservative_cost", 0.65))
            self.moderate_cost = float(unk_cfg.get("moderate_cost", 0.50))
            self.exploratory_cost = float(unk_cfg.get("exploratory_cost", 0.35))

            overrides = cfg.get("hard_safety_overrides", {})
            self.critical_slope_deg = float(overrides.get("critical_slope_deg", critical_slope_deg))
            self.critical_step_height_m = float(overrides.get("critical_step_height_m", critical_step_height_m))
            self.max_safe_slope_deg = float(overrides.get("max_safe_slope_deg", max_safe_slope_deg))
            self.max_safe_step_height_m = float(overrides.get("max_safe_step_height_m", max_safe_step_height_m))
        else:
            self.w_trav = w_traversability
            self.w_haz = w_hazard
            self.w_slope = w_slope
            self.w_rough = w_roughness
            self.w_height = w_height
            self.w_unk = w_unknown
            self.unknown_mode = unknown_mode
            self.conservative_cost = 0.65
            self.moderate_cost = 0.50
            self.exploratory_cost = 0.35
            self.critical_slope_deg = critical_slope_deg
            self.critical_step_height_m = critical_step_height_m
            self.max_safe_slope_deg = max_safe_slope_deg
            self.max_safe_step_height_m = max_safe_step_height_m

    def get_unknown_cost(self) -> float:
        """Return base cost assigned to unobserved cells based on active unknown policy."""
        if self.unknown_mode == UnknownPolicyMode.CONSERVATIVE:
            return self.conservative_cost
        elif self.unknown_mode == UnknownPolicyMode.MODERATE:
            return self.moderate_cost
        return self.exploratory_cost

    def fuse_layers(
        self,
        layers: Dict[str, np.ndarray],
        metadata: CostmapMetadata
    ) -> BEVTerrainCostmap:
        """Combine 8 raster layers into a canonical BEVTerrainCostmap with safety overrides."""
        trav = layers["traversability"]
        hazard = layers["hazard"]
        obstacle = layers["obstacle"]
        unknown = layers["unknown"]
        conf = layers["confidence"]
        slope = layers["slope_deg"]
        roughness = layers["roughness"]
        height_var = layers["height_var"]
        obs_count = layers["observation_count"]

        # 1. Physical Factor Normalization
        c_trav = 1.0 - np.clip(trav, 0.0, 1.0)
        c_hazard = np.clip(hazard, 0.0, 1.0)
        c_slope = np.clip(slope / max(self.max_safe_slope_deg, 1.0), 0.0, 1.0)
        c_roughness = np.clip(roughness, 0.0, 1.0)
        c_height = np.clip(height_var / max(self.max_safe_step_height_m, 0.01), 0.0, 1.0)
        c_unknown = np.full_like(trav, self.get_unknown_cost())

        # 2. Linear Weighted Base Cost
        raw_cost = (
            self.w_trav * c_trav
            + self.w_haz * c_hazard
            + self.w_slope * c_slope
            + self.w_rough * c_roughness
            + self.w_height * c_height
            + self.w_unk * c_unknown
        )
        cost = np.clip(raw_cost, 0.0, 1.0)

        # 3. Apply Unknown Policy
        cost[unknown] = self.get_unknown_cost()

        # 4. Enforce Hard Safety Overrides (Non-negotiable Architectural Rule)
        # Rule A: Explicit Critical/High Hazards (Water, Ditch, Boulder, Vehicle, Person)
        cost[obstacle] = 1.00
        cost[hazard >= 0.85] = 1.00

        # Rule B: Critical Rollover Slope Angle (>= 28 degrees)
        cost[slope >= self.critical_slope_deg] = 1.00

        # Rule C: Critical High-Centering Step Height (>= 20 cm)
        cost[height_var >= self.critical_step_height_m] = 1.00

        # Update final obstacle mask to include geometric lethality
        final_obstacle = obstacle | (slope >= self.critical_slope_deg) | (height_var >= self.critical_step_height_m)

        return BEVTerrainCostmap(
            cost=cost.astype(np.float32),
            traversability=trav.astype(np.float32),
            obstacle=final_obstacle,
            hazard=hazard.astype(np.float32),
            unknown=unknown,
            confidence=conf.astype(np.float32),
            slope_deg=slope.astype(np.float32),
            roughness=roughness.astype(np.float32),
            height_var=height_var.astype(np.float32),
            observation_count=obs_count.astype(np.int32),
            metadata=metadata
        )
