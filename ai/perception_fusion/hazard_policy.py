"""Hazard Policy Evaluator.

Loads and evaluates risk levels, cost multipliers, and fusion parameters
defined in configs/hazard_policy.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import yaml

from ai.obstacles.base import HazardType, RiskLevel

DEFAULT_POLICY_PATH = Path("configs/hazard_policy.yaml")


class HazardPolicy:
    """Evaluates risk and navigation costs for detected hazards based on vehicle policy."""

    def __init__(self, policy_path: Union[str, Path] = DEFAULT_POLICY_PATH) -> None:
        self.policy_path = Path(policy_path)
        self.config = self._load_config()

        self.risk_tiers = self.config.get("risk_tiers", {})
        self.rules = self.config.get("hazard_classification_rules", {})
        self.fusion_policy = self.config.get("fusion_policy", {})

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration or use fallback defaults."""
        if self.policy_path.exists():
            with open(self.policy_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {
            "risk_tiers": {
                "CRITICAL": {"cost": 1.0, "speed_factor": 0.0},
                "HIGH": {"cost": 0.85, "speed_factor": 0.25},
                "MEDIUM": {"cost": 0.50, "speed_factor": 0.60},
                "LOW": {"cost": 0.15, "speed_factor": 0.90},
                "UNKNOWN": {"cost": 0.50, "speed_factor": 0.50}
            },
            "hazard_classification_rules": {},
            "fusion_policy": {
                "weight_model_a": 0.50,
                "weight_model_b": 0.50,
                "disagreement_threshold": 0.40,
                "treat_disagreement_as_unknown": True,
                "obstacle_override_enabled": True,
                "min_obstacle_confidence": 0.40
            }
        }

    def evaluate_class(self, class_name: str) -> Tuple[HazardType, RiskLevel]:
        """Map raw detection class name to standardized HazardType and RiskLevel."""
        clean_name = class_name.strip().lower()
        if clean_name in self.rules:
            rule = self.rules[clean_name]
            h_type = HazardType(rule.get("hazard_type", "UNKNOWN_HAZARD"))
            r_level = RiskLevel(rule.get("risk_level", "HIGH"))
            return h_type, r_level

        # Fallback keyword matching
        if any(w in clean_name for w in ["water", "creek", "river", "puddle"]):
            return HazardType.WATER, RiskLevel.CRITICAL
        if any(w in clean_name for w in ["ditch", "pothole", "trench", "hole"]):
            return HazardType.DITCH, RiskLevel.CRITICAL
        if any(w in clean_name for w in ["person", "human", "pedestrian"]):
            return HazardType.PERSON, RiskLevel.CRITICAL
        if any(w in clean_name for w in ["car", "truck", "bus", "vehicle"]):
            return HazardType.VEHICLE, RiskLevel.CRITICAL
        if any(w in clean_name for w in ["rock", "stone", "boulder"]):
            return HazardType.ROCK, RiskLevel.HIGH
        if any(w in clean_name for w in ["tree", "trunk", "wood"]):
            return HazardType.TREE, RiskLevel.HIGH
        if any(w in clean_name for w in ["grass", "bush", "vegetation"]):
            return HazardType.VEGETATION, RiskLevel.LOW

        return HazardType.UNKNOWN_HAZARD, RiskLevel.HIGH

    def get_risk_cost(self, risk_level: RiskLevel) -> float:
        """Get baseline traversal cost [0.0, 1.0] for a given risk level."""
        tier = self.risk_tiers.get(risk_level.value, {})
        return float(tier.get("cost", 0.50))

    def get_speed_factor(self, risk_level: RiskLevel) -> float:
        """Get maximum speed scaling factor [0.0, 1.0] when navigating near risk."""
        tier = self.risk_tiers.get(risk_level.value, {})
        return float(tier.get("speed_factor", 0.50))
