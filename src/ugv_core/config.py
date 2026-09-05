"""Configuration loader and schema validator for UGV navigation system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

class CameraConfig(BaseModel):
    width: int = Field(default=640, description="Image capture width in pixels")
    height: int = Field(default=480, description="Image capture height in pixels")
    fps: int = Field(default=30, description="Target frame rate")
    fov_degrees: float = Field(default=75.0, description="Horizontal field of view")
    baseline_m: float = Field(default=0.12, description="Stereo baseline in meters if stereo")

class RobotSpecsConfig(BaseModel):
    max_linear_speed_mps: float = Field(default=1.2, description="Max forward velocity")
    min_linear_speed_mps: float = Field(default=-0.4, description="Max reverse velocity")
    max_angular_speed_radps: float = Field(default=1.5, description="Max turning rate")
    robot_radius_m: float = Field(default=0.35, description="Collision envelope radius")
    safety_margin_m: float = Field(default=0.15, description="Extra buffer beyond robot radius")

class TraversabilityConfig(BaseModel):
    safe_threshold: float = Field(default=0.35, description="Costs below this are considered safe")
    unsafe_threshold: float = Field(default=0.70, description="Costs above this are considered lethal")
    model_name: str = Field(default="mobilenet_v3_small", description="Backbone model name")
    use_quantized_onnx: bool = Field(default=True, description="Use ONNX runtime if available")
    device: str = Field(default="auto", description="'auto', 'cpu', or 'cuda'")

class CostmapConfig(BaseModel):
    resolution_m: float = Field(default=0.05, description="Meters per grid cell")
    width_m: float = Field(default=10.0, description="Width of local costmap in meters")
    height_m: float = Field(default=10.0, description="Length of local costmap in meters")
    inflation_radius_m: float = Field(default=0.5, description="Obstacle inflation radius in meters")
    decay_rate: float = Field(default=0.05, description="Observation decay factor per second")

class PlannerConfig(BaseModel):
    algorithm: str = Field(default="hybrid_astar", description="Path planning algorithm ('astar', 'hybrid_astar', 'dwa')")
    lookahead_dist_m: float = Field(default=1.5, description="Goal pursuit lookahead distance")
    replan_frequency_hz: float = Field(default=5.0, description="Frequency of dynamic replanning")
    goal_tolerance_m: float = Field(default=0.25, description="Tolerance radius to declare waypoint reached")

class RiskConfig(BaseModel):
    low_risk_speed_factor: float = Field(default=1.0, description="Speed multiplier under low risk")
    med_risk_speed_factor: float = Field(default=0.5, description="Speed multiplier under medium risk")
    high_risk_replan: bool = Field(default=True, description="Trigger immediate path replan under high risk")
    critical_risk_emergency_stop: bool = Field(default=True, description="Trigger immediate e-stop under critical risk")

class SystemConfig(BaseModel):
    project_name: str = "SIH26126-UGV"
    camera: CameraConfig = Field(default_factory=CameraConfig)
    robot: RobotSpecsConfig = Field(default_factory=RobotSpecsConfig)
    traversability: TraversabilityConfig = Field(default_factory=TraversabilityConfig)
    costmap: CostmapConfig = Field(default_factory=CostmapConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load configuration from YAML file or return robust defaults.
    
    Args:
        config_path: Optional path to YAML configuration file. If not provided,
                     looks for 'configs/system_config.yaml' relative to workspace root.
                     
    Returns:
        SystemConfig instance.
    """
    if config_path is None:
        # Default discovery
        current_dir = Path(__file__).resolve().parent
        # Root is src/ugv_core -> src -> root
        candidate = current_dir.parent.parent / "configs" / "system_config.yaml"
        if candidate.exists():
            config_path = str(candidate)
            
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return SystemConfig(**data)
            
    return SystemConfig()
