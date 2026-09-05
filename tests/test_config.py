"""Unit tests for configuration loader and validation."""

import pytest
from ugv_core.config import SystemConfig, load_config

def test_default_config_loading():
    cfg = load_config()
    assert cfg.project_name == "SIH26126-UGV"
    assert cfg.camera.width == 640
    assert cfg.camera.height == 480
    assert cfg.robot.max_linear_speed_mps > 0
    assert cfg.traversability.safe_threshold < cfg.traversability.unsafe_threshold
    assert cfg.costmap.resolution_m > 0
    assert cfg.planner.algorithm in ["astar", "hybrid_astar", "dwa"]
    assert cfg.risk.critical_risk_emergency_stop is True

def test_custom_config_values():
    cfg = SystemConfig(
        robot={"max_linear_speed_mps": 2.0},
        planner={"algorithm": "dwa"}
    )
    assert cfg.robot.max_linear_speed_mps == 2.0
    assert cfg.planner.algorithm == "dwa"
