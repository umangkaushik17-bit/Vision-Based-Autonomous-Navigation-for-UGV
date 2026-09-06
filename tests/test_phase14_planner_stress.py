"""Group 4: Planner stress, candidate scarcity, and supervisory safety clamping tests."""

from __future__ import annotations

import math
import pytest

from ai.planning.base import PlannerStatus, RobotState
from simulation.runners.phase14_sil_runner import Phase14SILRunner
from simulation.scenarios.dynamic.phase14.data_model import (
    Phase14FaultSpec,
    Phase14FaultType,
    Phase14ScenarioConfig,
)
from simulation.scenarios.dynamic.phase14.scenario_builders import (
    create_scenario_m14_09_planner_saturation,
    create_scenario_m14_10_zero_valid_trajectories,
    create_scenario_m14_15_invalid_planner_output,
)


def test_m14_09_planner_saturation_candidate_scarcity():
    """Assert F: Planner saturation rejects conflicting candidates without unsafe command."""
    cfg = create_scenario_m14_09_planner_saturation()
    cfg.max_duration_s = 0.30  # 3 steps bounded
    runner = Phase14SILRunner(cfg)
    metrics = runner.run()

    assert metrics.safety_outcome == "SAFE"
    assert metrics.collision_count == 0
    # Saturation should cause candidate rejections
    assert metrics.dwa_avg_rejected_candidates > 0
    # Commands must remain within physical envelope
    for step in runner.telemetry:
        assert not math.isnan(step["cmd_v"])
        assert not math.isnan(step["cmd_w"])
        assert 0.0 <= step["cmd_v"] <= 0.60


def test_m14_10_zero_valid_trajectories_safety_fallback():
    """M14-10 / Assert F: When planner produces zero valid candidates, vehicle safely clamps to 0."""
    cfg = create_scenario_m14_10_zero_valid_trajectories()
    cfg.max_duration_s = 0.50  # Fault active t in [0.3, 0.7]
    runner = Phase14SILRunner(cfg)
    metrics = runner.run()

    assert metrics.safety_outcome == "SAFE"
    assert metrics.collision_count == 0
    # Step at t=0.4 must have zero command due to zero valid candidates
    fault_steps = [s for s in runner.telemetry if 0.30 <= s["sim_time"] <= 0.70]
    assert len(fault_steps) > 0
    for s in fault_steps:
        assert s["cmd_v"] == 0.0, f"Expected cmd_v == 0.0 during zero-valid fault, got {s['cmd_v']}"
        assert s["cmd_w"] == 0.0, f"Expected cmd_w == 0.0 during zero-valid fault, got {s['cmd_w']}"


def test_m14_15_invalid_planner_output_supervisory_clamping():
    """M14-15 / Assert G: Malformed or NaN planner output is strictly intercepted and clamped to 0."""
    cfg = create_scenario_m14_15_invalid_planner_output()
    cfg.max_duration_s = 0.50  # Fault active t in [0.3, 0.6]
    runner = Phase14SILRunner(cfg)
    metrics = runner.run()

    assert metrics.safety_outcome == "SAFE"
    assert metrics.collision_count == 0
    # Verify that no NaN command ever reached telemetry or robot state
    for s in runner.telemetry:
        assert not math.isnan(s["cmd_v"]), "NaN cmd_v propagated!"
        assert not math.isnan(s["cmd_w"]), "NaN cmd_w propagated!"
        assert not math.isnan(s["robot_x"]), "NaN robot_x propagated!"
        assert not math.isnan(s["robot_y"]), "NaN robot_y propagated!"
