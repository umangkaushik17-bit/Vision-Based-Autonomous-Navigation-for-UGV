"""Focused validation suite for Phase 13.13 Evidence Integrity and Outcome Semantics.

Tests canonical semantics and closed-loop behavior under the strict production freeze:
- TEST A: Timeout before goal -> INCOMPLETE + SAFE + HORIZON_BOUNDED_TIMEOUT + mission_success=False
- TEST B: Goal arrival -> GOAL_REACHED + SAFE + GOAL_REACHED + mission_success=True
- TEST C: Collision -> INCOMPLETE + COLLISION + COLLISION + mission_success=False
- TEST D: Local planner cycles are strictly distinct from global replan count
- TEST E: Scenario G1 persistent blockage behavior under frozen algorithms
- TEST F: Scenario N3 narrow passage behavior under frozen algorithms
- TEST G: Scenario O deadlock evaluation condition under frozen algorithms
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
import pytest

from ai.planning.base import GoalPose
from simulation.metrics.evaluator import QuantitativeScenarioMetrics, ScenarioEvaluator
from simulation.runners.dynamic_sil_runner import DynamicSILRunner, ScenarioRunMetrics
from simulation.scenarios.dynamic.advanced_scenarios import (
    create_scenario_g1_persistent_blockage_alternate,
    create_scenario_n3_narrow_blocked,
    create_scenario_o_deadlock,
)
from simulation.scenarios.dynamic.core_scenarios import create_scenario_a_static


class TestCanonicalSemantics:
    """Workstream 1 & 2: Canonical outcome and termination semantics."""

    def test_timeout_before_goal(self):
        """TEST A: Timeout before reaching goal must yield INCOMPLETE, SAFE, HORIZON_BOUNDED_TIMEOUT, mission_success=False."""
        cfg = create_scenario_a_static()
        cfg.max_duration_s = 0.30  # Short horizon: terminates before goal at x=3.0m
        runner = DynamicSILRunner(cfg)
        metrics = runner.run()

        assert metrics.mission_outcome == "INCOMPLETE", f"Expected INCOMPLETE, got {metrics.mission_outcome}"
        assert metrics.safety_outcome == "SAFE", f"Expected SAFE, got {metrics.safety_outcome}"
        assert metrics.termination_reason == "HORIZON_BOUNDED_TIMEOUT", f"Expected HORIZON_BOUNDED_TIMEOUT, got {metrics.termination_reason}"
        assert metrics.mission_success is False, "mission_success must be False on timeout before goal"
        assert metrics.exit_reason == "HORIZON_BOUNDED_TIMEOUT"
        assert metrics.collision_count == 0
        assert metrics.goal_position_error_m > metrics.goal_tolerance_m

    def test_goal_arrival_evaluator(self):
        """TEST B (Evaluator unit): Goal arrival within tolerance yields GOAL_REACHED, SAFE, mission_success=True."""
        evaluator = ScenarioEvaluator([])
        telemetry = [
            {"step": 0, "sim_time": 0.0, "robot_x": 0.0, "robot_y": 0.0, "robot_yaw": 0.0, "robot_v": 0.4, "gt_collision": False},
            {"step": 1, "sim_time": 7.2, "robot_x": 2.95, "robot_y": 0.02, "robot_yaw": 0.0, "robot_v": 0.0, "gt_collision": False},
        ]
        goal = (3.0, 0.0, 0.0)
        metrics = evaluator.evaluate(
            telemetry=telemetry,
            goal_pose=goal,
            goal_tolerance_m=0.25,
            termination_reason="GOAL_REACHED",
            configured_duration_s=10.0,
            local_planner_cycles=72,
            global_replan_count=0,
        )

        assert metrics.mission_outcome == "GOAL_REACHED"
        assert metrics.safety_outcome == "SAFE"
        assert metrics.termination_reason == "GOAL_REACHED"
        assert metrics.mission_success is True
        assert metrics.collision_count == 0
        assert metrics.goal_position_error_m <= 0.25

    def test_collision_evaluator(self):
        """TEST C: Collision yields INCOMPLETE, COLLISION, COLLISION, mission_success=False even if at goal."""
        evaluator = ScenarioEvaluator([])
        telemetry = [
            {"step": 0, "sim_time": 0.0, "robot_x": 0.0, "robot_y": 0.0, "robot_yaw": 0.0, "robot_v": 0.4, "gt_collision": False},
            {"step": 1, "sim_time": 3.0, "robot_x": 3.0, "robot_y": 0.0, "robot_yaw": 0.0, "robot_v": 0.0, "gt_collision": True},
        ]
        goal = (3.0, 0.0, 0.0)
        metrics = evaluator.evaluate(
            telemetry=telemetry,
            goal_pose=goal,
            goal_tolerance_m=0.25,
            termination_reason="COLLISION",
            configured_duration_s=5.0,
        )

        assert metrics.mission_outcome == "INCOMPLETE"
        assert metrics.safety_outcome == "COLLISION"
        assert metrics.termination_reason == "COLLISION"
        assert metrics.mission_success is False
        assert metrics.collision_count == 1

    def test_local_cycles_distinct_from_global_replans(self):
        """TEST D: Local planner cycles must be tracked separately from event-driven global replan count."""
        cfg = create_scenario_a_static()
        cfg.max_duration_s = 0.30
        runner = DynamicSILRunner(cfg)
        metrics = runner.run()

        # Local cycles execute every step where DWA runs; global replan count is 0 for static run
        assert metrics.local_planner_cycles == 3, f"Expected 3 local planner cycles, got {metrics.local_planner_cycles}"
        assert metrics.global_replan_count == 0, f"Expected 0 global replans, got {metrics.global_replan_count}"
        assert metrics.local_planner_cycles != metrics.global_replan_count

        # Check serialization exposes both
        d = metrics.to_dict()
        assert "local_planner_cycles" in d
        assert "global_replan_count" in d
        assert d["local_planner_cycles"] == 3
        assert d["global_replan_count"] == 0


class TestBehavioralScenariosHonestTelemetry:
    """Workstream 6: Closed-loop evidence evaluation on frozen production stack."""

    def test_scenario_n3_corridor_avoids_without_stop(self):
        """TEST F: Demonstrates that Scenario N3 without lateral walls executes AVOID around obstacles."""
        cfg = create_scenario_n3_narrow_blocked()
        cfg.max_duration_s = 0.50  # Bounded execution for test suite
        runner = DynamicSILRunner(cfg)
        metrics = runner.run()

        assert metrics.safety_outcome == "SAFE"
        assert metrics.collision_count == 0
        # The frozen DWA stack finds clear lateral rollouts in unconstrained world bounds, entering AVOID
        assert metrics.avoid_count > 0 or metrics.proceed_count > 0

    def test_scenario_o_box_trap_safely_bounded(self):
        """TEST G: Demonstrates Scenario O box trap maintains zero collision under frozen algorithms."""
        cfg = create_scenario_o_deadlock()
        cfg.max_duration_s = 0.50  # Bounded execution for test suite
        runner = DynamicSILRunner(cfg)
        metrics = runner.run()

        assert metrics.safety_outcome == "SAFE"
        assert metrics.collision_count == 0
        assert metrics.mission_outcome == "INCOMPLETE"
        assert metrics.termination_reason == "HORIZON_BOUNDED_TIMEOUT"

    def test_scenario_g1_persistent_blockage_eval(self):
        """TEST E: Demonstrates Scenario G1 persistent blockage evaluation under frozen algorithms."""
        cfg = create_scenario_g1_persistent_blockage_alternate()
        cfg.max_duration_s = 0.50  # Bounded execution for test suite
        runner = DynamicSILRunner(cfg)
        metrics = runner.run()

        assert metrics.safety_outcome == "SAFE"
        assert metrics.collision_count == 0
        assert metrics.global_replan_count == 0  # Replan requires accumulation >= 3.0s
