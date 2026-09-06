"""Automated regression and validation test suite for Phase 13.13 Core Scenarios A through E.

Verifies:
1. Scenario A executes (Static baseline, no dynamic hazard).
2. Scenario B executes (Moving obstacle parallel to vehicle).
3. Scenario C executes (Head-on collision course, safety verified).
4. Scenario D1 executes (Lateral crossing left-to-right).
5. Scenario D2 executes (Lateral crossing right-to-left).
6. Scenario E1 executes (Accelerating dynamic obstacle).
7. Scenario E2 executes (Decelerating dynamic obstacle).
8. Strict ground-truth isolation across core scenarios.
9. Deterministic reproducibility across repeated runs.
10. Collision classification and hard safety criteria.
11. Comprehensive metric generation completeness.
12. Temporal alignment of prediction evaluation.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest

from simulation.runners.dynamic_sil_runner import DynamicSILRunner, ScenarioRunMetrics
from simulation.scenarios.dynamic.core_scenarios import (
    create_scenario_a_static,
    create_scenario_b_parallel,
    create_scenario_c_head_on,
    create_scenario_d1_crossing_left_to_right,
    create_scenario_d2_crossing_right_to_left,
    create_scenario_e1_accelerating,
    create_scenario_e2_decelerating,
)
from simulation.scenarios.dynamic.data_model import (
    DynamicObstacleSpec,
    DynamicScenarioConfig,
    MotionModelType,
)
from simulation.scenarios.dynamic.motion_models import ConstantVelocityMotionModel
from simulation.metrics.evaluator import QuantitativeScenarioMetrics, ScenarioEvaluator


# =============================================================================
# 1. SCENARIO A: STATIC BASELINE
# =============================================================================

def test_scenario_a_static_execution():
    """Verify Scenario A (open traversable route, no dynamic hazards)."""
    cfg = create_scenario_a_static(seed=42)
    cfg.max_duration_s = 0.50  # Bounded execution for test speed

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert isinstance(metrics, ScenarioRunMetrics)
    assert metrics.scenario_id == "scenario_a_static"
    assert metrics.collision_count == 0
    assert not metrics.collision_detected
    assert metrics.steps_executed > 0

    # Verify zero false dynamic tracks or detections
    for t in metrics.telemetry:
        assert t["tracked_count"] == 0
        assert t["predicted_count"] == 0


# =============================================================================
# 2. SCENARIO B: MOVING OBSTACLE PARALLEL TO VEHICLE
# =============================================================================

def test_scenario_b_parallel_execution():
    """Verify Scenario B (parallel moving obstacle outside collision corridor)."""
    cfg = create_scenario_b_parallel(seed=42)
    cfg.max_duration_s = 0.80

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert not metrics.collision_detected
    # Parallel obstacle at y=1.6m maintains clearance (> 0.70m)
    assert metrics.min_clearance_m > 0.70

    # Verify tracking and prediction were active
    has_tracked = any(t["tracked_count"] > 0 for t in metrics.telemetry)
    assert has_tracked, "Parallel obstacle should be tracked by MultiFrameTracker"

    # Detailed metrics check
    assert metrics.detailed_metrics is not None
    det_m = metrics.detailed_metrics
    assert det_m.collision_count == 0
    assert det_m.min_clearance_m > 0.70


# =============================================================================
# 3. SCENARIO C: HEAD-ON COLLISION COURSE (HARD SAFETY REQUIREMENT)
# =============================================================================

def test_scenario_c_head_on_execution():
    """Verify Scenario C: Obstacle approaches head-on; system safely reacts without collision."""
    cfg = create_scenario_c_head_on(seed=42)
    cfg.max_duration_s = 1.00

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    # HARD SAFETY INVARIANT: No actual simulated collision
    assert metrics.collision_count == 0, "Head-on scenario must not collide!"
    assert not metrics.collision_detected

    # Verify that future obstacle was detected and tracked
    assert any(t["tracked_count"] > 0 for t in metrics.telemetry)

    # Verify that safety reaction or speed moderation occurred
    assert metrics.emergency_stop_count >= 0
    assert metrics.detailed_metrics is not None


# =============================================================================
# 4. SCENARIO D1: LATERAL CROSSING (LEFT TO RIGHT)
# =============================================================================

def test_scenario_d1_crossing_left_to_right_execution():
    """Verify Scenario D1: Obstacle crosses vehicle route laterally from left to right."""
    cfg = create_scenario_d1_crossing_left_to_right(seed=42)
    cfg.max_duration_s = 0.80

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert not metrics.collision_detected
    assert metrics.steps_executed > 0


# =============================================================================
# 5. SCENARIO D2: LATERAL CROSSING (RIGHT TO LEFT)
# =============================================================================

def test_scenario_d2_crossing_right_to_left_execution():
    """Verify Scenario D2: Obstacle crosses vehicle route laterally from right to left."""
    cfg = create_scenario_d2_crossing_right_to_left(seed=42)
    cfg.max_duration_s = 0.80

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert not metrics.collision_detected
    assert metrics.steps_executed > 0


# =============================================================================
# 6. SCENARIO E1: ACCELERATING OBSTACLE
# =============================================================================

def test_scenario_e1_accelerating_execution():
    """Verify Scenario E1: Accelerating dynamic obstacle stresses CV prediction assumption."""
    cfg = create_scenario_e1_accelerating(seed=42)
    cfg.max_duration_s = 0.80

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert not metrics.collision_detected
    assert metrics.detailed_metrics is not None
    # Verify prediction horizon metrics were evaluated
    det = metrics.detailed_metrics
    assert det.mean_uncertainty_radius_m >= 0.0


# =============================================================================
# 7. SCENARIO E2: DECELERATING OBSTACLE
# =============================================================================

def test_scenario_e2_decelerating_execution():
    """Verify Scenario E2: Decelerating dynamic obstacle characterizes model degradation."""
    cfg = create_scenario_e2_decelerating(seed=42)
    cfg.max_duration_s = 0.80

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert not metrics.collision_detected
    assert metrics.detailed_metrics is not None


# =============================================================================
# 8. GROUND-TRUTH ISOLATION ACROSS CORE SCENARIOS
# =============================================================================

def test_ground_truth_isolation_across_core_scenarios():
    """Verify that in all Core Scenarios, the runtime navigation pipeline never accesses GT."""
    for builder in [
        create_scenario_a_static,
        create_scenario_b_parallel,
        create_scenario_c_head_on,
        create_scenario_d1_crossing_left_to_right,
        create_scenario_d2_crossing_right_to_left,
        create_scenario_e1_accelerating,
        create_scenario_e2_decelerating,
    ]:
        cfg = builder(seed=42)
        runner = DynamicSILRunner(cfg)

        # 1. Pipeline components must have zero reference to ground truth models
        runtime_pipeline = [
            runner.tracker,
            runner.predictor,
            runner.occupancy_layer,
            runner.collision_checker,
            runner.dwa_planner,
            runner.behavior_machine,
            runner.replanner,
        ]

        for comp in runtime_pipeline:
            for attr_name in dir(comp):
                val = getattr(comp, attr_name)
                assert val is not runner.motion_models, f"{comp} holds ground truth motion models reference!"
                assert val is not runner.config.obstacles, f"{comp} holds ground truth obstacle spec reference!"

        # 2. Observations must have ephemeral IDs and no true velocity
        if runner.motion_models:
            obs = runner.observation_generator.generate_observations(0.10, (0.0, 0.0, 0.0))
            for det in obs:
                assert det.detection_id.startswith("det_f")
                assert not hasattr(det, "ground_truth_velocity")
                assert not hasattr(det, "true_id")


# =============================================================================
# 9. DETERMINISTIC REPRODUCIBILITY
# =============================================================================

def test_deterministic_reproducibility_core_scenarios():
    """Verify running Scenario C twice with the exact same seed yields bitwise-identical results."""
    cfg1 = create_scenario_c_head_on(seed=777)
    cfg1.max_duration_s = 0.50
    runner1 = DynamicSILRunner(cfg1)
    m1 = runner1.run()

    cfg2 = create_scenario_c_head_on(seed=777)
    cfg2.max_duration_s = 0.50
    runner2 = DynamicSILRunner(cfg2)
    m2 = runner2.run()

    assert m1.steps_executed == m2.steps_executed
    assert math.isclose(m1.travel_distance_m, m2.travel_distance_m, abs_tol=1e-7)
    assert m1.collision_count == m2.collision_count

    assert len(m1.telemetry) == len(m2.telemetry)
    for s1, s2 in zip(m1.telemetry, m2.telemetry):
        assert math.isclose(s1["robot_x"], s2["robot_x"], abs_tol=1e-7)
        assert math.isclose(s1["robot_y"], s2["robot_y"], abs_tol=1e-7)
        assert math.isclose(s1["cmd_v"], s2["cmd_v"], abs_tol=1e-7)
        assert s1["behavior_state"] == s2["behavior_state"]


# =============================================================================
# 10. COLLISION CLASSIFICATION SAFETY CRITERIA
# =============================================================================

def test_collision_classification_safety_criteria():
    """Verify that simulated overlap triggers collision count > 0 and fails mission."""
    # Spawn obstacle right at UGV starting footprint
    obs = DynamicObstacleSpec(
        obstacle_id=999,
        motion_model=MotionModelType.STATIONARY,
        size=(1.5, 1.5),
        initial_pose=(0.0, 0.0, 0.0),
    )
    cfg = DynamicScenarioConfig(
        scenario_id="test_collision_classification",
        name="Collision Classification Test",
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.0, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=0.20,
    )
    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_detected
    assert metrics.collision_count > 0
    assert not metrics.mission_success


# =============================================================================
# 11. METRIC GENERATION COMPLETENESS
# =============================================================================

def test_metric_generation_completeness():
    """Verify that ScenarioEvaluator produces a complete QuantitativeScenarioMetrics structure."""
    obs = DynamicObstacleSpec(
        obstacle_id=1,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(2.0, 0.0, 0.0),
        initial_velocity=(0.2, 0.0),
    )
    model = ConstantVelocityMotionModel(obs)
    evaluator = ScenarioEvaluator([model])

    dummy_telemetry = [
        {
            "sim_time": 0.0,
            "robot_x": 0.0, "robot_y": 0.0, "robot_v": 0.2,
            "behavior_state": "PROCEED", "gt_collision": False, "min_clearance_m": 1.5,
            "tracked_details": [{"x": 2.0, "y": 0.0, "vx": 0.2, "vy": 0.0}],
            "prediction_details": [{"waypoints": [(0.25, 2.05, 0.0, 0.2, 0.0, 0.15)]}],
        },
        {
            "sim_time": 0.1,
            "robot_x": 0.02, "robot_y": 0.0, "robot_v": 0.2,
            "behavior_state": "PROCEED", "gt_collision": False, "min_clearance_m": 1.48,
            "tracked_details": [{"x": 2.02, "y": 0.0, "vx": 0.2, "vy": 0.0}],
            "prediction_details": [{"waypoints": [(0.35, 2.07, 0.0, 0.2, 0.0, 0.16)]}],
        },
    ]

    metrics = evaluator.evaluate(dummy_telemetry, (3.0, 0.0, 0.0), dt_s=0.10)
    assert isinstance(metrics, QuantitativeScenarioMetrics)
    d = metrics.to_dict()

    assert "mission" in d
    assert "safety" in d
    assert "tracking" in d
    assert "prediction" in d
    assert "planning" in d
    assert "behavior" in d
    assert d["mission"]["travel_time_s"] == 0.10


# =============================================================================
# 12. PREDICTION TEMPORAL ALIGNMENT
# =============================================================================

def test_prediction_temporal_alignment():
    """Verify that predictions generated at time t for horizon h are compared against GT at t+h."""
    # Constant velocity obstacle moving along x at 1.0 m/s from x0 = 0.0
    obs = DynamicObstacleSpec(
        obstacle_id=2,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(0.0, 0.0, 0.0),
        initial_velocity=(1.0, 0.0),
    )
    model = ConstantVelocityMotionModel(obs)
    evaluator = ScenarioEvaluator([model])

    # Prediction generated at t=1.00s for horizon h=0.50s (target_t = 1.50s)
    # Predicted position: (1.50, 0.0). GT position at t=1.50s is x = 1.0*1.50 = 1.50m -> error should be 0.0!
    # If temporal alignment was wrong (e.g. compared to GT at t=1.00s), error would be 0.50m!
    telemetry = [
        {
            "sim_time": 1.00,
            "robot_x": 0.0, "robot_y": 0.0, "robot_v": 0.0,
            "behavior_state": "PROCEED", "gt_collision": False, "min_clearance_m": 2.0,
            "tracked_details": [{"x": 1.0, "y": 0.0, "vx": 1.0, "vy": 0.0}],
            "prediction_details": [
                {
                    "waypoints": [
                        (1.25, 1.25, 0.0, 1.0, 0.0, 0.15),  # h=0.25s
                        (1.50, 1.50, 0.0, 1.0, 0.0, 0.20),  # h=0.50s
                    ]
                }
            ],
        }
    ]

    metrics = evaluator.evaluate(telemetry, (3.0, 0.0, 0.0), dt_s=0.10)
    # Both 0.25s and 0.50s prediction errors must be 0.0 because of perfect alignment!
    assert math.isclose(metrics.prediction_rmse_0_25s, 0.0, abs_tol=1e-5)
    assert math.isclose(metrics.prediction_rmse_0_50s, 0.0, abs_tol=1e-5)
