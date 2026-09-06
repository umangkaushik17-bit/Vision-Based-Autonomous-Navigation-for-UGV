"""Phase 13.13 Dynamic SIL Scenario Validation: Framework & Safety Tests.

Verifies:
1. Scenario loading and serialization.
2. Ground-truth and observation separation (isolation contract).
3. Deterministic seeding and reproducibility.
4. Obstacle motion generation (CV, CTR, Accel, Waypoint, Stationary).
5. Sensor noise and dropout injection models.
6. SAT polygon collision detection invariants.
7. Closed-loop state propagation and metric evaluation.
8. Hard safety invariants (zero ground-truth leakage, collision tracking).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from typing import List, Tuple

import numpy as np
import pytest

from simulation.scenarios.dynamic.data_model import (
    DynamicObstacleSpec,
    DynamicScenarioConfig,
    GroundTruthObstacleState,
    MotionModelType,
    polygon_intersection_sat,
)
from simulation.scenarios.dynamic.motion_models import (
    AcceleratedMotionModel,
    ConstantTurnRateMotionModel,
    ConstantVelocityMotionModel,
    StationaryObstacleMotionModel,
    WaypointMotionModel,
    create_motion_model,
)
from simulation.scenarios.dynamic.observation_model import DynamicObservationGenerator
from simulation.runners.dynamic_sil_runner import DynamicSILRunner, ScenarioRunMetrics


# =============================================================================
# 1. SCENARIO CONFIGURATION & SERIALIZATION
# =============================================================================

def test_scenario_config_serialization():
    """Verify scenario configuration serialization to/from dictionary and JSON file."""
    obs1 = DynamicObstacleSpec(
        obstacle_id=101,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.6),
        initial_pose=(2.0, 1.0, 0.0),
        initial_velocity=(0.5, 0.0),
        measurement_noise_std=0.03,
        dropout_probability=0.05,
    )
    obs2 = DynamicObstacleSpec(
        obstacle_id=102,
        motion_model=MotionModelType.WAYPOINT,
        size=(1.0, 1.0),
        initial_pose=(4.0, -1.0, 1.57),
        waypoints=[(4.0, -1.0), (4.0, 2.0), (2.0, 2.0)],
        spawn_time_s=1.0,
        despawn_time_s=10.0,
    )

    cfg = DynamicScenarioConfig(
        scenario_id="scenario_test_serialization",
        name="Test Serialization Scenario",
        description="Validates round-trip JSON serialization",
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(6.0, 0.0, 0.0),
        obstacles=[obs1, obs2],
        control_dt_s=0.10,
        max_duration_s=15.0,
        random_seed=12345,
        expected_outcome="SUCCESS",
    )

    d = cfg.to_dict()
    assert d["scenario_id"] == "scenario_test_serialization"
    assert len(d["obstacles"]) == 2
    assert d["obstacles"][0]["obstacle_id"] == 101
    assert d["obstacles"][1]["motion_model"] == "WAYPOINT"

    reconstructed = DynamicScenarioConfig.from_dict(d)
    assert reconstructed.scenario_id == cfg.scenario_id
    assert reconstructed.random_seed == cfg.random_seed
    assert len(reconstructed.obstacles) == 2
    assert reconstructed.obstacles[0].initial_velocity == (0.5, 0.0)
    assert reconstructed.obstacles[1].waypoints == [(4.0, -1.0), (4.0, 2.0), (2.0, 2.0)]

    # File serialization round-trip
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        cfg.save_to_file(temp_path)
        loaded = DynamicScenarioConfig.load_from_file(temp_path)
        assert loaded.scenario_id == cfg.scenario_id
        assert loaded.goal_pose == (6.0, 0.0, 0.0)
        assert len(loaded.obstacles) == 2
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =============================================================================
# 2. GROUND-TRUTH / OBSERVATION SEPARATION & ISOLATION (MANDATORY)
# =============================================================================

def test_ground_truth_isolation_contract():
    """MANDATORY CRITICAL TEST: Verify runtime perception stack receives ONLY

    isolated observations and CANNOT access scenario ground truth.

    Checks:
    - DynamicObservationGenerator strips true obstacle_id (uses ephemeral det_fXXXX_Y).
    - True velocity vector is NOT attached to the emitted detection.
    - True acceleration and yaw rate are NOT attached to the emitted detection.
    - Ground truth objects (GroundTruthObstacleState) are NEVER passed into tracker,
      predictor, occupancy layer, collision checker, or planner.
    """
    spec = DynamicObstacleSpec(
        obstacle_id=999,  # True internal ID
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.7, 0.5),
        initial_pose=(3.0, 0.0, 0.0),
        initial_velocity=(1.2, -0.4),  # True velocity
        acceleration=(0.1, 0.0),      # True acceleration
        yaw_rate_rad_s=0.05,          # True yaw rate
        measurement_noise_std=0.0,    # Zero noise to test pure values
        dropout_probability=0.0,
    )

    models = [create_motion_model(spec)]
    generator = DynamicObservationGenerator(motion_models=models, random_seed=42)

    detections = generator.generate_observations(timestamp_s=1.0, robot_pose=(0.0, 0.0, 0.0))
    assert len(detections) == 1
    det = detections[0]

    # Check 1: True ID must NOT leak into detection
    assert det.detection_id != "999"
    assert det.detection_id.startswith("det_f")

    # Check 2: Position should be observed at t=1.0 (x = 3.0 + 1.2*1.0 = 4.2)
    assert math.isclose(det.bbox_center[0], 4.2, abs_tol=1e-3)
    assert math.isclose(det.bbox_center[1], -0.4, abs_tol=1e-3)

    # Check 3: Raw detection has no velocity ground truth attribute
    assert not hasattr(det, "ground_truth_velocity")
    assert not hasattr(det, "true_id")
    assert not hasattr(det, "ground_truth")

    # Check 4: Inspect DynamicSILRunner component instances to verify no ground truth references
    cfg = DynamicScenarioConfig(
        scenario_id="isolation_check",
        name="Isolation Check",
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.0, 0.0, 0.0),
        obstacles=[spec],
    )
    runner = DynamicSILRunner(cfg)

    runtime_components = [
        runner.tracker,
        runner.predictor,
        runner.occupancy_layer,
        runner.collision_checker,
        runner.dwa_planner,
        runner.behavior_machine,
        runner.replanner,
    ]

    for comp in runtime_components:
        for attr_name in dir(comp):
            attr = getattr(comp, attr_name)
            assert attr is not runner.motion_models, f"{comp} holds direct reference to motion_models!"
            assert attr is not runner.config.obstacles, f"{comp} holds direct reference to obstacles spec!"


# =============================================================================
# 3. DETERMINISTIC SEEDING & REPRODUCIBILITY
# =============================================================================

def test_observation_generator_determinism():
    """Verify identical random seeds produce byte-identical observation sequences."""
    spec = DynamicObstacleSpec(
        obstacle_id=1,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.6),
        initial_pose=(2.0, 0.0, 0.0),
        initial_velocity=(0.2, 0.1),
        measurement_noise_std=0.08,
        dropout_probability=0.20,
    )

    gen1 = DynamicObservationGenerator([create_motion_model(spec)], random_seed=777)
    gen2 = DynamicObservationGenerator([create_motion_model(spec)], random_seed=777)

    for step in range(20):
        t = step * 0.10
        obs1 = gen1.generate_observations(t, (0.0, 0.0, 0.0))
        obs2 = gen2.generate_observations(t, (0.0, 0.0, 0.0))

        assert len(obs1) == len(obs2)
        if obs1:
            assert math.isclose(obs1[0].bbox_center[0], obs2[0].bbox_center[0], abs_tol=1e-9)
            assert math.isclose(obs1[0].bbox_center[1], obs2[0].bbox_center[1], abs_tol=1e-9)


def test_runner_determinism_reproducibility():
    """Verify running the same scenario twice with the same seed gives equivalent results."""
    spec = DynamicObstacleSpec(
        obstacle_id=1,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.6, 0.4),
        initial_pose=(3.0, 2.0, -1.57),
        initial_velocity=(0.0, -0.3),
        measurement_noise_std=0.02,
        dropout_probability=0.0,
    )
    cfg = DynamicScenarioConfig(
        scenario_id="reproducibility_test",
        name="Reproducibility Test",
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(2.0, 0.0, 0.0),
        obstacles=[spec],
        control_dt_s=0.10,
        max_duration_s=0.30,  # 3 steps
        random_seed=42,
    )

    runner1 = DynamicSILRunner(cfg)
    m1 = runner1.run()

    runner2 = DynamicSILRunner(cfg)
    m2 = runner2.run()

    assert m1.steps_executed == m2.steps_executed
    assert math.isclose(m1.travel_distance_m, m2.travel_distance_m, abs_tol=1e-6)
    assert m1.collision_count == m2.collision_count
    assert len(m1.telemetry) == len(m2.telemetry)
    for t1, t2 in zip(m1.telemetry, m2.telemetry):
        assert math.isclose(t1["robot_x"], t2["robot_x"], abs_tol=1e-6)
        assert math.isclose(t1["robot_y"], t2["robot_y"], abs_tol=1e-6)
        assert t1["behavior_state"] == t2["behavior_state"]


# =============================================================================
# 4. OBSTACLE MOTION GENERATION MODELS
# =============================================================================

def test_constant_velocity_motion_model():
    """Verify linear propagation: p(t) = p0 + v*t."""
    spec = DynamicObstacleSpec(
        obstacle_id=10,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(1.0, 2.0, 0.0),
        initial_velocity=(0.5, -0.25),
    )
    model = ConstantVelocityMotionModel(spec)

    state_t0 = model.evaluate(0.0)
    assert math.isclose(state_t0.x, 1.0)
    assert math.isclose(state_t0.y, 2.0)
    expected_yaw = math.atan2(-0.25, 0.5)
    assert math.isclose(state_t0.yaw, expected_yaw, abs_tol=1e-5)

    state_t2 = model.evaluate(2.0)
    assert math.isclose(state_t2.x, 1.0 + 0.5 * 2.0)
    assert math.isclose(state_t2.y, 2.0 + (-0.25) * 2.0)
    assert math.isclose(state_t2.vx, 0.5)
    assert math.isclose(state_t2.vy, -0.25)


def test_constant_turn_rate_motion_model():
    """Verify circular arc kinematics."""
    spec = DynamicObstacleSpec(
        obstacle_id=11,
        motion_model=MotionModelType.CONSTANT_TURN_RATE,
        size=(0.8, 0.5),
        initial_pose=(0.0, 0.0, 0.0),
        initial_velocity=(1.0, 0.0),  # v = 1.0 m/s heading east
        yaw_rate_rad_s=math.pi / 2.0,  # 90 deg/s
    )
    model = ConstantTurnRateMotionModel(spec)

    state_t1 = model.evaluate(1.0)
    assert math.isclose(state_t1.yaw, math.pi / 2.0, abs_tol=1e-5)
    R = 1.0 / (math.pi / 2.0)
    assert math.isclose(state_t1.x, R, abs_tol=1e-3)
    assert math.isclose(state_t1.y, R, abs_tol=1e-3)


def test_accelerated_motion_model():
    """Verify quadratic kinematics: p(t) = p0 + v0*t + 0.5*a*t^2."""
    spec = DynamicObstacleSpec(
        obstacle_id=12,
        motion_model=MotionModelType.ACCELERATED,
        size=(0.8, 0.5),
        initial_pose=(0.0, 0.0, 0.0),
        initial_velocity=(1.0, 0.0),
        acceleration=(0.5, 0.0),
    )
    model = AcceleratedMotionModel(spec)

    state_t2 = model.evaluate(2.0)
    assert math.isclose(state_t2.vx, 2.0)
    assert math.isclose(state_t2.x, 3.0)


def test_waypoint_motion_model():
    """Verify linear waypoint following."""
    wps = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    spec = DynamicObstacleSpec(
        obstacle_id=13,
        motion_model=MotionModelType.WAYPOINT,
        size=(0.8, 0.5),
        initial_pose=(0.0, 0.0, 0.0),
        initial_velocity=(1.0, 0.0),
        waypoints=wps,
    )
    model = WaypointMotionModel(spec)

    state_t1 = model.evaluate(1.0)
    assert math.isclose(state_t1.x, 1.0, abs_tol=1e-3)
    assert math.isclose(state_t1.y, 0.0, abs_tol=1e-3)

    state_t3 = model.evaluate(3.0)
    assert math.isclose(state_t3.x, 2.0, abs_tol=1e-3)
    assert math.isclose(state_t3.y, 1.0, abs_tol=1e-3)


def test_spawn_and_despawn_lifecycle():
    """Verify obstacle is active only inside [spawn_time_s, despawn_time_s]."""
    spec = DynamicObstacleSpec(
        obstacle_id=14,
        motion_model=MotionModelType.STATIONARY,
        size=(1.0, 1.0),
        initial_pose=(5.0, 5.0, 0.0),
        spawn_time_s=2.0,
        despawn_time_s=5.0,
    )
    model = StationaryObstacleMotionModel(spec)

    assert model.evaluate(1.9) is None
    assert model.evaluate(2.0).is_active
    assert model.evaluate(3.5).is_active
    assert model.evaluate(5.1) is None


# =============================================================================
# 5. SENSOR NOISE & DROPOUT INJECTION
# =============================================================================

def test_sensor_noise_statistics():
    """Verify Gaussian measurement noise distribution matches specified standard deviation."""
    noise_std = 0.15
    spec = DynamicObstacleSpec(
        obstacle_id=20,
        motion_model=MotionModelType.STATIONARY,
        size=(0.8, 0.6),
        initial_pose=(4.0, 2.0, 0.0),
        measurement_noise_std=noise_std,
        dropout_probability=0.0,
    )
    gen = DynamicObservationGenerator([create_motion_model(spec)], random_seed=123)

    errors_x = []
    errors_y = []
    for step in range(500):
        t = step * 0.05
        obs = gen.generate_observations(t, (0.0, 0.0, 0.0))
        assert len(obs) == 1
        errors_x.append(obs[0].bbox_center[0] - 4.0)
        errors_y.append(obs[0].bbox_center[1] - 2.0)

    measured_std_x = float(np.std(errors_x))
    measured_std_y = float(np.std(errors_y))
    measured_mean_x = float(np.mean(errors_x))

    assert math.isclose(measured_mean_x, 0.0, abs_tol=0.03)
    assert math.isclose(measured_std_x, noise_std, abs_tol=0.03)
    assert math.isclose(measured_std_y, noise_std, abs_tol=0.03)


def test_dropout_injection():
    """Verify dropout schedules match specified probability."""
    p_drop = 0.35
    spec = DynamicObstacleSpec(
        obstacle_id=21,
        motion_model=MotionModelType.STATIONARY,
        size=(0.8, 0.6),
        initial_pose=(3.0, 0.0, 0.0),
        measurement_noise_std=0.0,
        dropout_probability=p_drop,
    )
    gen = DynamicObservationGenerator([create_motion_model(spec)], random_seed=456)

    detected_count = 0
    total_steps = 1000
    for step in range(total_steps):
        t = step * 0.05
        obs = gen.generate_observations(t, (0.0, 0.0, 0.0))
        if obs:
            detected_count += 1

    actual_drop_rate = 1.0 - (detected_count / total_steps)
    assert math.isclose(actual_drop_rate, p_drop, abs_tol=0.04)


# =============================================================================
# 6. EXACT SAT POLYGON COLLISION DETECTION
# =============================================================================

def test_sat_collision_exact_cases():
    """Verify Separating Axis Theorem (SAT) detects polygon collisions accurately."""
    poly_center = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

    # Disjoint box
    poly_disjoint = [(2.0, -1.0), (4.0, -1.0), (4.0, 1.0), (2.0, 1.0)]
    assert not polygon_intersection_sat(poly_center, poly_disjoint)

    # Overlapping box
    poly_overlap = [(0.5, -1.0), (2.5, -1.0), (2.5, 1.0), (0.5, 1.0)]
    assert polygon_intersection_sat(poly_center, poly_overlap)

    # Rotated box separated diagonally
    poly_diag_clear = [(1.3, 1.3), (1.7, 1.3), (1.7, 1.7), (1.3, 1.7)]
    assert not polygon_intersection_sat(poly_center, poly_diag_clear)


# =============================================================================
# 7. CLOSED-LOOP PIPELINE INTEGRATION & METRICS
# =============================================================================

def test_closed_loop_runner_step_execution():
    """Verify closed-loop execution runs end-to-end and computes structured metrics."""
    spec = DynamicObstacleSpec(
        obstacle_id=50,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.6, 0.4),
        initial_pose=(2.0, 3.0, 0.0),
        initial_velocity=(0.1, 0.0),
        measurement_noise_std=0.01,
        dropout_probability=0.0,
    )
    cfg = DynamicScenarioConfig(
        scenario_id="step_exec_test",
        name="Step Execution Test",
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(1.5, 0.0, 0.0),
        obstacles=[spec],
        control_dt_s=0.10,
        max_duration_s=0.30,  # 3 steps
        random_seed=42,
    )

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert isinstance(metrics, ScenarioRunMetrics)
    assert metrics.scenario_id == "step_exec_test"
    assert metrics.steps_executed == 3
    assert math.isclose(metrics.total_time_s, 0.30, abs_tol=1e-3)
    assert metrics.collision_count == 0
    assert not metrics.collision_detected
    assert metrics.travel_distance_m >= 0.0
    assert metrics.planner_latency_mean_s >= 0.0


def test_collision_detection_event_logging():
    """Verify that an unavoidable collision is recorded in collision_count."""
    # Spawn an obstacle directly on top of the robot's initial position
    spec = DynamicObstacleSpec(
        obstacle_id=99,
        motion_model=MotionModelType.STATIONARY,
        size=(1.5, 1.5),
        initial_pose=(0.1, 0.0, 0.0),
    )
    cfg = DynamicScenarioConfig(
        scenario_id="collision_test",
        name="Immediate Collision",
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(2.0, 0.0, 0.0),
        obstacles=[spec],
        control_dt_s=0.10,
        max_duration_s=0.20,
        random_seed=42,
    )

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_count > 0
    assert metrics.collision_detected
    assert not metrics.mission_success
