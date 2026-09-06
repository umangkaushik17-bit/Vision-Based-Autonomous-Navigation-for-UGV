"""Failure-injection and recovery scenario builders for Phase 13.14 (M14-06 through M14-17)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from simulation.scenarios.dynamic.data_model import (
    DynamicObstacleSpec,
    MotionModelType,
)
from simulation.scenarios.dynamic.phase14.data_model import (
    Phase14FaultSpec,
    Phase14FaultType,
    Phase14ScenarioConfig,
)


def create_scenario_m14_06_dropout(
    dropout_level: str = "LOW",
    seed: int = 101,
) -> Phase14ScenarioConfig:
    """M14-06: Detection Dropout.
    
    Levels:
    - LOW: 10% dropout
    - MEDIUM: 25% dropout
    - HIGH: 40% dropout
    """
    prob_map = {"LOW": 0.10, "MEDIUM": 0.25, "HIGH": 0.40}
    prob = prob_map.get(dropout_level.upper(), 0.10)

    obstacles = [
        DynamicObstacleSpec(
            obstacle_id="obs_drop_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(2.5, 0.3, 0.0),
            initial_velocity=(0.12, 0.0),
            confidence=0.95,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_drop_2",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(3.5, -1.5, math.pi / 2),
            initial_velocity=(0.0, 0.25),
            confidence=0.93,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_drop_3",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.75, 0.55),
            initial_pose=(4.5, 0.8, math.pi),
            initial_velocity=(-0.16, 0.0),
            confidence=0.94,
        ),
    ]

    faults = [
        Phase14FaultSpec(
            fault_id=f"fault_m14_06_dropout_{dropout_level.lower()}",
            fault_type=Phase14FaultType.DROPOUT,
            target_component="observation_generator",
            start_time=0.0,
            end_time=10.0,
            expected_response="TRACK_DEGRADED_OR_LOST_WITHOUT_COLLISION",
            parameters={"dropout_probability": prob},
        )
    ]

    return Phase14ScenarioConfig(
        scenario_id=f"m14_06_dropout_{dropout_level.lower()}",
        name=f"M14-06: Detection Dropout ({dropout_level.upper()} {int(prob*100)}%)",
        description=f"Multi-obstacle scene with {int(prob*100)}% detection dropout.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.0,  # 10 steps
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="PERCEPTION_FAILURE",
        fault_specs=faults,
    )


def create_scenario_m14_07_stale_track(seed: int = 1407) -> Phase14ScenarioConfig:
    """M14-07: Stale Track Handling.
    
    Observation gap causes track to age and be classified stale, then reacquired.
    """
    obstacles = [
        DynamicObstacleSpec(
            obstacle_id="obs_stale_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(2.8, 0.2, 0.0),
            initial_velocity=(0.10, 0.0),
            confidence=0.95,
        ),
    ]

    faults = [
        # Injects stale timestamp offset between t=0.3s and t=0.7s
        Phase14FaultSpec(
            fault_id="fault_m14_07_stale",
            fault_type=Phase14FaultType.STALE_TIMESTAMP,
            target_component="observation_generator",
            start_time=0.30,
            end_time=0.70,
            expected_response="STALE_OBSERVATION_REJECTED",
            parameters={"obstacle_id": "obs_stale_1", "offset_s": 2.5},
        )
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_07_stale_track",
        name="M14-07: Stale Track Handling",
        description="Observation timestamp corruption testing invariant K rejection and recovery.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.0,
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="PERCEPTION_FAILURE",
        fault_specs=faults,
    )


def create_scenario_m14_08_noisy_multi_object(seed: int = 1408) -> Phase14ScenarioConfig:
    """M14-08: Noisy Multi-Object Perception.
    
    3 obstacles with Gaussian position noise (sigma=0.15m) and confidence fluctuation.
    """
    obstacles = [
        DynamicObstacleSpec(
            obstacle_id="obs_noisy_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(2.2, 0.4, 0.0),
            initial_velocity=(0.12, 0.0),
            confidence=0.90,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_noisy_2",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(3.2, -1.2, math.pi / 2),
            initial_velocity=(0.0, 0.22),
            confidence=0.88,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_noisy_3",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.75, 0.55),
            initial_pose=(4.2, 0.5, math.pi),
            initial_velocity=(-0.15, 0.0),
            confidence=0.89,
        ),
    ]

    faults = [
        Phase14FaultSpec(
            fault_id="fault_m14_08_noise",
            fault_type=Phase14FaultType.NOISY_OBSERVATIONS,
            target_component="observation_generator",
            start_time=0.0,
            end_time=10.0,
            expected_response="KALMAN_FILTER_SMOOTHING",
            parameters={"pos_std": 0.15, "conf_std": 0.10},
        )
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_08_noisy_perception",
        name="M14-08: Noisy Multi-Object Perception",
        description="Gaussian noise on 3 targets measuring tracking and velocity RMSE.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.0,
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="PERCEPTION_FAILURE",
        fault_specs=faults,
        noise_position_std=0.15,
        noise_confidence_std=0.10,
    )


def create_scenario_m14_09_planner_saturation(seed: int = 1409) -> Phase14ScenarioConfig:
    """M14-09: Multi-Obstacle Planner Saturation.
    
    Dense cluster of obstacles where candidate rollouts are largely in collision,
    forcing DWA velocity reduction or wait/stop.
    """
    obstacles = [
        DynamicObstacleSpec(
            obstacle_id="obs_sat_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.85, 0.65),
            initial_pose=(1.1, 0.1, math.pi),
            initial_velocity=(-0.10, 0.0),
            confidence=0.96,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_sat_2",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.85, 0.65),
            initial_pose=(1.3, -0.5, math.pi),
            initial_velocity=(-0.10, 0.0),
            confidence=0.95,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_sat_3",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.85, 0.65),
            initial_pose=(1.3, 0.6, math.pi),
            initial_velocity=(-0.10, 0.0),
            confidence=0.95,
        ),
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_09_planner_saturation",
        name="M14-09: Multi-Obstacle Planner Saturation",
        description="High-density obstacle cluster evaluating DWA candidate scarcity.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.0,
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="PLANNER_STRESS",
    )


def create_scenario_m14_10_zero_valid_trajectories(seed: int = 1410) -> Phase14ScenarioConfig:
    """M14-10: Zero-Valid-Trajectory Injection.
    
    Tests current frozen stack behavior when DWA candidate set is empty.
    """
    faults = [
        Phase14FaultSpec(
            fault_id="fault_m14_10_zero_valid",
            fault_type=Phase14FaultType.ZERO_VALID_TRAJECTORIES,
            target_component="dwa_planner",
            start_time=0.30,
            end_time=0.70,
            expected_response="SAFE_STOP_OR_CLAMP",
        )
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_10_zero_valid_trajectories",
        name="M14-10: Zero-Valid-Trajectory Injection",
        description="Simulated zero valid candidates testing behavioral safe fallback.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=[],
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.0,
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="PLANNER_STRESS",
        fault_specs=faults,
    )


def create_scenario_m14_11_replanner_failure(case: str = "A", seed: int = 1411) -> Phase14ScenarioConfig:
    """M14-11: Global Replanner Failure.
    
    - Case A: Replan returns no path (REPLANNER_NO_PATH)
    - Case B: Replan request fails with exception (REPLANNER_EXCEPTION)
    - Case C: Replan result is invalid (REPLANNER_INVALID_RESULT)
    """
    case_map = {
        "A": Phase14FaultType.REPLANNER_NO_PATH,
        "B": Phase14FaultType.REPLANNER_EXCEPTION,
        "C": Phase14FaultType.REPLANNER_INVALID_RESULT,
    }
    ftype = case_map.get(case.upper(), Phase14FaultType.REPLANNER_NO_PATH)

    obstacles = [
        DynamicObstacleSpec(
            obstacle_id="obs_replan_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(2.5, 0.0, 0.0),
            initial_velocity=(0.0, 0.0),  # Static blocking obstacle
            confidence=0.96,
        ),
    ]

    faults = [
        Phase14FaultSpec(
            fault_id=f"fault_m14_11_replanner_case_{case.lower()}",
            fault_type=ftype,
            target_component="global_replanner",
            start_time=0.20,
            end_time=1.0,
            expected_response="STALE_PATH_NOT_ACCEPTED_AND_SAFE_FALLBACK",
        )
    ]

    return Phase14ScenarioConfig(
        scenario_id=f"m14_11_replanner_failure_case_{case.lower()}",
        name=f"M14-11: Global Replanner Failure (Case {case.upper()})",
        description=f"Injected replanner failure ({ftype.value}) testing fallback safety.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.0,
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="REPLANNER_FAILURE",
        fault_specs=faults,
    )


def create_scenario_m14_12_replanner_timeout(seed: int = 1412) -> Phase14ScenarioConfig:
    """M14-12: Replanner Response Timeout.
    
    Simulates late or missing replan response exceeding timeout boundary.
    """
    faults = [
        Phase14FaultSpec(
            fault_id="fault_m14_12_timeout",
            fault_type=Phase14FaultType.REPLANNER_TIMEOUT,
            target_component="global_replanner",
            start_time=0.20,
            end_time=1.0,
            expected_response="TIMEOUT_RECORDED_AND_FALLBACK_ACTIVE",
        )
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_12_replanner_timeout",
        name="M14-12: Replanner Response Timeout",
        description="Global replanner timeout testing bounded latency fallback.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=[],
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.0,
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="REPLANNER_FAILURE",
        fault_specs=faults,
    )


def create_scenario_m14_13_tracker_degradation(seed: int = 1413) -> Phase14ScenarioConfig:
    """M14-13: Tracker Degradation.
    
    Injects temporary track loss and tracker reset.
    """
    obstacles = [
        DynamicObstacleSpec(
            obstacle_id="obs_trk_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(2.5, 0.2, 0.0),
            initial_velocity=(0.10, 0.0),
            confidence=0.95,
        ),
    ]

    faults = [
        Phase14FaultSpec(
            fault_id="fault_m14_13_tracker_loss",
            fault_type=Phase14FaultType.TRACK_LOSS,
            target_component="tracker",
            start_time=0.30,
            end_time=0.60,
            expected_response="GRACEFUL_TRACK_LOSS_HANDLING",
        )
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_13_tracker_degradation",
        name="M14-13: Tracker Degradation",
        description="Track loss injection testing downstream stability without GT leakage.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.0,
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="TRACKER_FAILURE",
        fault_specs=faults,
    )


def create_scenario_m14_14_prediction_degradation(seed: int = 1414) -> Phase14ScenarioConfig:
    """M14-14: Prediction Degradation.
    
    Injects shortened prediction horizon and degraded confidence.
    """
    obstacles = [
        DynamicObstacleSpec(
            obstacle_id="obs_pred_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(2.5, 0.2, 0.0),
            initial_velocity=(0.10, 0.0),
            confidence=0.95,
        ),
    ]

    faults = [
        Phase14FaultSpec(
            fault_id="fault_m14_14_pred_shorten",
            fault_type=Phase14FaultType.PREDICTION_HORIZON_SHORTEN,
            target_component="predictor",
            start_time=0.30,
            end_time=0.70,
            expected_response="OCCUPANCY_HANDLES_SHORTENED_HORIZON",
            parameters={"max_points": 2},
        )
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_14_prediction_degradation",
        name="M14-14: Prediction Degradation",
        description="Truncated prediction horizon testing occupancy layer robustness.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.0,
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="TRACKER_FAILURE",
        fault_specs=faults,
    )


def create_scenario_m14_15_invalid_planner_output(seed: int = 1415) -> Phase14ScenarioConfig:
    """M14-15: Invalid Planner Output.
    
    Injects malformed / non-finite coordinates from planner to verify supervisory safety clamping.
    """
    faults = [
        Phase14FaultSpec(
            fault_id="fault_m14_15_invalid_output",
            fault_type=Phase14FaultType.MALFORMED_PLANNER_OUTPUT,
            target_component="dwa_planner",
            start_time=0.30,
            end_time=0.60,
            expected_response="SAFETY_LAYER_CLAMPS_COMMAND_TO_ZERO",
        )
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_15_invalid_planner_output",
        name="M14-15: Invalid Planner Output",
        description="Malformed planner outputs testing supervisory safety clamping.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=[],
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.0,
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="SAFETY_TEST",
        fault_specs=faults,
    )


def create_scenario_m14_16_recovery(seed: int = 1416) -> Phase14ScenarioConfig:
    """M14-16: Obstacle / Failure Recovery.
    
    Starts with 3 dynamic obstacles and temporary detector blackout (t=0.3s to t=0.8s).
    Once the fault clears, observations and tracks recover, and vehicle resumes safe motion.
    """
    obstacles = [
        DynamicObstacleSpec(
            obstacle_id="obs_rec_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(2.2, 0.3, 0.0),
            initial_velocity=(0.10, 0.0),
            confidence=0.95,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_rec_2",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(3.2, -1.4, math.pi / 2),
            initial_velocity=(0.0, 0.22),
            confidence=0.93,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_rec_3",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.75, 0.55),
            initial_pose=(4.2, 0.6, math.pi),
            initial_velocity=(-0.14, 0.0),
            confidence=0.94,
        ),
    ]

    faults = [
        # Temporary 100% blackout for 0.5s, then restores
        Phase14FaultSpec(
            fault_id="fault_m14_16_temporary_blackout",
            fault_type=Phase14FaultType.DROPOUT,
            target_component="observation_generator",
            start_time=0.30,
            end_time=0.80,
            expected_response="REACQUIRE_TRACKS_AND_RESUME_MOTION",
            parameters={"dropout_probability": 1.0},
        )
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_16_recovery",
        name="M14-16: Fault Recovery After Clearance",
        description="Temporary blackout clears; validates reacquisition and motion resumption.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=2.0,  # 20 steps (~30s CPU)
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="RECOVERY",
        fault_specs=faults,
    )


def create_scenario_m14_17_compound_failure(seed: int = 1417) -> Phase14ScenarioConfig:
    """M14-17: Compound Failure Safety Case.
    
    4 dynamic obstacles + detector dropout + global replanner failure.
    Verifies that multiple simultaneous faults yield safe degraded behavior with zero collision.
    """
    obstacles = [
        DynamicObstacleSpec(
            obstacle_id="obs_comp_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(2.4, 0.2, 0.0),
            initial_velocity=(0.10, 0.0),
            confidence=0.95,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_comp_2",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(3.2, -1.5, math.pi / 2),
            initial_velocity=(0.0, 0.24),
            confidence=0.93,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_comp_3",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.75, 0.55),
            initial_pose=(4.0, 0.8, math.pi),
            initial_velocity=(-0.16, 0.0),
            confidence=0.94,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_comp_4",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(3.8, -1.0, 3 * math.pi / 4),
            initial_velocity=(-0.12, 0.12),
            confidence=0.91,
        ),
    ]

    faults = [
        # Detector dropout 30%
        Phase14FaultSpec(
            fault_id="fault_m14_17_dropout",
            fault_type=Phase14FaultType.DROPOUT,
            target_component="observation_generator",
            start_time=0.20,
            end_time=1.80,
            expected_response="DEGRADED_PERCEPTION_HANDLED",
            parameters={"dropout_probability": 0.30},
        ),
        # Replanner failure (returns no path)
        Phase14FaultSpec(
            fault_id="fault_m14_17_replanner",
            fault_type=Phase14FaultType.REPLANNER_NO_PATH,
            target_component="global_replanner",
            start_time=0.40,
            end_time=1.80,
            expected_response="SAFE_DEGRADED_STATE_AND_ZERO_COLLISION",
        ),
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_17_compound_failure",
        name="M14-17: Compound Failure Safety Case",
        description="4 obstacles + 30% dropout + replanner failure verifying zero collision.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=2.0,  # 20 steps (~30s CPU)
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="COMPOUND_FAILURE",
        fault_specs=faults,
    )
