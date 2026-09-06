"""Deterministic specifications for Phase 13.13 Core Scenarios A through E."""

from __future__ import annotations

import os
from typing import Dict, List

from simulation.scenarios.dynamic.data_model import (
    DynamicObstacleSpec,
    DynamicScenarioConfig,
    MotionModelType,
)


def create_scenario_a_static(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario A: Static world / no dynamic hazards (baseline regression)."""
    return DynamicScenarioConfig(
        scenario_id="scenario_a_static",
        name="Scenario A — Static / No Dynamic Hazard",
        description="Regression baseline: open traversable corridor with no dynamic obstacles.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.0, 0.0, 0.0),
        obstacles=[],
        control_dt_s=0.10,
        max_duration_s=12.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_b_parallel(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario B: Moving obstacle parallel to UGV."""
    obs = DynamicObstacleSpec(
        obstacle_id=201,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(0.5, 1.6, 0.0),
        initial_velocity=(0.35, 0.0),
        measurement_noise_std=0.03,
        dropout_probability=0.05,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_b_parallel",
        name="Scenario B — Moving Obstacle Parallel to Vehicle",
        description="Dynamic obstacle moves forward in adjacent corridor parallel to vehicle.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_c_head_on(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario C: Head-on collision course approaching UGV path."""
    obs = DynamicObstacleSpec(
        obstacle_id=301,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(3.8, 0.0, 3.14159),
        initial_velocity=(-0.30, 0.0),
        measurement_noise_std=0.02,
        dropout_probability=0.0,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_c_head_on",
        name="Scenario C — Head-On Collision Course",
        description="Dynamic obstacle approaches head-on along the planned UGV route.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_d1_crossing_left_to_right(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario D1: Lateral crossing from left to right."""
    obs = DynamicObstacleSpec(
        obstacle_id=401,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.7, 0.5),
        initial_pose=(2.0, -1.8, 1.5708),
        initial_velocity=(0.0, 0.40),
        measurement_noise_std=0.02,
        dropout_probability=0.0,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_d1_crossing_left_to_right",
        name="Scenario D1 — Lateral Crossing (Left-to-Right)",
        description="Obstacle crosses UGV planned route laterally from left to right.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_d2_crossing_right_to_left(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario D2: Lateral crossing from right to left."""
    obs = DynamicObstacleSpec(
        obstacle_id=402,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.7, 0.5),
        initial_pose=(2.0, 1.8, -1.5708),
        initial_velocity=(0.0, -0.40),
        measurement_noise_std=0.02,
        dropout_probability=0.0,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_d2_crossing_right_to_left",
        name="Scenario D2 — Lateral Crossing (Right-to-Left)",
        description="Obstacle crosses UGV planned route laterally from right to left.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_e1_accelerating(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario E1: Accelerating dynamic obstacle."""
    obs = DynamicObstacleSpec(
        obstacle_id=501,
        motion_model=MotionModelType.ACCELERATED,
        size=(0.8, 0.5),
        initial_pose=(0.8, 1.4, 0.0),
        initial_velocity=(0.15, 0.0),
        acceleration=(0.12, 0.0),
        measurement_noise_std=0.03,
        dropout_probability=0.0,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_e1_accelerating",
        name="Scenario E1 — Accelerating Dynamic Obstacle",
        description="Obstacle accelerates forward, stressing constant-velocity prediction assumption.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_e2_decelerating(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario E2: Decelerating dynamic obstacle."""
    obs = DynamicObstacleSpec(
        obstacle_id=502,
        motion_model=MotionModelType.ACCELERATED,
        size=(0.8, 0.5),
        initial_pose=(0.8, 1.4, 0.0),
        initial_velocity=(0.55, 0.0),
        acceleration=(-0.10, 0.0),
        measurement_noise_std=0.03,
        dropout_probability=0.0,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_e2_decelerating",
        name="Scenario E2 — Decelerating Dynamic Obstacle",
        description="Obstacle decelerates forward, stressing constant-velocity prediction assumption.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


CORE_SCENARIO_BUILDERS = {
    "scenario_a_static": create_scenario_a_static,
    "scenario_b_parallel": create_scenario_b_parallel,
    "scenario_c_head_on": create_scenario_c_head_on,
    "scenario_d1_crossing_left_to_right": create_scenario_d1_crossing_left_to_right,
    "scenario_d2_crossing_right_to_left": create_scenario_d2_crossing_right_to_left,
    "scenario_e1_accelerating": create_scenario_e1_accelerating,
    "scenario_e2_decelerating": create_scenario_e2_decelerating,
}


def export_core_scenarios_to_json(output_dir: str) -> List[str]:
    """Export all Core Scenarios A through E to reproducible JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for sc_id, builder in CORE_SCENARIO_BUILDERS.items():
        cfg = builder()
        out_path = os.path.join(output_dir, f"{sc_id}.json")
        cfg.save_to_file(out_path)
        saved_paths.append(out_path)
    return saved_paths
