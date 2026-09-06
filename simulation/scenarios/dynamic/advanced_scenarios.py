"""Deterministic specifications for Phase 13.13 Advanced Scenarios F through O.

Scenarios:
- F: Temporary Blockage (obstacle blocks corridor, stops/waits, clears, vehicle recovers)
- G1: Persistent Blockage with Alternate Route (replan finds alternate detour)
- G2: Persistent Blockage with No Alternate Route (safe failure / stop, no collision)
- H: Multiple Dynamic Obstacles (4 distinct obstacles, independent tracks & predictions)
- I1: Low Sensor Noise
- I2: Moderate Sensor Noise
- I3: High Sensor Noise
- J1: Short Detection Dropout
- J2: Medium Contiguous Dropout
- J3: Repeated Intermittent Dropout
- K: Stale / Out-of-Order Data Rejection
- L: Unknown Prediction Horizon Conservatism
- M: Mixed-Quality Tracks (TRACKING, DEGRADED, LOST, noisy, intermittent)
- N1: Narrow Passage — Clear viable corridor
- N2: Narrow Passage — Marginal corridor
- N3: Narrow Passage — Genuinely too narrow (safe stop)
- O: Local Deadlock / No Feasible Path (safe stop, zero collisions)
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from simulation.scenarios.dynamic.data_model import (
    DynamicObstacleSpec,
    DynamicScenarioConfig,
    MotionModelType,
)


def create_scenario_f_temporary_blockage(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario F: Dynamic obstacle blocks corridor temporarily and clears."""
    # Obstacle stays in path until t=1.0s, then moves laterally away to y=2.0m by t=2.0s
    obs = DynamicObstacleSpec(
        obstacle_id=601,
        motion_model=MotionModelType.WAYPOINT,
        size=(0.8, 0.5),
        initial_pose=(1.8, 0.0, 0.0),
        initial_velocity=(0.0, 0.0),
        waypoints=[
            (0.0, 1.8, 0.0),
            (0.3, 1.8, 0.0),
            (0.6, 1.8, 2.0),
            (10.0, 1.8, 2.0),
        ],
        measurement_noise_std=0.01,
        dropout_probability=0.0,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_f_temporary_blockage",
        name="Scenario F — Temporary Blockage",
        description="Obstacle blocks the intended local route temporarily, then clears, enabling recovery.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=15.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_g1_persistent_blockage_alternate(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario G1: Persistent stationary blockage with alternate route available."""
    obs = DynamicObstacleSpec(
        obstacle_id=701,
        motion_model=MotionModelType.STATIONARY,
        size=(0.8, 0.8),
        initial_pose=(2.0, 0.0, 0.0),
        measurement_noise_std=0.01,
        dropout_probability=0.0,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_g1_persistent_blockage_alternate",
        name="Scenario G1 — Persistent Blockage with Alternate Route",
        description="Obstacle permanently blocks direct path; planner replans around obstacle.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=18.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_g2_persistent_blockage_no_route(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario G2: Persistent blockage where corridor is blocked and no detour exists."""
    # Flanking static walls along y = 0.8 and y = -0.8
    static_walls = []
    for x in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        static_walls.append({"x": x, "y": 0.8, "radius": 0.35})
        static_walls.append({"x": x, "y": -0.8, "radius": 0.35})

    # Obstacle completely plugs the corridor at x=2.0
    obs = DynamicObstacleSpec(
        obstacle_id=702,
        motion_model=MotionModelType.STATIONARY,
        size=(0.8, 0.8),
        initial_pose=(2.0, 0.0, 0.0),
        measurement_noise_std=0.01,
        dropout_probability=0.0,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_g2_persistent_blockage_no_route",
        name="Scenario G2 — Persistent Blockage (No Route)",
        description="Obstacle permanently blocks walled corridor; replanning fails, safe stop enforced.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[obs],
        static_obstacles=static_walls,
        control_dt_s=0.10,
        max_duration_s=15.0,
        random_seed=seed,
        expected_outcome="SAFE_STOP",
    )


def create_scenario_h_multiple_obstacles(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario H: 4 simultaneous dynamic obstacles with diverse motions."""
    obs1 = DynamicObstacleSpec(
        obstacle_id=801,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(0.5, -1.6, 0.0),
        initial_velocity=(0.30, 0.0),
        measurement_noise_std=0.02,
    )
    obs2 = DynamicObstacleSpec(
        obstacle_id=802,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.7, 0.5),
        initial_pose=(2.5, -2.0, 1.5708),
        initial_velocity=(0.0, 0.35),
        measurement_noise_std=0.02,
    )
    obs3 = DynamicObstacleSpec(
        obstacle_id=803,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.6, 0.4),
        initial_pose=(3.5, 2.0, -2.356),
        initial_velocity=(-0.25, -0.20),
        measurement_noise_std=0.03,
    )
    obs4 = DynamicObstacleSpec(
        obstacle_id=804,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.9, 0.6),
        initial_pose=(1.8, 1.5, 0.0),
        initial_velocity=(0.15, 0.0),
        measurement_noise_std=0.02,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_h_multiple_obstacles",
        name="Scenario H — Multiple Dynamic Obstacles",
        description="4 simultaneous dynamic obstacles with distinct positions, velocities, and crossings.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[obs1, obs2, obs3, obs4],
        control_dt_s=0.10,
        max_duration_s=16.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_i1_noise_low(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario I1: Sensor noise — Low."""
    obs = DynamicObstacleSpec(
        obstacle_id=901,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(1.0, 1.5, 0.0),
        initial_velocity=(0.30, 0.0),
        measurement_noise_std=0.01,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_i1_noise_low",
        name="Scenario I1 — Sensor Noise (Low)",
        description="Obstacle tracking under low Gaussian position measurement noise (std=0.01m).",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_i2_noise_med(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario I2: Sensor noise — Moderate."""
    obs = DynamicObstacleSpec(
        obstacle_id=902,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(1.0, 1.5, 0.0),
        initial_velocity=(0.30, 0.0),
        measurement_noise_std=0.05,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_i2_noise_med",
        name="Scenario I2 — Sensor Noise (Moderate)",
        description="Obstacle tracking under moderate Gaussian position measurement noise (std=0.05m).",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_i3_noise_high(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario I3: Sensor noise — High."""
    obs = DynamicObstacleSpec(
        obstacle_id=903,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(1.0, 1.5, 0.0),
        initial_velocity=(0.30, 0.0),
        measurement_noise_std=0.12,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_i3_noise_high",
        name="Scenario I3 — Sensor Noise (High)",
        description="Obstacle tracking under high Gaussian position measurement noise (std=0.12m).",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_j1_dropout_short(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario J1: Detection dropout — Short isolated dropout."""
    obs = DynamicObstacleSpec(
        obstacle_id=1001,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(1.0, 1.5, 0.0),
        initial_velocity=(0.30, 0.0),
        dropout_windows=[(0.4, 0.7)],  # 0.3s dropout (3 misses)
        measurement_noise_std=0.02,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_j1_dropout_short",
        name="Scenario J1 — Detection Dropout (Short)",
        description="Isolated short dropout window (0.3s) causing coasting and degradation before recovery.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_j2_dropout_medium(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario J2: Detection dropout — Medium contiguous dropout."""
    obs = DynamicObstacleSpec(
        obstacle_id=1002,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(1.0, 1.5, 0.0),
        initial_velocity=(0.30, 0.0),
        dropout_windows=[(0.5, 1.6)],  # 1.1s contiguous dropout
        measurement_noise_std=0.02,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_j2_dropout_medium",
        name="Scenario J2 — Detection Dropout (Medium)",
        description="Medium contiguous dropout (1.1s) exercising DEGRADED and LOST status transitions.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_j3_dropout_intermittent(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario J3: Detection dropout — Repeated / intermittent dropout."""
    obs = DynamicObstacleSpec(
        obstacle_id=1003,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(1.0, 1.5, 0.0),
        initial_velocity=(0.30, 0.0),
        dropout_windows=[(0.3, 0.5), (0.8, 1.0), (1.4, 1.6)],
        measurement_noise_std=0.02,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_j3_dropout_intermittent",
        name="Scenario J3 — Detection Dropout (Intermittent)",
        description="Repeated intermittent dropout windows testing lifecycle stability.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_k_stale_data(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario K: Stale / Out-of-order data injection."""
    obs = DynamicObstacleSpec(
        obstacle_id=1101,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(1.0, 1.5, 0.0),
        initial_velocity=(0.30, 0.0),
        stale_timestamp_offset_s=0.20,  # Positive offset generates stale/lagging timestamps
        measurement_noise_std=0.02,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_k_stale_data",
        name="Scenario K — Stale / Out-of-Order Data",
        description="Injects backward-moving timestamps; runner must reject them to preserve monotonicity.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_l_unknown_horizon(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario L: Unknown prediction horizon (conservative safety policy)."""
    # Obstacle with limited motion information / truncated prediction
    obs = DynamicObstacleSpec(
        obstacle_id=1201,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(2.2, 0.0, 3.14159),
        initial_velocity=(-0.10, 0.0),
        measurement_noise_std=0.02,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_l_unknown_horizon",
        name="Scenario L — Unknown Prediction Horizon",
        description="Exercises unknown horizon safety: UNKNOWN != FREE, conservative stopping/waiting.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[obs],
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_m_mixed_quality_tracks(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario M: Mixed-quality tracks simultaneously present."""
    # Obstacle 1: High quality confirmed TRACKING
    obs1 = DynamicObstacleSpec(
        obstacle_id=1301,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(0.5, -1.8, 0.0),
        initial_velocity=(0.30, 0.0),
        measurement_noise_std=0.01,
    )
    # Obstacle 2: Medium dropout -> DEGRADED
    obs2 = DynamicObstacleSpec(
        obstacle_id=1302,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.7, 0.5),
        initial_pose=(1.0, 1.8, 0.0),
        initial_velocity=(0.25, 0.0),
        dropout_windows=[(0.4, 0.6)],
        measurement_noise_std=0.03,
    )
    # Obstacle 3: Prolonged dropout -> LOST
    obs3 = DynamicObstacleSpec(
        obstacle_id=1303,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.6, 0.4),
        initial_pose=(2.5, -1.8, 0.0),
        initial_velocity=(0.20, 0.0),
        dropout_windows=[(0.5, 1.4)],
        measurement_noise_std=0.02,
    )
    # Obstacle 4: High noise
    obs4 = DynamicObstacleSpec(
        obstacle_id=1304,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.7, 0.5),
        initial_pose=(3.0, 1.8, 0.0),
        initial_velocity=(0.20, 0.0),
        measurement_noise_std=0.10,
    )
    return DynamicScenarioConfig(
        scenario_id="scenario_m_mixed_quality_tracks",
        name="Scenario M — Mixed-Quality Tracks",
        description="Scene contains TRACKING, DEGRADED, LOST, noisy, and intermittent tracks simultaneously.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[obs1, obs2, obs3, obs4],
        control_dt_s=0.10,
        max_duration_s=15.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_n1_narrow_clear(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario N1: Narrow passage with clear viable corridor."""
    # Flanking static obstacles at x=1.5, y=0.70 and y=-0.70 (width = 1.40m, viable for 0.45m UGV)
    static_obs = [
        {"x": 1.5, "y": 0.70, "radius": 0.25},
        {"x": 1.5, "y": -0.70, "radius": 0.25},
    ]
    return DynamicScenarioConfig(
        scenario_id="scenario_n1_narrow_clear",
        name="Scenario N1 — Narrow Passage (Clear Viable Corridor)",
        description="Corridor width 1.4m allows safe clearance for UGV (0.45m width).",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.0, 0.0, 0.0),
        static_obstacles=static_obs,
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_n2_narrow_marginal(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario N2: Narrow passage with marginal corridor."""
    # Flanking static obstacles at x=1.5, y=0.45 and y=-0.45 (width = 0.90m, marginal clearance)
    static_obs = [
        {"x": 1.5, "y": 0.45, "radius": 0.20},
        {"x": 1.5, "y": -0.45, "radius": 0.20},
    ]
    return DynamicScenarioConfig(
        scenario_id="scenario_n2_narrow_marginal",
        name="Scenario N2 — Narrow Passage (Marginal Corridor)",
        description="Corridor width 0.90m creates marginal clearance around UGV footprint.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.0, 0.0, 0.0),
        static_obstacles=static_obs,
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SUCCESS",
    )


def create_scenario_n3_narrow_blocked(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario N3: Narrow passage genuinely too narrow (safe stop required)."""
    # Flanking static obstacles at x=1.5, y=0.20 and y=-0.20 (width = 0.40m, smaller than 0.45m UGV)
    static_obs = [
        {"x": 1.5, "y": 0.25, "radius": 0.25},
        {"x": 1.5, "y": -0.25, "radius": 0.25},
    ]
    return DynamicScenarioConfig(
        scenario_id="scenario_n3_narrow_blocked",
        name="Scenario N3 — Narrow Passage (Too Narrow / Safe Stop)",
        description="Corridor width 0.40m is physically impassable; UGV must safely stop before entry.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.0, 0.0, 0.0),
        static_obstacles=static_obs,
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SAFE_STOP",
    )


def create_scenario_o_deadlock(seed: int = 42) -> DynamicScenarioConfig:
    """Scenario O: Local deadlock / No feasible path (safe stop, zero collision)."""
    # Box trap enclosing UGV forward field
    static_obs = [
        {"x": 1.5, "y": 0.0, "radius": 0.35},
        {"x": 1.5, "y": 0.5, "radius": 0.35},
        {"x": 1.5, "y": -0.5, "radius": 0.35},
        {"x": 0.8, "y": 0.7, "radius": 0.35},
        {"x": 0.8, "y": -0.7, "radius": 0.35},
    ]
    return DynamicScenarioConfig(
        scenario_id="scenario_o_deadlock",
        name="Scenario O — Local Deadlock / No Feasible Local Path",
        description="UGV surrounded by unnavigable obstacles with no feasible local trajectory.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.0, 0.0, 0.0),
        static_obstacles=static_obs,
        control_dt_s=0.10,
        max_duration_s=14.0,
        random_seed=seed,
        expected_outcome="SAFE_STOP",
    )


ADVANCED_SCENARIO_BUILDERS = {
    "scenario_f_temporary_blockage": create_scenario_f_temporary_blockage,
    "scenario_g1_persistent_blockage_alternate": create_scenario_g1_persistent_blockage_alternate,
    "scenario_g2_persistent_blockage_no_route": create_scenario_g2_persistent_blockage_no_route,
    "scenario_h_multiple_obstacles": create_scenario_h_multiple_obstacles,
    "scenario_i1_noise_low": create_scenario_i1_noise_low,
    "scenario_i2_noise_med": create_scenario_i2_noise_med,
    "scenario_i3_noise_high": create_scenario_i3_noise_high,
    "scenario_j1_dropout_short": create_scenario_j1_dropout_short,
    "scenario_j2_dropout_medium": create_scenario_j2_dropout_medium,
    "scenario_j3_dropout_intermittent": create_scenario_j3_dropout_intermittent,
    "scenario_k_stale_data": create_scenario_k_stale_data,
    "scenario_l_unknown_horizon": create_scenario_l_unknown_horizon,
    "scenario_m_mixed_quality_tracks": create_scenario_m_mixed_quality_tracks,
    "scenario_n1_narrow_clear": create_scenario_n1_narrow_clear,
    "scenario_n2_narrow_marginal": create_scenario_n2_narrow_marginal,
    "scenario_n3_narrow_blocked": create_scenario_n3_narrow_blocked,
    "scenario_o_deadlock": create_scenario_o_deadlock,
}


def export_advanced_scenarios_to_json(output_dir: str) -> List[str]:
    """Export all Advanced Scenarios F through O to reproducible JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for sc_id, builder in ADVANCED_SCENARIO_BUILDERS.items():
        cfg = builder()
        out_path = os.path.join(output_dir, f"{sc_id}.json")
        cfg.save_to_file(out_path)
        saved_paths.append(out_path)
    return saved_paths
