"""Multi-obstacle scenario builders for Phase 13.14 (M14-01 through M14-05)."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from simulation.scenarios.dynamic.data_model import (
    DynamicObstacleSpec,
    MotionModelType,
)
from simulation.scenarios.dynamic.phase14.data_model import (
    Phase14ScenarioConfig,
)


def create_scenario_m14_01_nominal(seed: int = 1401) -> Phase14ScenarioConfig:
    """M14-01: 4-Obstacle Nominal Scene.
    
    Validates basic multi-object propagation through detection, tracking, prediction,
    occupancy, collision checking, and DWA with 4 non-overlapping simultaneous obstacles.
    """
    obstacles = [
        # Obstacle 1: slowly moving ahead in forward lane
        DynamicObstacleSpec(
            obstacle_id="obs_nominal_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(2.5, 0.3, 0.0),
            initial_velocity=(0.12, 0.0),
            confidence=0.96,
        ),
        # Obstacle 2: crossing path from right to left ahead of UGV
        DynamicObstacleSpec(
            obstacle_id="obs_nominal_2",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(3.6, -1.8, math.pi / 2),
            initial_velocity=(0.0, 0.28),
            confidence=0.94,
        ),
        # Obstacle 3: diagonal approach from left
        DynamicObstacleSpec(
            obstacle_id="obs_nominal_3",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.75, 0.55),
            initial_pose=(1.8, 1.6, -math.pi / 4),
            initial_velocity=(0.14, -0.14),
            confidence=0.92,
        ),
        # Obstacle 4: oncoming in opposite lane
        DynamicObstacleSpec(
            obstacle_id="obs_nominal_4",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(4.8, 0.8, math.pi),
            initial_velocity=(-0.18, 0.0),
            confidence=0.95,
        ),
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_01_nominal_4obs",
        name="M14-01: 4-Obstacle Nominal Scene",
        description="4 simultaneous dynamic obstacles validating tracking and prediction pipeline.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.0,  # Short SIL: 10 steps (~15-20s CPU)
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="MULTI_OBSTACLE",
    )


def create_scenario_m14_02_crossing(seed: int = 1402) -> Phase14ScenarioConfig:
    """M14-02: 4-Obstacle Crossing.
    
    Stresses multi-object association and trajectory prediction:
    - Obstacle 1 crosses UGV path.
    - Obstacle 2 crosses Obstacle 1's path.
    - Obstacle 3 runs parallel to UGV.
    - Obstacle 4 approaches from an offset angle.
    """
    obstacles = [
        # Obstacle 1: Crosses UGV path from right to left
        DynamicObstacleSpec(
            obstacle_id="obs_cross_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(2.6, -1.8, math.pi / 2),
            initial_velocity=(0.0, 0.32),
            confidence=0.95,
        ),
        # Obstacle 2: Crosses Obstacle 1 path diagonally
        DynamicObstacleSpec(
            obstacle_id="obs_cross_2",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(1.2, 0.8, -0.3),
            initial_velocity=(0.28, -0.09),
            confidence=0.93,
        ),
        # Obstacle 3: Parallel to UGV on left flank
        DynamicObstacleSpec(
            obstacle_id="obs_cross_3",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.75, 0.55),
            initial_pose=(1.5, 1.8, 0.0),
            initial_velocity=(0.22, 0.0),
            confidence=0.94,
        ),
        # Obstacle 4: Approaches from offset diagonal angle
        DynamicObstacleSpec(
            obstacle_id="obs_cross_4",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(4.2, -1.5, 3 * math.pi / 4),
            initial_velocity=(-0.18, 0.18),
            confidence=0.91,
        ),
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_02_crossing_4obs",
        name="M14-02: 4-Obstacle Crossing",
        description="Complex 4-obstacle crossing geometry stressing multi-target association.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.0,  # Short SIL: 10 steps
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="MULTI_OBSTACLE",
    )


def create_scenario_m14_03_mixed_static_dynamic(seed: int = 1403) -> Phase14ScenarioConfig:
    """M14-03: Mixed Static + Dynamic Obstacles.
    
    Verifies that static terrain cost and dynamic predicted occupancy are
    coherently fused without collision.
    """
    static_obs = [
        {"x": 2.5, "y": -0.6, "radius": 0.40},
        {"x": 2.5, "y": 1.6, "radius": 0.40},
    ]

    dynamic_obs = [
        # Obstacle 1: moving between static obstacles
        DynamicObstacleSpec(
            obstacle_id="obs_mixed_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.75, 0.55),
            initial_pose=(1.8, 0.5, 0.0),
            initial_velocity=(0.14, 0.0),
            confidence=0.95,
        ),
        # Obstacle 2: crossing from lower region
        DynamicObstacleSpec(
            obstacle_id="obs_mixed_2",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(3.2, -1.6, math.pi / 2),
            initial_velocity=(0.0, 0.22),
            confidence=0.92,
        ),
        # Obstacle 3: oncoming from top region
        DynamicObstacleSpec(
            obstacle_id="obs_mixed_3",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(4.5, 0.6, math.pi),
            initial_velocity=(-0.16, 0.0),
            confidence=0.94,
        ),
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_03_mixed_static_dynamic",
        name="M14-03: Mixed Static + Dynamic Obstacles",
        description="Static pillars combined with 3 dynamic obstacles evaluating fusion layer.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=dynamic_obs,
        static_obstacles=static_obs,
        control_dt_s=0.10,
        max_duration_s=1.0,  # Short SIL: 10 steps
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="MULTI_OBSTACLE",
    )


def create_scenario_m14_04_dynamic_bottleneck(seed: int = 1404) -> Phase14ScenarioConfig:
    """M14-04: Dynamic Bottleneck (Longer SIL).
    
    Narrow corridor where multiple obstacles temporally shift the feasible opening,
    requiring continuous path reassessment and velocity regulation.
    """
    # Channel formed by static barriers
    static_barriers = [
        {"x": 2.0, "y": 1.4, "radius": 0.45},
        {"x": 3.2, "y": 1.4, "radius": 0.45},
        {"x": 2.0, "y": -1.4, "radius": 0.45},
        {"x": 3.2, "y": -1.4, "radius": 0.45},
    ]

    dynamic_obs = [
        # Obstacle 1: sweeps across bottleneck opening downwards
        DynamicObstacleSpec(
            obstacle_id="obs_bottle_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(2.2, 0.8, -math.pi / 2),
            initial_velocity=(0.0, -0.25),
            confidence=0.95,
        ),
        # Obstacle 2: sweeps across bottleneck exit upwards
        DynamicObstacleSpec(
            obstacle_id="obs_bottle_2",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(3.2, -0.8, math.pi / 2),
            initial_velocity=(0.0, 0.22),
            confidence=0.94,
        ),
        # Obstacle 3: leading vehicle exiting the corridor
        DynamicObstacleSpec(
            obstacle_id="obs_bottle_3",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(4.2, 0.0, 0.0),
            initial_velocity=(0.10, 0.0),
            confidence=0.92,
        ),
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_04_dynamic_bottleneck",
        name="M14-04: Dynamic Bottleneck",
        description="Temporally shifting corridor bottleneck testing velocity regulation and clearance.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=dynamic_obs,
        static_obstacles=static_barriers,
        control_dt_s=0.10,
        max_duration_s=2.5,  # 25 steps (~40-50s CPU) - bounded representative long SIL
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="MULTI_OBSTACLE",
    )


def create_scenario_m14_05_multi_head_on(seed: int = 1405) -> Phase14ScenarioConfig:
    """M14-05: Multi-Obstacle Head-On.
    
    3 dynamic obstacles simultaneously approaching the forward corridor,
    stressing candidate trajectory scarcity and emergency safety clamping.
    """
    obstacles = [
        # Obstacle 1: Direct head-on in central path
        DynamicObstacleSpec(
            obstacle_id="obs_headon_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(4.0, 0.0, math.pi),
            initial_velocity=(-0.25, 0.0),
            confidence=0.96,
        ),
        # Obstacle 2: Flanking from left corridor
        DynamicObstacleSpec(
            obstacle_id="obs_headon_2",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.75, 0.55),
            initial_pose=(4.4, 0.6, math.pi),
            initial_velocity=(-0.22, 0.0),
            confidence=0.93,
        ),
        # Obstacle 3: Flanking from right corridor
        DynamicObstacleSpec(
            obstacle_id="obs_headon_3",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.75, 0.55),
            initial_pose=(4.6, -0.6, math.pi),
            initial_velocity=(-0.20, 0.0),
            confidence=0.92,
        ),
    ]

    return Phase14ScenarioConfig(
        scenario_id="m14_05_multi_head_on",
        name="M14-05: Multi-Obstacle Head-On",
        description="3 dynamic obstacles converging on forward path testing trajectory scarcity.",
        world_bounds_m=((-3.0, 10.0), (-5.0, 5.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(5.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=1.2,  # 12 steps (~20s CPU)
        random_seed=seed,
        expected_outcome="SAFE",
        goal_tolerance_m=0.25,
        benchmark_category="MULTI_OBSTACLE",
    )
