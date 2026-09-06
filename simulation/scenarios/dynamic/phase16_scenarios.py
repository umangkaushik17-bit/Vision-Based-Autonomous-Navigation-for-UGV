"""Canonical scenario specifications for Milestone 13.16: Full Closed-Loop Benchmark (B01-B12).

Scenarios:
- B01: Flat Open Terrain Baseline (nominal)
- B02: Rough / Rocky Terrain (terrain-aware cost trade-off)
- B03: Static Hazard Environment (detour around boulders & high-cost zones)
- B04: Static Blockage / Detour (direct corridor blocked; global replan detour)
- B05: Single Dynamic Obstacle (lateral crossing, Kalman tracking, DWA reaction)
- B06: Multi-Dynamic Obstacles (crossing, parallel, approaching; multi-target tracking)
- B07: Mixed Static + Dynamic (static gate with crossing dynamic traffic)
- B08: Dynamic Bottleneck (moving obstacle temporarily chokes doorway; WAIT -> PROCEED recovery)
- B09: Perception Degraded (25% dropout + 0.08m measurement noise)
- B10: Localization Degraded (injected pose noise & bounded drift)
- B11: Sensor Failure + Recovery (pre-fault -> blackout -> recovery -> post-recovery navigation)
- B12: Compound Stress Mission (3 dynamic obstacles + perception noise + temporary replanner failure)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from simulation.scenarios.dynamic.data_model import (
    DynamicObstacleSpec,
    DynamicScenarioConfig,
    MotionModelType,
)
from simulation.scenarios.dynamic.phase14.data_model import (
    Phase14FaultSpec,
    Phase14FaultType,
    Phase14ScenarioConfig,
)


@dataclass
class Phase16ScenarioConfig(Phase14ScenarioConfig):
    """Scenario configuration extended for Phase 13.16 benchmark suite."""

    terrain_patches: List[Dict[str, Any]] = field(default_factory=list)
    localization_noise_xy_std: float = 0.0
    localization_noise_yaw_std: float = 0.0
    localization_drift_rate_m_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["terrain_patches"] = self.terrain_patches
        d["localization_noise_xy_std"] = self.localization_noise_xy_std
        d["localization_noise_yaw_std"] = self.localization_noise_yaw_std
        d["localization_drift_rate_m_s"] = self.localization_drift_rate_m_s
        return d


# ---------------------------------------------------------------------------
# B01 — Flat Open Terrain Baseline
# ---------------------------------------------------------------------------
def create_b01_flat_baseline(seed: int = 42) -> Phase16ScenarioConfig:
    """B01: Clean nominal baseline in flat open space with zero dynamic obstacles."""
    return Phase16ScenarioConfig(
        scenario_id="B01_flat_baseline",
        name="B01: Flat Open Terrain Baseline",
        description="Clean nominal baseline with flat terrain and zero obstacles to verify nominal navigation.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(3.5, 0.0, 0.0),
        obstacles=[],
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=15.0,
        random_seed=seed,
        expected_outcome="GOAL_REACHED",
        goal_tolerance_m=0.25,
        benchmark_category="NOMINAL",
    )


# ---------------------------------------------------------------------------
# B02 — Rough / Rocky Terrain
# ---------------------------------------------------------------------------
def create_b02_rough_terrain(seed: int = 42) -> Phase16ScenarioConfig:
    """B02: Navigation across terrain with elevated roughness / high traversability cost."""
    # High cost patch in the center corridor (x in [1.2, 2.8], y in [-1.2, 1.2]) with cost=0.75
    # Flanking corridors (y > 1.5 or y < -1.5) remain low cost (0.10)
    terrain_patches = [
        {"x_min": 1.2, "x_max": 2.8, "y_min": -1.2, "y_max": 1.2, "cost": 0.75}
    ]
    return Phase16ScenarioConfig(
        scenario_id="B02_rough_terrain",
        name="B02: Rough / Rocky Terrain",
        description="Traversability-aware navigation across uneven rocky terrain with elevated costs.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[],
        static_obstacles=[],
        terrain_patches=terrain_patches,
        control_dt_s=0.10,
        max_duration_s=18.0,
        random_seed=seed,
        expected_outcome="GOAL_REACHED",
        goal_tolerance_m=0.25,
        benchmark_category="TERRAIN",
    )


# ---------------------------------------------------------------------------
# B03 — Static Hazard Environment
# ---------------------------------------------------------------------------
def create_b03_static_hazards(seed: int = 42) -> Phase16ScenarioConfig:
    """B03: Several static obstacles and high-cost patches requiring local avoidance."""
    static_obstacles = [
        {"x": 2.0, "y": 0.0, "radius": 0.40},    # Boulder in center line
        {"x": 3.2, "y": -0.7, "radius": 0.35},   # Secondary hazard
    ]
    terrain_patches = [
        {"x_min": 1.6, "x_max": 2.8, "y_min": 0.5, "y_max": 1.4, "cost": 0.70}  # Mud patch
    ]
    return Phase16ScenarioConfig(
        scenario_id="B03_static_hazards",
        name="B03: Static Hazard Environment",
        description="Navigation through static boulders and mud patches requiring route adjustment.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[],
        static_obstacles=static_obstacles,
        terrain_patches=terrain_patches,
        control_dt_s=0.10,
        max_duration_s=18.0,
        random_seed=seed,
        expected_outcome="GOAL_REACHED",
        goal_tolerance_m=0.25,
        benchmark_category="STATIC_AVOIDANCE",
    )


# ---------------------------------------------------------------------------
# B04 — Static Blockage / Detour
# ---------------------------------------------------------------------------
def create_b04_static_detour(seed: int = 42) -> Phase16ScenarioConfig:
    """B04: Direct route blocked by a static wall; global replanner must find open lateral detour."""
    # Wall across x=2.0 from y=-2.0 to y=0.8. Lateral bypass open at y in [1.5, 3.5].
    wall = []
    for y_pos in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 0.8]:
        wall.append({"x": 2.0, "y": y_pos, "radius": 0.35})

    return Phase16ScenarioConfig(
        scenario_id="B04_static_detour",
        name="B04: Static Blockage / Detour",
        description="Direct corridor blocked by static wall; event-driven replanner executes lateral detour.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[],
        static_obstacles=wall,
        control_dt_s=0.10,
        max_duration_s=22.0,
        random_seed=seed,
        expected_outcome="GOAL_REACHED",
        goal_tolerance_m=0.25,
        benchmark_category="REPLANNING",
    )


# ---------------------------------------------------------------------------
# B05 — Single Dynamic Obstacle
# ---------------------------------------------------------------------------
def create_b05_single_dynamic(seed: int = 42) -> Phase16ScenarioConfig:
    """B05: Single dynamic obstacle crossing the UGV's path laterally."""
    obs = DynamicObstacleSpec(
        obstacle_id="obs_b05_cross",
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(2.0, -1.8, math.pi / 2),
        initial_velocity=(0.0, 0.30),
        measurement_noise_std=0.02,
        dropout_probability=0.0,
    )
    return Phase16ScenarioConfig(
        scenario_id="B05_single_dynamic",
        name="B05: Single Dynamic Obstacle",
        description="Single dynamic obstacle crossing path laterally; validates tracking, prediction, and DWA reaction.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[obs],
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=18.0,
        random_seed=seed,
        expected_outcome="GOAL_REACHED",
        goal_tolerance_m=0.25,
        benchmark_category="DYNAMIC_AVOIDANCE",
    )


# ---------------------------------------------------------------------------
# B06 — Multi-Dynamic Obstacles
# ---------------------------------------------------------------------------
def create_b06_multi_dynamic(seed: int = 42) -> Phase16ScenarioConfig:
    """B06: Primary multi-obstacle benchmark: 3 dynamic obstacles with crossing, parallel, approaching motions."""
    obstacles = [
        # Obstacle 1: Lateral crossing
        DynamicObstacleSpec(
            obstacle_id="obs_b06_cross",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.5),
            initial_pose=(2.0, -1.8, math.pi / 2),
            initial_velocity=(0.0, 0.28),
            measurement_noise_std=0.02,
            confidence=0.95,
        ),
        # Obstacle 2: Parallel in adjacent corridor
        DynamicObstacleSpec(
            obstacle_id="obs_b06_parallel",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(0.8, 1.2, 0.0),
            initial_velocity=(0.22, 0.0),
            measurement_noise_std=0.02,
            confidence=0.92,
        ),
        # Obstacle 3: Approaching at an angle
        DynamicObstacleSpec(
            obstacle_id="obs_b06_approach",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(4.0, 1.0, -2.6),
            initial_velocity=(-0.18, -0.10),
            measurement_noise_std=0.02,
            confidence=0.94,
        ),
    ]
    return Phase16ScenarioConfig(
        scenario_id="B06_multi_dynamic",
        name="B06: Multi-Dynamic Obstacles",
        description="Three dynamic obstacles (crossing, parallel, approaching); stresses multi-frame tracking and prediction.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        control_dt_s=0.10,
        max_duration_s=22.0,
        random_seed=seed,
        expected_outcome="GOAL_REACHED",
        goal_tolerance_m=0.25,
        benchmark_category="MULTI_TARGET",
    )


# ---------------------------------------------------------------------------
# B07 — Mixed Static + Dynamic
# ---------------------------------------------------------------------------
def create_b07_mixed_static_dynamic(seed: int = 42) -> Phase16ScenarioConfig:
    """B07: Integrated cost reasoning: static gate pillars with 2 dynamic crossing obstacles."""
    static_obstacles = [
        {"x": 2.2, "y": 1.0, "radius": 0.35},   # Top pillar
        {"x": 2.2, "y": -1.0, "radius": 0.35},  # Bottom pillar
    ]
    obstacles = [
        # Obstacle 1: Moving downward before gate
        DynamicObstacleSpec(
            obstacle_id="obs_b07_down",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(1.6, 1.8, -math.pi / 2),
            initial_velocity=(0.0, -0.28),
            measurement_noise_std=0.02,
        ),
        # Obstacle 2: Moving upward after gate
        DynamicObstacleSpec(
            obstacle_id="obs_b07_up",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(2.8, -1.8, math.pi / 2),
            initial_velocity=(0.0, 0.25),
            measurement_noise_std=0.02,
        ),
    ]
    return Phase16ScenarioConfig(
        scenario_id="B07_mixed_static_dynamic",
        name="B07: Mixed Static + Dynamic",
        description="Static gate passage with crossing dynamic traffic; validates unified spatial-temporal cost reasoning.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=static_obstacles,
        control_dt_s=0.10,
        max_duration_s=22.0,
        random_seed=seed,
        expected_outcome="GOAL_REACHED",
        goal_tolerance_m=0.25,
        benchmark_category="MIXED_COST",
    )


# ---------------------------------------------------------------------------
# B08 — Dynamic Bottleneck
# ---------------------------------------------------------------------------
def create_b08_dynamic_bottleneck(seed: int = 42) -> Phase16ScenarioConfig:
    """B08: Moving obstacle temporarily blocks a narrow passage, then clears, testing WAIT/PROCEED behavior."""
    # Static walls leaving a central opening at x=2.0
    static_walls = []
    for y_pos in [1.0, 1.5, 2.0, -1.0, -1.5, -2.0]:
        static_walls.append({"x": 2.0, "y": y_pos, "radius": 0.35})

    # Moving obstacle enters opening at t=1.0s, stops until t=4.0s, then clears to y=2.0m
    obs = DynamicObstacleSpec(
        obstacle_id="obs_b08_blocker",
        motion_model=MotionModelType.WAYPOINT,
        size=(0.8, 0.6),
        initial_pose=(2.0, 0.0, 0.0),
        waypoints=[
            (0.0, 2.0, 0.0),    # in bottleneck
            (1.0, 2.0, 0.0),    # holding bottleneck
            (4.0, 2.0, 0.0),    # holding
            (5.5, 2.0, 2.0),    # clearing to side
            (30.0, 2.0, 2.0),   # stays clear
        ],
        measurement_noise_std=0.01,
    )
    return Phase16ScenarioConfig(
        scenario_id="B08_dynamic_bottleneck",
        name="B08: Dynamic Bottleneck",
        description="Constrained doorway blocked by moving obstacle that clears; tests WAIT/STOP to PROCEED state recovery.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[obs],
        static_obstacles=static_walls,
        control_dt_s=0.10,
        max_duration_s=22.0,
        random_seed=seed,
        expected_outcome="GOAL_REACHED",
        goal_tolerance_m=0.25,
        benchmark_category="BOTTLENECK",
    )


# ---------------------------------------------------------------------------
# B09 — Perception Degraded
# ---------------------------------------------------------------------------
def create_b09_perception_degraded(seed: int = 42) -> Phase16ScenarioConfig:
    """B09: Multi-obstacle scenario with controlled 25% detection dropout and 0.08m measurement noise."""
    obstacles = [
        DynamicObstacleSpec(
            obstacle_id="obs_b09_1",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.5),
            initial_pose=(2.0, -1.8, math.pi / 2),
            initial_velocity=(0.0, 0.28),
            confidence=0.90,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_b09_2",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(3.2, 1.8, -math.pi / 2),
            initial_velocity=(0.0, -0.25),
            confidence=0.90,
        ),
    ]
    faults = [
        Phase14FaultSpec(
            fault_id="fault_b09_dropout",
            fault_type=Phase14FaultType.DROPOUT,
            target_component="observation_generator",
            start_time=0.0,
            end_time=30.0,
            expected_response="TRACK_DEGRADED_COASTING_SAFE",
            parameters={"dropout_probability": 0.25},
        ),
        Phase14FaultSpec(
            fault_id="fault_b09_noise",
            fault_type=Phase14FaultType.NOISY_OBSERVATIONS,
            target_component="observation_generator",
            start_time=0.0,
            end_time=30.0,
            expected_response="FILTER_COVARIANCE_BOUNDED",
            parameters={"position_noise_std": 0.08},
        ),
    ]
    return Phase16ScenarioConfig(
        scenario_id="B09_perception_degraded",
        name="B09: Perception Degraded",
        description="Controlled perception degradation (25% dropout, 0.08m noise); verifies tracker coasting and safe control.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        fault_specs=faults,
        control_dt_s=0.10,
        max_duration_s=18.0,
        random_seed=seed,
        expected_outcome="GOAL_REACHED",
        goal_tolerance_m=0.25,
        benchmark_category="PERCEPTION_ROBUSTNESS",
    )


# ---------------------------------------------------------------------------
# B10 — Localization Degraded
# ---------------------------------------------------------------------------
def create_b10_localization_degraded(seed: int = 42) -> Phase16ScenarioConfig:
    """B10: Controlled realistic localization degradation (pose noise and bounded drift)."""
    return Phase16ScenarioConfig(
        scenario_id="B10_localization_degraded",
        name="B10: Localization Degraded",
        description="Controlled pose measurement noise (0.08m std) and drift; tests path tracking and goal convergence.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[],
        static_obstacles=[],
        localization_noise_xy_std=0.08,
        localization_noise_yaw_std=0.04,
        localization_drift_rate_m_s=0.005,
        control_dt_s=0.10,
        max_duration_s=18.0,
        random_seed=seed,
        expected_outcome="GOAL_REACHED",
        goal_tolerance_m=0.25,
        benchmark_category="LOCALIZATION_ROBUSTNESS",
    )


# ---------------------------------------------------------------------------
# B11 — Temporary Sensor Failure + Recovery
# ---------------------------------------------------------------------------
def create_b11_sensor_recovery(seed: int = 42) -> Phase16ScenarioConfig:
    """B11: Full-horizon temporary observation blackout with complete recovery cycle.
    
    Timeline:
    - 0.0s to 2.5s: Nominal observation and tracking (pre-fault)
    - 2.5s to 5.5s: Complete sensor blackout (fault active; tracks degrade/age)
    - 5.5s to 7.5s: Observations resume, tracker re-establishes confirmed tracks (recovery)
    - 7.5s to 20.0s: Nominal navigation continues to goal (post-recovery)
    """
    obs = DynamicObstacleSpec(
        obstacle_id="obs_b11_cross",
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(2.0, -1.8, math.pi / 2),
        initial_velocity=(0.0, 0.30),
        measurement_noise_std=0.02,
    )
    faults = [
        Phase14FaultSpec(
            fault_id="fault_b11_blackout",
            fault_type=Phase14FaultType.DROPOUT,
            target_component="observation_generator",
            start_time=2.5,
            end_time=5.5,
            expected_response="TRACKS_LOST_THEN_REACQUIRED",
            parameters={"dropout_probability": 1.0},
        )
    ]
    return Phase16ScenarioConfig(
        scenario_id="B11_sensor_recovery",
        name="B11: Temporary Sensor Failure + Recovery",
        description="Full-horizon sensor failure and recovery: pre-fault -> blackout -> reacquisition -> goal arrival.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        obstacles=[obs],
        static_obstacles=[],
        fault_specs=faults,
        control_dt_s=0.10,
        max_duration_s=20.0,
        random_seed=seed,
        expected_outcome="GOAL_REACHED",
        goal_tolerance_m=0.25,
        benchmark_category="FAILURE_RECOVERY",
    )


# ---------------------------------------------------------------------------
# B12 — Compound Stress Mission
# ---------------------------------------------------------------------------
def create_b12_compound_stress(seed: int = 42) -> Phase16ScenarioConfig:
    """B12: Ultimate robustness demonstration: 3 dynamic obstacles + perception noise + temporary replanner failure."""
    obstacles = [
        DynamicObstacleSpec(
            obstacle_id="obs_b12_cross",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.5),
            initial_pose=(2.0, -1.8, math.pi / 2),
            initial_velocity=(0.0, 0.28),
            measurement_noise_std=0.03,
            confidence=0.92,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_b12_parallel",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.7, 0.5),
            initial_pose=(0.8, 1.2, 0.0),
            initial_velocity=(0.20, 0.0),
            measurement_noise_std=0.03,
            confidence=0.90,
        ),
        DynamicObstacleSpec(
            obstacle_id="obs_b12_approach",
            motion_model=MotionModelType.CONSTANT_VELOCITY,
            size=(0.8, 0.6),
            initial_pose=(4.2, 1.0, -2.5),
            initial_velocity=(-0.16, -0.08),
            measurement_noise_std=0.03,
            confidence=0.91,
        ),
    ]
    faults = [
        # Moderate detection dropout throughout
        Phase14FaultSpec(
            fault_id="fault_b12_dropout",
            fault_type=Phase14FaultType.DROPOUT,
            target_component="observation_generator",
            start_time=0.0,
            end_time=30.0,
            expected_response="TRACK_CONTINUITY_MAINTAINED",
            parameters={"dropout_probability": 0.15},
        ),
        # Temporary replanner failure event from t=2.5 to t=4.5s
        Phase14FaultSpec(
            fault_id="fault_b12_replanner",
            fault_type=Phase14FaultType.REPLANNER_NO_PATH,
            target_component="global_replanner",
            start_time=2.5,
            end_time=4.5,
            expected_response="LOCAL_DWA_COASTS_SAFELY",
            parameters={},
        ),
    ]
    return Phase16ScenarioConfig(
        scenario_id="B12_compound_stress",
        name="B12: Compound Stress Mission",
        description="Compound stress scenario: 3 dynamic obstacles + perception noise + temporary replanner failure.",
        world_bounds_m=((-5.0, 20.0), (-6.0, 6.0)),
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.5, 0.0, 0.0),
        obstacles=obstacles,
        static_obstacles=[],
        fault_specs=faults,
        control_dt_s=0.10,
        max_duration_s=22.0,
        random_seed=seed,
        expected_outcome="GOAL_REACHED",
        goal_tolerance_m=0.25,
        benchmark_category="COMPOUND_STRESS",
    )


# ---------------------------------------------------------------------------
# Registry of Benchmark Scenarios
# ---------------------------------------------------------------------------
BENCHMARK_SCENARIO_REGISTRY: Dict[str, Callable[[int], Phase16ScenarioConfig]] = {
    "B01": create_b01_flat_baseline,
    "B02": create_b02_rough_terrain,
    "B03": create_b03_static_hazards,
    "B04": create_b04_static_detour,
    "B05": create_b05_single_dynamic,
    "B06": create_b06_multi_dynamic,
    "B07": create_b07_mixed_static_dynamic,
    "B08": create_b08_dynamic_bottleneck,
    "B09": create_b09_perception_degraded,
    "B10": create_b10_localization_degraded,
    "B11": create_b11_sensor_recovery,
    "B12": create_b12_compound_stress,
}
