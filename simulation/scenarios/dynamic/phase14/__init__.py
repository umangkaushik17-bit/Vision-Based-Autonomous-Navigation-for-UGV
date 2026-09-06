"""Phase 13.14 Multi-Obstacle and Failure Injection Scenarios."""

from simulation.scenarios.dynamic.phase14.data_model import (
    Phase14FaultSpec,
    Phase14FaultType,
    Phase14ScenarioConfig,
)
from simulation.scenarios.dynamic.phase14.failure_models import (
    Phase14FaultInjector,
)
from simulation.scenarios.dynamic.phase14.multi_obstacle_scenarios import (
    create_scenario_m14_01_nominal,
    create_scenario_m14_02_crossing,
    create_scenario_m14_03_mixed_static_dynamic,
    create_scenario_m14_04_dynamic_bottleneck,
    create_scenario_m14_05_multi_head_on,
)
from simulation.scenarios.dynamic.phase14.scenario_builders import (
    create_scenario_m14_06_dropout,
    create_scenario_m14_07_stale_track,
    create_scenario_m14_08_noisy_multi_object,
    create_scenario_m14_09_planner_saturation,
    create_scenario_m14_10_zero_valid_trajectories,
    create_scenario_m14_11_replanner_failure,
    create_scenario_m14_12_replanner_timeout,
    create_scenario_m14_13_tracker_degradation,
    create_scenario_m14_14_prediction_degradation,
    create_scenario_m14_15_invalid_planner_output,
    create_scenario_m14_16_recovery,
    create_scenario_m14_17_compound_failure,
)

__all__ = [
    "Phase14FaultType",
    "Phase14FaultSpec",
    "Phase14ScenarioConfig",
    "Phase14FaultInjector",
    "create_scenario_m14_01_nominal",
    "create_scenario_m14_02_crossing",
    "create_scenario_m14_03_mixed_static_dynamic",
    "create_scenario_m14_04_dynamic_bottleneck",
    "create_scenario_m14_05_multi_head_on",
    "create_scenario_m14_06_dropout",
    "create_scenario_m14_07_stale_track",
    "create_scenario_m14_08_noisy_multi_object",
    "create_scenario_m14_09_planner_saturation",
    "create_scenario_m14_10_zero_valid_trajectories",
    "create_scenario_m14_11_replanner_failure",
    "create_scenario_m14_12_replanner_timeout",
    "create_scenario_m14_13_tracker_degradation",
    "create_scenario_m14_14_prediction_degradation",
    "create_scenario_m14_15_invalid_planner_output",
    "create_scenario_m14_16_recovery",
    "create_scenario_m14_17_compound_failure",
]
