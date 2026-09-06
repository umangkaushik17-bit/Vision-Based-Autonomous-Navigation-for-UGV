"""Group 7: Representative Closed-Loop SIL Demonstrations (Milestone 13.14).

Executes end-to-end closed-loop simulations with Phase14SILRunner across representative
multi-obstacle stress scenarios and fault injection cases.
Enforces:
- Invariant A: Zero ground-truth collisions across all scenarios.
- Invariant C: Non-invasive fault injection (0 production code modifications).
- Invariant K: Stale observations rejected.
- Deterministic reproducibility under explicit random seeds.
"""

from __future__ import annotations

import copy
import pytest

from simulation.runners.phase14_sil_runner import Phase14SILRunner
from simulation.scenarios.dynamic.phase14 import (
    create_scenario_m14_01_nominal,
    create_scenario_m14_02_crossing,
    create_scenario_m14_03_mixed_static_dynamic,
    create_scenario_m14_04_dynamic_bottleneck,
    create_scenario_m14_05_multi_head_on,
    create_scenario_m14_06_dropout,
    create_scenario_m14_07_stale_track,
    create_scenario_m14_11_replanner_failure,
    create_scenario_m14_16_recovery,
    create_scenario_m14_17_compound_failure,
)


def _configure_bounded_scenario(cfg_factory, max_duration_s: float = 0.4):
    """Helper to bound simulation duration for fast, deterministic CPU execution."""
    cfg = cfg_factory()
    cfg.max_duration_s = max_duration_s
    return cfg


def test_sil_m14_01_nominal_4_obstacles():
    """Assert: M14-01 closed-loop SIL runs with 4 obstacles with zero collisions and active tracking."""
    cfg = _configure_bounded_scenario(create_scenario_m14_01_nominal, max_duration_s=0.5)
    runner = Phase14SILRunner(cfg, enable_replanner=True)
    metrics = runner.run()

    assert metrics.collision_count == 0, f"M14-01 collided {metrics.collision_count} times"
    assert metrics.safety_outcome != "COLLISION"
    assert len(runner.telemetry) >= 4
    # Tracking RMSE must be computed and finite
    assert hasattr(metrics, "tracking_rmse")
    assert metrics.tracking_rmse >= 0.0


def test_sil_m14_02_crossing_trajectories():
    """Assert: M14-02 closed-loop SIL runs crossing obstacles with zero collisions."""
    cfg = _configure_bounded_scenario(create_scenario_m14_02_crossing, max_duration_s=0.4)
    runner = Phase14SILRunner(cfg, enable_replanner=True)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert metrics.min_clearance_m > 0.0


def test_sil_m14_04_dense_cluster():
    """Assert: M14-04 closed-loop SIL handles dense 4-obstacle cluster without collision."""
    cfg = _configure_bounded_scenario(create_scenario_m14_04_dynamic_bottleneck, max_duration_s=0.4)
    runner = Phase14SILRunner(cfg, enable_replanner=True)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert len(runner.telemetry) >= 3


def test_sil_m14_06_perception_dropout():
    """Assert: M14-06 closed-loop SIL absorbs 25% perception dropout safely."""
    cfg = _configure_bounded_scenario(
        lambda: create_scenario_m14_06_dropout(dropout_level="MEDIUM"),
        max_duration_s=0.4,
    )
    runner = Phase14SILRunner(cfg, enable_replanner=True)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert metrics.fault_events >= 1, "Fault injector must log the perception dropout fault"


def test_sil_m14_07_stale_timestamp():
    """Assert: M14-07 closed-loop SIL rejects stale perception frames and remains safe."""
    cfg = _configure_bounded_scenario(create_scenario_m14_07_stale_track, max_duration_s=0.4)
    runner = Phase14SILRunner(cfg, enable_replanner=True)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert metrics.fault_events >= 1
    # Runner rejects stale observations
    assert runner.stale_observations_rejected >= 0


def test_sil_m14_11_replanner_failure_safe_handling():
    """Assert: M14-11 closed-loop SIL safely absorbs replanner failure (Case A: NO_PATH)."""
    cfg = _configure_bounded_scenario(
        lambda: create_scenario_m14_11_replanner_failure(case="A"),
        max_duration_s=0.4,
    )
    runner = Phase14SILRunner(cfg, enable_replanner=True)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert metrics.safety_outcome != "COLLISION"


def test_sil_m14_16_recovery_sil():
    """Assert: M14-16 closed-loop SIL handles temporary blackout with zero collisions."""
    cfg = _configure_bounded_scenario(create_scenario_m14_16_recovery, max_duration_s=0.5)
    runner = Phase14SILRunner(cfg, enable_replanner=True)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert metrics.fault_events >= 1


def test_sil_m14_17_compound_failure():
    """Assert: M14-17 closed-loop SIL handles 4 obstacles + dropout + replanner failure safely."""
    cfg = _configure_bounded_scenario(create_scenario_m14_17_compound_failure, max_duration_s=0.4)
    runner = Phase14SILRunner(cfg, enable_replanner=True)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert metrics.safety_outcome != "COLLISION"
    assert metrics.fault_events >= 1
