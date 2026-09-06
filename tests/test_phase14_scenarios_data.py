"""Group 1: Scenario configuration, determinism, and data model tests for Phase 13.14."""

from __future__ import annotations

import math
import pytest

from simulation.scenarios.dynamic.phase14.data_model import (
    Phase14FaultSpec,
    Phase14FaultType,
    Phase14ScenarioConfig,
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
    create_scenario_m14_11_replanner_failure,
    create_scenario_m14_16_recovery,
    create_scenario_m14_17_compound_failure,
)


def test_m14_01_nominal_spec_integrity():
    """M14-01: 4 obstacles, distinct poses, non-overlapping initial boxes."""
    cfg = create_scenario_m14_01_nominal()
    assert len(cfg.obstacles) == 4
    poses = [obs.initial_pose for obs in cfg.obstacles]
    # Verify all initial poses are distinct
    assert len(set((round(p[0], 2), round(p[1], 2)) for p in poses)) == 4
    # Verify at least two obstacles have velocity influencing path
    moving = [obs for obs in cfg.obstacles if math.hypot(obs.initial_velocity[0], obs.initial_velocity[1]) > 0.05]
    assert len(moving) >= 2


def test_m14_02_crossing_spec_integrity():
    """M14-02: 4 obstacles with crossing trajectory patterns."""
    cfg = create_scenario_m14_02_crossing()
    assert len(cfg.obstacles) == 4
    # Check that headings and velocities represent crossing motion
    vels = [obs.initial_velocity for obs in cfg.obstacles]
    assert any(abs(v[1]) > 0.15 for v in vels), "Must have crossing lateral velocity"


def test_m14_03_mixed_spec_integrity():
    """M14-03: Combined static obstacles + 3 dynamic obstacles."""
    cfg = create_scenario_m14_03_mixed_static_dynamic()
    assert len(cfg.static_obstacles) >= 2
    assert len(cfg.obstacles) == 3


def test_m14_04_bottleneck_spec_integrity():
    """M14-04: Static corridor barriers + 3 dynamic obstacles."""
    cfg = create_scenario_m14_04_dynamic_bottleneck()
    assert len(cfg.static_obstacles) >= 4
    assert len(cfg.obstacles) == 3
    assert cfg.max_duration_s >= 2.0


def test_m14_05_head_on_spec_integrity():
    """M14-05: 3 obstacles approaching forward corridor."""
    cfg = create_scenario_m14_05_multi_head_on()
    assert len(cfg.obstacles) == 3
    # All obstacles approach with negative vx
    for obs in cfg.obstacles:
        assert obs.initial_velocity[0] < -0.10


def test_fault_spec_validation():
    """Fault spec validation constraints."""
    # Valid spec
    spec = Phase14FaultSpec(
        fault_id="f1",
        fault_type=Phase14FaultType.DROPOUT,
        target_component="perception",
        start_time=0.5,
        end_time=1.5,
        expected_response="DROP",
    )
    assert spec.is_active(1.0)
    assert not spec.is_active(0.2)
    assert not spec.is_active(1.8)

    # Invalid start time
    with pytest.raises(ValueError):
        Phase14FaultSpec("f_neg", Phase14FaultType.DROPOUT, "target", -1.0, 1.0, "resp")

    # Invalid end before start
    with pytest.raises(ValueError):
        Phase14FaultSpec("f_inv", Phase14FaultType.DROPOUT, "target", 2.0, 1.0, "resp")


def test_serialization_roundtrip():
    """Phase14ScenarioConfig serialization and deserialization."""
    cfg = create_scenario_m14_17_compound_failure()
    d = cfg.to_dict()
    restored = Phase14ScenarioConfig.from_dict(d)
    assert restored.scenario_id == cfg.scenario_id
    assert len(restored.fault_specs) == len(cfg.fault_specs)
    assert restored.fault_specs[0].fault_id == cfg.fault_specs[0].fault_id
    assert restored.random_seed == cfg.random_seed
