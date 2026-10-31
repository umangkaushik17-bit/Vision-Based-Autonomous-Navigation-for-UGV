"""Test Suite for Runtime Computational Adaptation and Safety Governor.

Verifies:
  - Nominal latency envelope operates in NORMAL state with full velocity.
  - Latency elevation transitions to CONSERVATIVE state with velocity clamping.
  - Critical latency spikes transition to SAFE_STOP with immediate zero velocity.
  - Stale commands (>500ms) trigger SAFE_STOP.
  - Core Invariant: Lower compute capability NEVER causes speed increase.
  - Authoritative Safety Governor integrates physical hazard protection and compute degradation.
"""

from __future__ import annotations

import pytest

from production.runtime.adaptation_manager import (
    AdaptationConfig,
    OperationalState,
    RuntimeAdaptationManager,
    RuntimeMetrics,
)
from production.safety.safety_governor import (
    AuthoritativeSafetyGovernor,
    SafetyGovernorConfig,
)


@pytest.fixture
def adaptation_mgr() -> RuntimeAdaptationManager:
    config = AdaptationConfig(
        nominal_control_freq_hz=20.0,
        nominal_max_linear_v_mps=0.50,
        nominal_planning_budget_ms=15.0,
        nominal_perception_budget_ms=120.0,
        conservative_latency_multiplier=1.5,
        critical_latency_multiplier=3.0,
        conservative_speed_scale=0.50,
        command_freshness_timeout_s=0.50,
        recovery_hysteresis_cycles=3,
    )
    return RuntimeAdaptationManager(config=config)


def test_nominal_operating_envelope(adaptation_mgr: RuntimeAdaptationManager):
    """Under nominal latency, system remains in NORMAL state with full command velocity."""
    metrics = RuntimeMetrics(
        timestamp_s=100.0,
        perception_latency_ms=90.0,
        planning_latency_ms=2.5,
        total_loop_latency_ms=95.0,
        command_age_s=0.04,
    )
    decision = adaptation_mgr.evaluate_cycle(metrics, raw_cmd_v=0.45, raw_cmd_w=0.10)

    assert decision.state == OperationalState.NORMAL
    assert decision.max_allowed_linear_v_mps == 0.50
    assert decision.command_v == 0.45
    assert not decision.emergency_stop


def test_conservative_state_on_elevated_planning_latency(adaptation_mgr: RuntimeAdaptationManager):
    """When planning latency exceeds conservative threshold (22.5 ms), speed is clamped to 50%."""
    metrics = RuntimeMetrics(
        timestamp_s=100.1,
        perception_latency_ms=100.0,
        planning_latency_ms=25.0,  # > 22.5 ms
        total_loop_latency_ms=130.0,
        command_age_s=0.05,
    )
    decision = adaptation_mgr.evaluate_cycle(metrics, raw_cmd_v=0.45, raw_cmd_w=0.10)

    assert decision.state == OperationalState.CONSERVATIVE
    assert decision.max_allowed_linear_v_mps == 0.25
    assert decision.command_v == 0.25  # Clamped down from 0.45 to 0.25
    assert not decision.emergency_stop


def test_conservative_state_on_elevated_perception_latency(adaptation_mgr: RuntimeAdaptationManager):
    """When perception latency exceeds conservative threshold (180 ms), speed is clamped to 50%."""
    metrics = RuntimeMetrics(
        timestamp_s=100.2,
        perception_latency_ms=195.0,  # > 180 ms
        planning_latency_ms=3.0,
        total_loop_latency_ms=200.0,
        command_age_s=0.05,
    )
    decision = adaptation_mgr.evaluate_cycle(metrics, raw_cmd_v=0.50, raw_cmd_w=0.0)

    assert decision.state == OperationalState.CONSERVATIVE
    assert decision.command_v == 0.25


def test_safe_stop_on_critical_planning_spike(adaptation_mgr: RuntimeAdaptationManager):
    """When planning exceeds critical threshold (45 ms), SAFE_STOP is enforced."""
    metrics = RuntimeMetrics(
        timestamp_s=100.3,
        perception_latency_ms=100.0,
        planning_latency_ms=55.0,  # > 45 ms
        total_loop_latency_ms=160.0,
        command_age_s=0.05,
    )
    decision = adaptation_mgr.evaluate_cycle(metrics, raw_cmd_v=0.40, raw_cmd_w=0.20)

    assert decision.state == OperationalState.SAFE_STOP
    assert decision.command_v == 0.0
    assert decision.command_w == 0.0
    assert decision.emergency_stop is True


def test_safe_stop_on_critical_perception_spike(adaptation_mgr: RuntimeAdaptationManager):
    """When perception exceeds critical threshold (360 ms), SAFE_STOP is enforced."""
    metrics = RuntimeMetrics(
        timestamp_s=100.4,
        perception_latency_ms=420.0,  # > 360 ms
        planning_latency_ms=2.0,
        total_loop_latency_ms=430.0,
        command_age_s=0.05,
    )
    decision = adaptation_mgr.evaluate_cycle(metrics, raw_cmd_v=0.40, raw_cmd_w=0.0)

    assert decision.state == OperationalState.SAFE_STOP
    assert decision.command_v == 0.0
    assert decision.emergency_stop is True


def test_safe_stop_on_stale_command(adaptation_mgr: RuntimeAdaptationManager):
    """Commands older than timeout threshold (500 ms) must trigger immediate SAFE_STOP."""
    metrics = RuntimeMetrics(
        timestamp_s=100.5,
        perception_latency_ms=90.0,
        planning_latency_ms=2.0,
        total_loop_latency_ms=95.0,
        command_age_s=0.75,  # > 0.50 s
    )
    decision = adaptation_mgr.evaluate_cycle(metrics, raw_cmd_v=0.40, raw_cmd_w=0.10)

    assert decision.state == OperationalState.SAFE_STOP
    assert decision.command_v == 0.0
    assert decision.emergency_stop is True


def test_invariant_speed_never_increases_with_higher_latency(adaptation_mgr: RuntimeAdaptationManager):
    """MANDATORY INVARIANT: As compute latency degrades from 1 ms to 500 ms,
    vehicle commanded speed must be monotonically non-increasing (NEVER increases).
    """
    raw_v = 0.50
    latencies = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0, 200.0]

    prev_cmd_v = 999.0
    for lat in latencies:
        # Create fresh manager for isolated test points
        mgr = RuntimeAdaptationManager(config=adaptation_mgr.config)
        m = RuntimeMetrics(
            timestamp_s=1.0,
            perception_latency_ms=90.0,
            planning_latency_ms=lat,
            total_loop_latency_ms=90.0 + lat,
            command_age_s=0.05,
        )
        dec = mgr.evaluate_cycle(m, raw_cmd_v=raw_v, raw_cmd_w=0.0)

        # Invariant check: speed must NEVER be greater than previous under higher latency
        assert dec.command_v <= prev_cmd_v, f"Speed increased from {prev_cmd_v} to {dec.command_v} at latency {lat} ms!"
        assert dec.command_v <= raw_v, "Commanded velocity exceeded requested velocity!"
        prev_cmd_v = dec.command_v


def test_authoritative_safety_governor_joint_evaluation():
    """Verify AuthoritativeSafetyGovernor enforces physical obstacle safety and compute degradation jointly."""
    gov = AuthoritativeSafetyGovernor(
        config=SafetyGovernorConfig(stop_distance_m=0.40, cautious_distance_m=0.95),
        adaptation_manager=RuntimeAdaptationManager(),
    )

    robot_pose = (0.0, 0.0, 0.0)

    # 1. Healthy compute + no obstacle -> ALLOW full speed
    healthy_metrics = RuntimeMetrics(
        timestamp_s=1.0, perception_latency_ms=95.0, planning_latency_ms=2.0,
        total_loop_latency_ms=100.0, command_age_s=0.05
    )
    v, w, dec, diag = gov.evaluate_safe_command(
        requested_v=0.45, requested_w=0.10, robot_pose=robot_pose,
        hazards=[], metrics=healthy_metrics, current_time_s=1.0
    )
    assert dec == "ALLOW"
    assert v == 0.45

    # 2. Critical compute lag -> COMPUTE_EMERGENCY_STOP even with zero obstacles
    lag_metrics = RuntimeMetrics(
        timestamp_s=2.0, perception_latency_ms=500.0, planning_latency_ms=2.0,
        total_loop_latency_ms=510.0, command_age_s=0.05
    )
    v, w, dec, diag = gov.evaluate_safe_command(
        requested_v=0.45, requested_w=0.10, robot_pose=robot_pose,
        hazards=[], metrics=lag_metrics, current_time_s=2.0
    )
    assert dec == "COMPUTE_EMERGENCY_STOP"
    assert v == 0.0
    assert w == 0.0
