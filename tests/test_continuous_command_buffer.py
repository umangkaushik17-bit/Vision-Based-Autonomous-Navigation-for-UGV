"""Unit tests for SafeCommandBuffer and continuous high-frequency command dispatching (Phase 13.22).

Validates:
1. Planner slower than command publisher (4 Hz planner vs 20 Hz publisher).
2. Planner temporary delay (decay phase smoothly ramps velocity down).
3. Command freshness window.
4. Hard stale command timeout.
5. Immediate safety stop override.
6. Continuous publication gap statistics.
7. Reset / shutdown state.
8. Zero command handling.
"""

from __future__ import annotations

import time
import pytest
import numpy as np

from ai.control.command_buffer import DispatchStatus, SafeCommandBuffer, SafeCommandBufferConfig


def test_command_freshness_nominal():
    """Assert: While command age <= freshness_timeout, dispatched command is FRESH with full amplitude."""
    cfg = SafeCommandBufferConfig(publish_rate_hz=20.0, freshness_timeout_s=0.35, stale_timeout_s=0.50)
    buf = SafeCommandBuffer(cfg)

    t0 = 100.0
    buf.set_command(linear_v=0.50, angular_w=0.10, timestamp_s=t0, is_safe=True)

    # Dispatched at t0 + 0.10s (age = 0.10s <= 0.35s)
    v, w, status, diag = buf.get_dispatch_command(current_time_s=t0 + 0.10)
    assert status == DispatchStatus.FRESH
    assert pytest.approx(v, abs=1e-3) == 0.50
    assert pytest.approx(w, abs=1e-3) == 0.10
    assert diag["age_ms"] == pytest.approx(100.0, abs=1.0)


def test_planner_slower_than_publisher():
    """Assert: Publisher running at 20 Hz (50ms) smoothly outputs fresh commands from a 4 Hz (250ms) planner."""
    cfg = SafeCommandBufferConfig(publish_rate_hz=20.0, freshness_timeout_s=0.35, stale_timeout_s=0.50)
    buf = SafeCommandBuffer(cfg)

    # Simulate 1 second of operation: planner updates at 4 Hz (t = 0.0, 0.25, 0.50, 0.75)
    # publisher ticks at 20 Hz (t = 0.00, 0.05, 0.10, ...)
    dispatched_velocities = []
    dispatched_statuses = []

    for pub_step in range(20):
        t_pub = round(pub_step * 0.05, 3)

        # Check if planner ran at this timestamp or prior
        if pub_step % 5 == 0:  # 0, 5, 10, 15 -> every 0.25s
            buf.set_command(linear_v=0.60, angular_w=0.0, timestamp_s=t_pub, is_safe=True)

        v, w, status, _ = buf.get_dispatch_command(current_time_s=t_pub)
        dispatched_velocities.append(v)
        dispatched_statuses.append(status)

    # All 20 ticks must be FRESH, because 250ms < 350ms freshness timeout
    assert all(st == DispatchStatus.FRESH for st in dispatched_statuses)
    assert all(pytest.approx(v, abs=1e-3) == 0.60 for v in dispatched_velocities)

    metrics = buf.get_metrics()
    assert metrics["fresh_count"] == 20
    assert metrics["decay_count"] == 0
    assert metrics["stale_count"] == 0


def test_planner_temporary_delay_decay():
    """Assert: If planner experiences a temporary 450ms delay, command decays smoothly toward zero."""
    cfg = SafeCommandBufferConfig(publish_rate_hz=20.0, freshness_timeout_s=0.30, stale_timeout_s=0.50)
    buf = SafeCommandBuffer(cfg)

    t0 = 10.0
    buf.set_command(linear_v=0.60, angular_w=0.20, timestamp_s=t0, is_safe=True)

    # At t0 + 0.40s (age = 0.40s, between 0.30s and 0.50s): decaying
    v, w, status, diag = buf.get_dispatch_command(current_time_s=t0 + 0.40)
    assert status == DispatchStatus.DECAYING
    # decay fraction = (0.50 - 0.40) / (0.50 - 0.30) = 0.10 / 0.20 = 0.50
    assert pytest.approx(v, abs=1e-2) == 0.30
    assert pytest.approx(w, abs=1e-2) == 0.10

    # At t0 + 0.45s: further decay
    v, w, status, diag = buf.get_dispatch_command(current_time_s=t0 + 0.45)
    assert status == DispatchStatus.DECAYING
    assert v < 0.30

    # When fresh planner result arrives at t0 + 0.46s, recovers immediately
    buf.set_command(linear_v=0.55, angular_w=0.0, timestamp_s=t0 + 0.46, is_safe=True)
    v_rec, _, status_rec, _ = buf.get_dispatch_command(current_time_s=t0 + 0.47)
    assert status_rec == DispatchStatus.FRESH
    assert pytest.approx(v_rec, abs=1e-3) == 0.55


def test_stale_command_timeout():
    """Assert: When command age exceeds stale_timeout, output is strictly 0.0 with STALE_STOP."""
    cfg = SafeCommandBufferConfig(publish_rate_hz=20.0, freshness_timeout_s=0.30, stale_timeout_s=0.50)
    buf = SafeCommandBuffer(cfg)

    t0 = 50.0
    buf.set_command(linear_v=0.80, angular_w=0.30, timestamp_s=t0, is_safe=True)

    # Dispatched at t0 + 0.55s (> 0.50s)
    v, w, status, diag = buf.get_dispatch_command(current_time_s=t0 + 0.55)
    assert status == DispatchStatus.STALE_STOP
    assert v == 0.0
    assert w == 0.0


def test_immediate_safety_stop():
    """Assert: emergency_stop() or is_safe=False immediately clamps output to 0.0 regardless of age."""
    cfg = SafeCommandBufferConfig()
    buf = SafeCommandBuffer(cfg)

    t0 = 10.0
    buf.set_command(linear_v=0.60, angular_w=0.20, timestamp_s=t0, is_safe=True)

    # Immediate emergency stop at t0 + 0.05s
    buf.emergency_stop(timestamp_s=t0 + 0.05, reason="Lethal obstacle proximity")

    v, w, status, diag = buf.get_dispatch_command(current_time_s=t0 + 0.06)
    assert status == DispatchStatus.SAFETY_STOP
    assert v == 0.0
    assert w == 0.0
    assert diag["reason"] == "Lethal obstacle proximity"


def test_zero_command_behavior():
    """Assert: Intentionally commanding 0.0 linear and angular produces ZERO_CMD status."""
    cfg = SafeCommandBufferConfig()
    buf = SafeCommandBuffer(cfg)

    buf.set_command(linear_v=0.0, angular_w=0.0, timestamp_s=20.0, is_safe=True)
    v, w, status, _ = buf.get_dispatch_command(current_time_s=20.05)
    assert status == DispatchStatus.ZERO_CMD
    assert v == 0.0
    assert w == 0.0


def test_dispatch_gap_statistics():
    """Assert: Command buffer measures gap distribution and percentiles."""
    cfg = SafeCommandBufferConfig(publish_rate_hz=20.0)
    buf = SafeCommandBuffer(cfg)

    t_now = 0.0
    buf.set_command(linear_v=0.5, angular_w=0.0, timestamp_s=0.0, is_safe=True)

    for i in range(50):
        t_now += 0.050  # exactly 50 ms
        buf.get_dispatch_command(current_time_s=t_now)

    metrics = buf.get_metrics()
    assert metrics["total_dispatches"] == 50
    assert pytest.approx(metrics["gap_mean_ms"], abs=1.0) == 50.0
    assert pytest.approx(metrics["gap_p50_ms"], abs=1.0) == 50.0
    assert pytest.approx(metrics["gap_p95_ms"], abs=1.0) == 50.0


def test_reset_functionality():
    """Assert: reset() clears metrics and commands completely."""
    cfg = SafeCommandBufferConfig()
    buf = SafeCommandBuffer(cfg)

    buf.set_command(linear_v=0.5, angular_w=0.0, timestamp_s=1.0, is_safe=True)
    buf.get_dispatch_command(current_time_s=1.05)
    assert buf.get_metrics()["total_dispatches"] == 1

    buf.reset()
    assert buf.get_metrics()["total_dispatches"] == 0
    v, w, status, _ = buf.get_dispatch_command(current_time_s=2.0)
    assert v == 0.0
    assert w == 0.0
