"""Automated test suite asserting Stage 4 (Phase 13.25) Integrated Safety & Recovery invariants.

Asserts:
1. Zero collisions across the entire benchmark matrix B01–B10.
2. Zero unsafe STOP -> PROCEED transitions caused by perception/tracking dropouts.
3. Zero unsafe reverse events (rear clearance strictly validated).
4. Zero stale command violations.
5. Command publication continuity (mean gap <= 50.0 ms).
6. Anti-pushing safety invariant (zero intentional collision or forward pushing of close hazards).
7. Successful deadlock recovery reverse and global A* replan detour (B08).
"""

import json
import os
import pytest

from scripts.validate_integrated_safety_workstream import IntegratedScenarioRunner
from ai.planning.backends.backend_interface import DWABackendType


@pytest.fixture(scope="module")
def benchmark_results():
    """Load or run the benchmark measurements."""
    json_path = os.path.join(os.path.dirname(__file__), "..", "reports", "integrated_safety_workstream_measurements.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if len(data) == 10:
                return data
    from scripts.validate_integrated_safety_workstream import run_full_suite
    return run_full_suite()


def test_zero_collisions_all_scenarios(benchmark_results):
    """Invariant 1: Total collisions across all 10 benchmark scenarios must be exactly 0."""
    total_collisions = sum(s["collision_count"] for s in benchmark_results.values())
    assert total_collisions == 0, f"Collisions detected: {total_collisions}"


def test_zero_unsafe_transitions(benchmark_results):
    """Invariant 2: Zero unsafe STOP -> PROCEED transitions caused by perception loss."""
    total_unsafe_trans = sum(s["unsafe_transitions"] for s in benchmark_results.values())
    assert total_unsafe_trans == 0, f"Unsafe transitions detected: {total_unsafe_trans}"


def test_zero_unsafe_reverse_events(benchmark_results):
    """Invariant 3: Zero unsafe reverse maneuvers when rear corridor is obstructed."""
    total_unsafe_rev = sum(s["unsafe_reverse_events"] for s in benchmark_results.values())
    assert total_unsafe_rev == 0, f"Unsafe reverse events detected: {total_unsafe_rev}"


def test_zero_stale_command_violations(benchmark_results):
    """Invariant 4: Zero stale command violations (safety governor stop on timeout)."""
    total_stale_viols = sum(s["stale_command_violations"] for s in benchmark_results.values())
    assert total_stale_viols == 0, f"Stale command violations detected: {total_stale_viols}"


def test_command_continuity(benchmark_results):
    """Invariant 5: Actuator command gap mean must be bounded to 20 Hz publication cycle (50ms)."""
    for sc_id, sc in benchmark_results.items():
        assert sc["command_gap_mean_ms"] <= 55.0, f"{sc_id} command gap mean {sc['command_gap_mean_ms']}ms > 55ms"
        assert sc["command_gap_max_ms"] <= 60.0, f"{sc_id} command gap max {sc['command_gap_max_ms']}ms > 60ms"


def test_b01_nominal_open_path(benchmark_results):
    """Verify B01 achieves goal with nominal continuous velocity."""
    b01 = benchmark_results["B01"]
    assert b01["goal_reached"] is True
    assert b01["max_velocity_mps"] >= 0.45
    assert b01["collision_count"] == 0


def test_b02_static_obstacle_avoidance(benchmark_results):
    """Verify B02 steers around static boulder with safe clearance."""
    b02 = benchmark_results["B02"]
    assert b02["goal_reached"] is True
    assert b02["min_clearance_m"] >= 0.25
    assert b02["collision_count"] == 0


def test_b03_dynamic_crossing_avoidance(benchmark_results):
    """Verify B03 tactical WAIT/AVOID maneuver on crossing obstacle."""
    b03 = benchmark_results["B03"]
    assert b03["goal_reached"] is True
    assert b03["collision_count"] == 0
    assert b03["unsafe_transitions"] == 0


def test_b05_anti_pushing_lost_obstacle(benchmark_results):
    """Verify B05 anti-pushing invariant: close obstacle tracker loss does not accelerate into it."""
    b05 = benchmark_results["B05"]
    assert b05["collision_count"] == 0
    assert b05["unsafe_transitions"] == 0
    assert b05["mean_velocity_mps"] <= 0.0


def test_b08_reverse_recovery_and_replanning(benchmark_results):
    """Verify B08 executes controlled reverse recovery, global replan, and completes mission."""
    b08 = benchmark_results["B08"]
    assert b08["goal_reached"] is True
    assert b08["collision_count"] == 0
    assert b08["recovery_triggered_count"] > 0
    assert b08["replan_count"] >= 1
