"""Automated tests for Subphase 16.2: Judge-Facing Interactive RViz2 & 2.5D Telemetry Dashboard.

Verifies:
1. `JudgeDashboardTelemetryAdapter` in `simulation/nodes/presentation_visualization_adapter.py`
   generates the 3-marker HUD array (`judge_attitude_stability_hud`, `judge_rollover_margin_bar`,
   `judge_system_performance_hud`) containing `Roll`, `Pitch`, `RolloverMargin`, `BellyClearance`,
   `EKFDrift`, and `State`.
2. Color transitions of the 3D Rollover Stability Margin Bar across:
   - Safe state (Green: M_roll >= 0.45)
   - Cautious state (Amber: 0.20 <= M_roll < 0.45)
   - Critical state (Red: M_roll < 0.20)
3. `simulation/rviz/hero_presentation.rviz` configures `/telemetry/judge_dashboard`.
4. Dashboard marker generation executes in `< 1.5 ms` on CPU.
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pytest
import yaml

from simulation.nodes.presentation_visualization_adapter import JudgeDashboardTelemetryAdapter


def test_judge_dashboard_marker_generation_and_hud_fields():
    """Verify JudgeDashboardTelemetryAdapter outputs all required 2.5D attitude, rollover, and performance fields."""
    m_arr = JudgeDashboardTelemetryAdapter.generate_dashboard_markers(
        robot_pose=(30.0, 6.2, 0.15),
        roll_deg=3.5,
        pitch_deg=10.0,
        elevation_m=0.42,
        rollover_margin=0.62,
        belly_clearance_m=0.115,
        planner_backend="C++17 AVX2",
        astar_latency_ms=1.53,
        dwa_latency_ms=1.32,
        ekf_confidence=0.96,
        ekf_drift_pct=0.01,
        behavioral_state="PROCEED",
        frame_id="odom",
    )

    assert len(m_arr.markers) == 3
    by_ns = {m.ns: m for m in m_arr.markers}
    assert set(by_ns.keys()) == {
        "judge_attitude_stability_hud",
        "judge_rollover_margin_bar",
        "judge_system_performance_hud",
    }

    att_text = by_ns["judge_attitude_stability_hud"].text
    assert "Roll=+3.5deg" in att_text
    assert "Pitch=+10.0deg" in att_text
    assert "Z=+0.42m" in att_text
    assert "RolloverMargin=0.62" in att_text
    assert "BellyClearance=0.115m" in att_text

    perf_text = by_ns["judge_system_performance_hud"].text
    assert "Backend=C++17 AVX2" in perf_text
    assert "A*=1.53ms" in perf_text
    assert "DWA=1.32ms" in perf_text
    assert "EKFDrift=0.01%" in perf_text
    assert "State=PROCEED" in perf_text


def test_rollover_margin_bar_color_transitions_safe_cautious_critical():
    """Verify Green (>= 0.45), Amber (0.20..0.45), and Red (< 0.20) rollover bar color transitions."""
    # 1. Safe (M_roll = 0.75 >= 0.45) -> Green
    m_safe = JudgeDashboardTelemetryAdapter.generate_dashboard_markers(rollover_margin=0.75)
    bar_safe = [m for m in m_safe.markers if m.ns == "judge_rollover_margin_bar"][0]
    assert bar_safe.color.g > 0.80 and bar_safe.color.r < 0.30

    # 2. Cautious (M_roll = 0.30 in [0.20, 0.45)) -> Amber
    m_caut = JudgeDashboardTelemetryAdapter.generate_dashboard_markers(rollover_margin=0.30)
    bar_caut = [m for m in m_caut.markers if m.ns == "judge_rollover_margin_bar"][0]
    assert bar_caut.color.r > 0.85 and 0.55 <= bar_caut.color.g <= 0.85 and bar_caut.color.b < 0.30

    # 3. Critical (M_roll = 0.12 < 0.20) -> Red
    m_crit = JudgeDashboardTelemetryAdapter.generate_dashboard_markers(rollover_margin=0.12)
    bar_crit = [m for m in m_crit.markers if m.ns == "judge_rollover_margin_bar"][0]
    assert bar_crit.color.r > 0.85 and bar_crit.color.g < 0.30


def test_hero_rviz_includes_judge_dashboard_topic():
    """Verify hero_presentation.rviz configures /telemetry/judge_dashboard."""
    rviz_path = Path("simulation/rviz/hero_presentation.rviz")
    with open(rviz_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    displays = data["Visualization Manager"]["Displays"]
    topics = {
        d.get("Topic", {}).get("Value")
        for d in displays
        if isinstance(d.get("Topic"), dict)
    }
    assert "/telemetry/judge_dashboard" in topics


def test_judge_dashboard_execution_latency_under_1_5ms():
    """Verify JudgeDashboardTelemetryAdapter.generate_dashboard_markers executes in < 1.5 ms."""
    for _ in range(5):
        JudgeDashboardTelemetryAdapter.generate_dashboard_markers(
            robot_pose=(15.0, 2.0, 0.1), roll_deg=8.0, pitch_deg=6.0, linear_v_mps=0.50, angular_w_radps=0.20
        )

    times_ms = []
    for _ in range(50):
        t0 = time.perf_counter()
        JudgeDashboardTelemetryAdapter.generate_dashboard_markers(
            robot_pose=(15.0, 2.0, 0.1), roll_deg=8.0, pitch_deg=6.0, linear_v_mps=0.50, angular_w_radps=0.20
        )
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))
    print(f"\n[PHASE 16.2 BENCHMARK] JudgeDashboardTelemetryAdapter: mean={mean_ms:.3f} ms, P95={p95_ms:.3f} ms")
    assert mean_ms < 1.5, f"Expected < 1.5 ms execution time, got {mean_ms:.3f} ms"
