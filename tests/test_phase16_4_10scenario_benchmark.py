"""Automated tests for Subphase 16.4: 10-Scenario SIH Grand Finale Comprehensive Benchmark Suite.

Verifies:
1. `run_phase16_sih_grand_finale_benchmark` (`benchmarks/run_phase16_sih_grand_finale_benchmark.py`)
   executes all 10 canonical SIH outdoor scenarios (`S01` through `S10`).
2. 10/10 scenarios PASS (`mission_success_rate_pct == 100.0%`) with:
   - `total_collisions == 0`
   - `total_rollover_violations == 0`
   - `total_high_centering_events == 0`
   - `mean_control_loop_ms < 15.0 ms`
   - `max_ekf_drift_pct < 0.05%`
3. `reports/phase16_grand_finale_benchmark.json` is written and contains complete per-scenario telemetry.
"""

from __future__ import annotations

import json
import os
import pytest

from benchmarks.run_phase16_sih_grand_finale_benchmark import (
    CANONICAL_SCENARIOS,
    run_phase16_sih_grand_finale_benchmark,
)


def test_10_scenario_sih_grand_finale_benchmark_100pct_pass():
    """Execute the 10-scenario SIH Grand Finale benchmark suite and verify 10/10 PASS and telemetry limits."""
    summary = run_phase16_sih_grand_finale_benchmark(save_outputs=True)

    assert summary["scenarios_total"] == 10
    assert summary["scenarios_passed"] == 10
    assert summary["mission_success_rate_pct"] == pytest.approx(100.0)
    assert summary["total_collisions"] == 0
    assert summary["total_rollover_violations"] == 0
    assert summary["total_high_centering_events"] == 0
    assert summary["mean_control_loop_ms"] < 15.0
    assert summary["max_ekf_drift_pct"] < 0.05
    assert summary["min_rollover_margin"] >= 0.20

    expected_ids = [s[0] for s in CANONICAL_SCENARIOS]
    actual_ids = [r["scenario_id"] for r in summary["scenarios"]]
    assert actual_ids == expected_ids

    for r in summary["scenarios"]:
        assert r["mission_success"] is True, f"Scenario {r['scenario_id']} failed: {r}"
        assert r["collisions"] == 0
        assert r["rollover_violations"] == 0
        assert r["high_centering_events"] == 0

    s10 = [r for r in summary["scenarios"] if r["scenario_id"] == "S10_FULL_100M_5WP_GRAND_FINALE"][0]
    assert s10["waypoints_completed"] == "5/5"

    json_path = os.path.join("reports", "phase16_grand_finale_benchmark.json")
    assert os.path.exists(json_path), "Benchmark JSON artifact must be written to reports/phase16_grand_finale_benchmark.json"
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["scenarios_passed"] == 10
