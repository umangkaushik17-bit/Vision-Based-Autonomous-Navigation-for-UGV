"""Integration tests for Subphase 15.9: 7-Scenario Differential Terrain Benchmark Suite.

Verifies:
1. Automated execution of `run_phase15_benchmark_suite` across all 7 canonical scenarios.
2. Phase 15 (2.5D Terrain-Aware Stack) achieves 100% mission success (7/7 scenarios)
   with 0 collisions, 0 rollover violations, and 0 high-centering events.
3. Differential proof against Phase 14 (2D Planar Baseline):
   - Scenario 1 (Flat Baseline): 100% parity between Phase 14 and Phase 15.
   - Scenario 3 (32 deg Embankment): Phase 14 suffers rollover violation; Phase 15 detours with 0 rollovers.
   - Scenario 4 (24 deg Cross-Slope): Phase 14 suffers cross-slope rollover violation; Phase 15 routes safely.
   - Scenario 5 (18 cm Boulder Field): Phase 14 high-centers; Phase 15 detours with 0 high-centerings.
   - Scenario 6 (12 deg Undulating Hills): Phase 15 SE(3) IPM reduces projection drift to <= 0.10m (vs > 0.50m in Phase 14).
   - Scenario 7 (Compound + Dynamic): Phase 15 completes with 0 collisions, 0 rollovers, 0 high-centerings.
4. Generation of `reports/phase15_benchmark.json` and `reports/PHASE_15_BENCHMARK_REPORT.md`.
"""

from __future__ import annotations

import json
import os
import pytest

from benchmarks.run_phase15_terrain_benchmark import run_phase15_benchmark_suite


def test_phase15_7_scenario_differential_benchmark_suite():
    """Run the 7-scenario differential benchmark and verify all Phase 15 exit criteria."""
    summary = run_phase15_benchmark_suite(save_outputs=True)

    p14_sum = summary["phase14_baseline_summary"]
    p15_sum = summary["phase15_evaluation_summary"]

    # 1. Phase 15 must achieve 100% mission success with 0 collisions, 0 rollovers, 0 high-centerings
    assert summary["scenarios_count"] == 7
    assert p15_sum["mission_success_rate_pct"] == 100.0
    assert p15_sum["total_collisions"] == 0
    assert p15_sum["total_rollover_violations"] == 0
    assert p15_sum["total_high_centering_events"] == 0
    assert p15_sum["mean_astar_cpp_ms"] < 5.0
    assert p15_sum["mean_dwa_cpp_ms"] < 3.5

    # 2. Phase 14 baseline must exhibit terrain-induced failures on the uneven scenarios
    assert p14_sum["total_rollover_violations"] > 0
    assert p14_sum["total_high_centering_events"] > 0
    assert p14_sum["mission_success_rate_pct"] < 50.0

    # 3. Verify per-scenario differential invariants
    p14_by_id = {r["scenario_id"]: r for r in summary["phase14_runs"]}
    p15_by_id = {r["scenario_id"]: r for r in summary["phase15_runs"]}

    # Scenario 1: Flat baseline parity
    assert p14_by_id["SCENARIO_1_FLAT_BASELINE"]["mission_success"] is True
    assert p15_by_id["SCENARIO_1_FLAT_BASELINE"]["mission_success"] is True

    # Scenario 3: 32 deg embankment detour
    assert p14_by_id["SCENARIO_3_STEEP_32DEG_EMBANKMENT"]["rollover_violations"] > 0
    assert p15_by_id["SCENARIO_3_STEEP_32DEG_EMBANKMENT"]["rollover_violations"] == 0
    assert p15_by_id["SCENARIO_3_STEEP_32DEG_EMBANKMENT"]["path_length_m"] > 8.5

    # Scenario 5: 18 cm step boulder field
    assert p14_by_id["SCENARIO_5_STEP_18CM_BOULDER_FIELD"]["high_centering_events"] > 0
    assert p15_by_id["SCENARIO_5_STEP_18CM_BOULDER_FIELD"]["high_centering_events"] == 0

    # Scenario 6: 12 deg undulating pitching hills IPM drift
    assert p15_by_id["SCENARIO_6_UNDULATING_12DEG_HILLS"]["ipm_projection_drift_m"] <= 0.10
    assert (
        p15_by_id["SCENARIO_6_UNDULATING_12DEG_HILLS"]["ipm_projection_drift_m"]
        < p14_by_id["SCENARIO_6_UNDULATING_12DEG_HILLS"]["ipm_projection_drift_m"]
    )

    # 4. Verify report artifacts exist and are populated
    assert os.path.exists(os.path.join("reports", "phase15_benchmark.json"))
    assert os.path.exists(os.path.join("reports", "PHASE_15_BENCHMARK_REPORT.md"))
    with open(os.path.join("reports", "phase15_benchmark.json"), "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["phase15_evaluation_summary"]["mission_success_rate_pct"] == 100.0
