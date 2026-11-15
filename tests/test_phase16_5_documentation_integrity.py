"""Automated tests for Subphase 16.5: Final SIH Judge Dossier, Architecture Freeze & Roadmap Closure.

Verifies:
1. All Phase 15 and Phase 16 benchmark reports and machine-readable JSON telemetry files
   exist and are internally consistent (`7/7` Phase 15 scenarios PASS, `10/10` Phase 16 scenarios PASS).
2. `reports/PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md` contains the complete 10-scenario metrics table,
   Phase 0-16 cumulative verification matrix, and 1-command Judge Quick-Start Runbook.
3. All required Gazebo SDF worlds (`outdoor_presentation_100m.sdf`, `outdoor_uneven_terrain_100m.sdf`),
   launch scripts, and compiled C++17 shared libraries (`astar_cpp` and `dwa_cpp` `.dll`/`.so`) exist.
4. Top-level documentation (`docs/ROADMAP.md`, `README.md`, `docs/PHASE_15_MASTER_IMPLEMENTATION_PLAN.md`,
   `docs/PHASE_16_MASTER_IMPLEMENTATION_PLAN.md`) marks Phases 14, 15, and 16 as COMPLETE & VERIFIED.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest


def test_phase15_and_phase16_reports_and_json_telemetry_consistency():
    """Verify Phase 15 and Phase 16 JSON telemetry and Markdown reports exist and are internally consistent."""
    p15_json_path = Path("reports/phase15_benchmark.json")
    p15_md_path = Path("reports/PHASE_15_BENCHMARK_REPORT.md")
    p16_json_path = Path("reports/phase16_grand_finale_benchmark.json")
    p16_md_path = Path("reports/PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md")

    for p in (p15_json_path, p15_md_path, p16_json_path, p16_md_path):
        assert p.exists(), f"Missing required benchmark artifact: {p}"

    with open(p15_json_path, "r", encoding="utf-8") as f:
        p15_data = json.load(f)
    assert p15_data["scenarios_count"] == 7
    assert p15_data["phase15_evaluation_summary"]["mission_success_rate_pct"] == pytest.approx(100.0)
    assert p15_data["phase15_evaluation_summary"]["total_rollover_violations"] == 0
    assert p15_data["phase15_evaluation_summary"]["total_high_centering_events"] == 0

    with open(p16_json_path, "r", encoding="utf-8") as f:
        p16_data = json.load(f)
    assert p16_data["scenarios_total"] == 10
    assert p16_data["scenarios_passed"] == 10
    assert p16_data["mission_success_rate_pct"] == pytest.approx(100.0)
    assert p16_data["total_collisions"] == 0
    assert p16_data["total_rollover_violations"] == 0
    assert p16_data["total_high_centering_events"] == 0

    p16_md_text = p16_md_path.read_text(encoding="utf-8")
    for scen in p16_data["scenarios"]:
        assert scen["scenario_id"] in p16_md_text, f"Missing {scen['scenario_id']} in PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md"


def test_judge_quick_start_runbook_and_cumulative_matrix_in_final_report():
    """Verify final SIH benchmark report includes the Phase 0-16 matrix and 1-command Judge Runbook."""
    report_text = Path("reports/PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md").read_text(encoding="utf-8")

    # Verify all phases 0 through 16 are listed in the cumulative verification matrix
    for phase_num in range(17):
        assert f"Phase {phase_num}" in report_text, f"Missing Phase {phase_num} in cumulative matrix"

    # Verify 1-command judge launchers are documented
    assert "prototype_presentation.launch.py" in report_text
    assert "launch_presentation_wasd.bat" in report_text
    assert "scripts/run_live_camera_hil_demo.py" in report_text
    assert "pytest" in report_text


def test_sdf_worlds_cpp_shared_libraries_and_demo_scripts_exist():
    """Verify SDF worlds, C++17 native planner shared libraries, and demo runners exist on disk."""
    required_files = [
        Path("simulation/worlds/outdoor_presentation_100m.sdf"),
        Path("simulation/worlds/outdoor_uneven_terrain_100m.sdf"),
        Path("simulation/launch/prototype_presentation.launch.py"),
        Path("simulation/nodes/interactive_ugv_navigation_node.py"),
        Path("simulation/nodes/presentation_visualization_adapter.py"),
        Path("production/perception/photometric_resilience.py"),
        Path("production/perception/hil_camera_bridge.py"),
        Path("scripts/run_live_camera_hil_demo.py"),
        Path("benchmarks/run_phase15_terrain_benchmark.py"),
        Path("benchmarks/run_phase16_sih_grand_finale_benchmark.py"),
    ]
    for rf in required_files:
        assert rf.exists(), f"Missing required Phase 15/16 artifact: {rf}"

    # Verify compiled C++17 A* and DWA shared libraries (.dll on Windows or .so on Linux)
    cpp_dir = Path("ai/planning/native_cpp")
    has_astar_bin = (cpp_dir / "astar_cpp.dll").exists() or (cpp_dir / "libastar_cpp.so").exists()
    has_dwa_bin = (cpp_dir / "dwa_cpp.dll").exists() or (cpp_dir / "libdwa_cpp.so").exists()
    assert has_astar_bin, "Missing compiled C++17 A* shared library in ai/planning/native_cpp/"
    assert has_dwa_bin, "Missing compiled C++17 DWA shared library in ai/planning/native_cpp/"


def test_roadmap_and_readme_closure_status():
    """Verify docs/ROADMAP.md, README.md, and docs/PHASE_16_MASTER_IMPLEMENTATION_PLAN.md reflect COMPLETE & VERIFIED status."""
    roadmap_text = Path("docs/ROADMAP.md").read_text(encoding="utf-8")
    readme_text = Path("README.md").read_text(encoding="utf-8")
    p16_plan_text = Path("docs/PHASE_16_MASTER_IMPLEMENTATION_PLAN.md").read_text(encoding="utf-8")

    assert "Phase 14: GPS-Denied Outdoor Navigation & Mission Executive (COMPLETE & VERIFIED)" in roadmap_text
    assert "Phase 15: Advanced 2.5D Terrain-Aware Navigation (COMPLETE & VERIFIED)" in roadmap_text
    assert "Phase 16: Full Benchmarking and SIH Demonstration (COMPLETE & VERIFIED)" in roadmap_text
    assert "PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md" in roadmap_text

    assert "PHASE_15_BENCHMARK_REPORT.md" in readme_text
    assert "PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md" in readme_text
    assert "COMPLETE & VERIFIED" in readme_text

    assert "SUBPHASES 16.0–16.5 COMPLETE & VERIFIED" in p16_plan_text
