#!/usr/bin/env python3
"""Reusable Authoritative CPU Regression Runner.

Executes the full pytest suite, captures runtime and pass/fail/skip counts,
categorizes all failures into the 7 canonical audit categories, and emits
structured machine-readable logs and JSON summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def classify_failure(test_name: str, traceback_str: str) -> Tuple[str, str]:
    """Classifies a failed test into one of the 7 audit categories with justification."""
    # Category 2: Known pre-existing Phase 12 A* straight line point count assertion
    if "test_astar_flat_ground_straight_line" in test_name:
        return (
            "Known pre-existing failure",
            "Phase 12 straight-line A* test expected <=4 waypoints, but path smoothing interpolation "
            "produces 31 points (0.1m resolution over 3.0m). Documented pre-existing baseline behavior."
        )

    # Category 4: Test harness defect in test_dwa_acceleration_parity.py MockCostmap schema
    if "test_dwa_acceleration_parity" in test_name and ("test_end_to_end_nominal_planning_parity" in test_name or "test_dynamic_obstacle_avoidance_parity" in test_name):
        return (
            "Test harness defect",
            "MockCostmap in test fixture lacks x_range_m/y_range_m bounds metadata matching the production "
            "BEVTerrainCostmap/WorldTerrainMap convention (top-left row 0 at x_max), causing synthetic "
            "out-of-bounds evaluation inside the accelerated static collision wrapper."
        )

    # Category 2 / 6: Known pre-existing Phase 11/12 camera intrinsics baseline mismatch
    if "test_simulated_camera_generation" in test_name:
        return (
            "Known pre-existing failure",
            "Phase 11/12 GazeboSimEngine configured with RealSense D435 calibrated intrinsics (fx=fy=462.14 for 69.4 deg HFOV on 640x480), but test assertion expected legacy 525.0. Identical in Phase 12 baseline commit 7555164."
        )

    # Category 3: Environment limitation (ROS 2 / Gazebo / MATLAB / CUDA missing)
    if any(k in traceback_str.lower() for k in ["rclpy", "ros2", "cuda", "gazebo", "matlab"]):
        return (
            "Environment limitation",
            "Test requires external runtime or hardware daemon not present in local execution environment."
        )

    # Default classification
    return (
        "Genuine regression",
        "Unexpected test failure requiring engineering investigation."
    )


def run_cpu_regression(
    output_dir: Path,
    pytest_targets: List[str] = None,
    extra_flags: List[str] = None,
) -> int:
    """Runs the full authoritative CPU regression suite."""
    print("=" * 80)
    print("       SIH2026: AUTHORITATIVE CPU REGRESSION HARNESS (PHASE 13.18)")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = pytest_targets or ["tests/"]
    flags = extra_flags or ["-v", "--tb=short"]

    cmd = [sys.executable, "-m", "pytest"] + targets + flags
    print(f"Executing: {' '.join(cmd)}")
    print("Running test suite (this may take several minutes)...")

    start_time = time.time()
    results_file = output_dir / "pytest_results.txt"
    stdout_lines: List[str] = []

    with open(results_file, "w", encoding="utf-8") as f_out:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f_out.write(line)
            f_out.flush()
            stdout_lines.append(line)
        proc.wait()

    elapsed = time.time() - start_time
    stdout = "".join(stdout_lines)
    print(f"\n[+] Raw results saved to: {results_file}")

    # Parse counts from pytest summary line
    # e.g.: "==== 1 failed, 808 passed in 120.50s ===="
    passed = 0
    failed = 0
    skipped = 0
    errors = 0
    warnings = 0

    pass_match = re.search(r"(\d+)\s+passed", stdout)
    fail_match = re.search(r"(\d+)\s+failed", stdout)
    skip_match = re.search(r"(\d+)\s+skipped", stdout)
    err_match = re.search(r"(\d+)\s+errors?", stdout)
    warn_match = re.search(r"(\d+)\s+warnings?", stdout)

    if pass_match:
        passed = int(pass_match.group(1))
    if fail_match:
        failed = int(fail_match.group(1))
    if skip_match:
        skipped = int(skip_match.group(1))
    if err_match:
        errors = int(err_match.group(1))
    if warn_match:
        warnings = int(warn_match.group(1))

    total = passed + failed + skipped + errors

    # Extract failed test names
    failure_details: List[Dict[str, Any]] = []
    lines = stdout.splitlines()
    in_summary = False
    for line in lines:
        if "=== short test summary info ===" in line:
            in_summary = True
            continue
        if in_summary and line.startswith("FAILED "):
            parts = line[7:].split(" - ")
            t_name = parts[0].strip()
            tb = parts[1].strip() if len(parts) > 1 else ""
            cat, reason = classify_failure(t_name, tb)
            failure_details.append({
                "test": t_name,
                "category": cat,
                "reason": reason,
                "traceback_summary": tb,
            })

    summary: Dict[str, Any] = {
        "timestamp": time.time(),
        "command": " ".join(cmd),
        "platform": platform.platform(),
        "python": sys.version,
        "elapsed_seconds": round(elapsed, 2),
        "total_collected": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "warnings": warnings,
        "pass_rate_pct": round((passed / total * 100) if total > 0 else 0.0, 2),
        "failures": failure_details,
    }

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[+] Summary saved to: {summary_file}")
    print("\n--- Regression Results Summary ---")
    print(f"  Total Evaluated: {total}")
    print(f"  Passed:          {passed}")
    print(f"  Failed:          {failed}")
    print(f"  Skipped:         {skipped}")
    print(f"  Errors:          {errors}")
    print(f"  Warnings:        {warnings}")
    print(f"  Elapsed Time:    {elapsed:.2f}s")
    print(f"  Pass Rate:       {summary['pass_rate_pct']}%")

    if failure_details:
        print("\n--- Failure Breakdown & Categorization ---")
        for idx, f in enumerate(failure_details, 1):
            print(f"  {idx}. {f['test']}")
            print(f"     Category: {f['category']}")
            print(f"     Reason:   {f['reason']}")

    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Authoritative CPU Regression Harness")
    parser.add_argument("--output-dir", type=str, default="simulation/outputs/phase18_regression/cpu",
                        help="Output directory for regression logs and summary")
    parser.add_argument("pytest_args", nargs="*", default=[],
                        help="Optional specific test targets or pytest arguments")
    args = parser.parse_args()

    targets = args.pytest_args if args.pytest_args else None
    return run_cpu_regression(Path(args.output_dir), pytest_targets=targets)


if __name__ == "__main__":
    sys.exit(main())
