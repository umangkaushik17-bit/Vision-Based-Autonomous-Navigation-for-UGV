"""Milestone 13.16: Full Closed-Loop Benchmark Execution Driver.

Executes canonical scenarios B01 through B12, multi-seed robustness suites,
and dedicated performance profiling under strict ground-truth isolation.

Outputs authoritative artifacts to simulation/outputs/phase16/:
- benchmark_manifest.json
- scenario_matrix.json
- mission_results.json
- perception_results.json
- planning_results.json
- safety_results.json
- performance_results.json
- aggregate_results.json
- seed_statistics.json
- telemetry/
- plots/
- logs/
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from simulation.runners.phase16_sil_runner import Phase16SILRunner
from simulation.scenarios.dynamic.phase16_scenarios import (
    BENCHMARK_SCENARIO_REGISTRY,
    Phase16ScenarioConfig,
)

OUTPUT_DIR = os.path.join("simulation", "outputs", "phase16")
TELEMETRY_DIR = os.path.join(OUTPUT_DIR, "telemetry")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")


def get_environment_info() -> Dict[str, Any]:
    """Capture execution environment metadata without exposing secrets."""
    cpu_info = platform.processor() or platform.machine()
    python_ver = sys.version.split()[0]
    os_info = f"{platform.system()} {platform.release()}"

    gpu_name = "N/A"
    gpu_vram = "N/A"
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = res.stdout.strip().split(",")
            gpu_name = parts[0].strip()
            if len(parts) > 1:
                gpu_vram = parts[1].strip()
    except Exception:
        pass

    return {
        "hostname": platform.node(),
        "os": os_info,
        "python_version": python_ver,
        "cpu_processor": cpu_info,
        "cpu_count": os.cpu_count(),
        "gpu_present": gpu_name != "N/A",
        "gpu_name": gpu_name,
        "gpu_vram": gpu_vram,
        "gpu_used_by_planner": False,  # DWA/A* is CPU-bound pure Python
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def define_batches() -> Dict[str, List[Tuple[str, int]]]:
    """Define deterministic mission runs per batch."""
    return {
        "1": [
            ("B01", 42),
            ("B02", 42),
            ("B03", 42),
            ("B04", 42),
        ],
        "2": [
            ("B05", 42),
            ("B06", 42),
            ("B07", 42),
            ("B08", 42),
        ],
        "3": [
            ("B09", 42),
            ("B10", 42),
            ("B11", 42),
            ("B12", 42),
        ],
        "4": [
            # Repeat seeds for B01, B06, B07, B12
            ("B01", 101),
            ("B01", 202),
            ("B06", 101),
            ("B06", 202),
            ("B07", 101),
            ("B07", 202),
            ("B12", 101),
            ("B12", 202),
        ],
        "5": [
            # Dedicated performance characterization run (>=100 steps)
            ("B06", 999),
        ],
    }


def execute_single_run(
    scenario_key: str,
    seed: int,
    max_steps_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute one benchmark mission and extract multi-dimensional metrics."""
    builder = BENCHMARK_SCENARIO_REGISTRY[scenario_key]
    cfg = builder(seed)

    if max_steps_override is not None:
        cfg.max_duration_s = max_steps_override * cfg.control_dt_s

    print(f"\n[{scenario_key} | Seed {seed}] Starting {cfg.name} (Max Duration: {cfg.max_duration_s:.1f}s)...")

    t_wall_start = time.perf_counter()
    runner = Phase16SILRunner(cfg, enable_replanner=True)

    metrics = runner.run()

    t_wall_total = time.perf_counter() - t_wall_start

    # Extract detailed evaluation
    det = metrics.detailed_metrics

    # Calculate localization errors if ground truth vs estimate exists in telemetry
    pos_errs = []
    yaw_errs = []
    for t in runner.telemetry:
        if "est_x" in t and "robot_x" in t:
            pos_err = math.hypot(t["robot_x"] - t["est_x"], t["robot_y"] - t["est_y"])
            yaw_err = abs(math.atan2(math.sin(t["robot_yaw"] - t["est_yaw"]), math.cos(t["robot_yaw"] - t["est_yaw"])))
            pos_errs.append(pos_err)
            yaw_errs.append(yaw_err)

    loc_pos_rmse = float(np.sqrt(np.mean(np.square(pos_errs)))) if pos_errs else 0.0
    loc_yaw_rmse = float(np.sqrt(np.mean(np.square(yaw_errs)))) if yaw_errs else 0.0
    loc_max_pos = float(np.max(pos_errs)) if pos_errs else 0.0
    loc_max_yaw = float(np.max(yaw_errs)) if yaw_errs else 0.0

    # Step latencies
    step_lats = runner.step_latencies_ms
    lat_mean = float(np.mean(step_lats)) if step_lats else 0.0
    lat_p50 = float(np.percentile(step_lats, 50)) if step_lats else 0.0
    lat_p95 = float(np.percentile(step_lats, 95)) if step_lats else 0.0
    lat_p99 = float(np.percentile(step_lats, 99)) if step_lats else 0.0
    lat_max = float(np.max(step_lats)) if step_lats else 0.0

    # Path efficiency: straight_line / actual_travel
    sx, sy, _ = cfg.start_pose
    gx, gy, _ = cfg.goal_pose
    straight_dist = math.hypot(gx - sx, gy - sy)
    actual_path_len = metrics.travel_distance_m
    path_eff = (straight_dist / actual_path_len) if (actual_path_len > 0.1 and metrics.mission_outcome == "GOAL_REACHED") else None

    # Candidate statistics
    avg_valid_cands = float(np.mean(runner.dwa_valid_candidates_history)) if runner.dwa_valid_candidates_history else 0.0
    avg_rej_cands = float(np.mean(runner.dwa_rejected_candidates_history)) if runner.dwa_rejected_candidates_history else 0.0

    run_record = {
        "scenario_key": scenario_key,
        "scenario_id": cfg.scenario_id,
        "scenario_name": cfg.name,
        "seed": seed,
        "category": cfg.benchmark_category,
        "configured_duration_s": cfg.max_duration_s,
        "actual_simulation_time_s": metrics.actual_simulation_time_s,
        "wall_clock_duration_s": round(t_wall_total, 3),
        "steps_executed": metrics.steps_executed,
        "real_time_factor": round(metrics.real_time_factor, 3),
        # Mission Level
        "mission_outcome": metrics.mission_outcome,
        "safety_outcome": metrics.safety_outcome,
        "termination_reason": metrics.termination_reason,
        "mission_success": metrics.mission_success,
        "goal_position_error_m": round(metrics.goal_position_error_m, 4),
        "goal_tolerance_m": metrics.goal_tolerance_m,
        "collision_count": metrics.collision_count,
        "minimum_clearance_m": round(metrics.minimum_clearance_m, 3),
        "path_length_m": round(actual_path_len, 3),
        "straight_line_distance_m": round(straight_dist, 3),
        "path_efficiency": round(path_eff, 4) if path_eff is not None else None,
        "average_speed_m_s": round(metrics.average_speed_m_s, 3),
        "max_speed_m_s": round(metrics.max_speed_m_s, 3),
        "proceed_count": metrics.proceed_count,
        "avoid_count": metrics.avoid_count,
        "wait_count": metrics.wait_count,
        "stop_count": metrics.stop_count,
        "deadlock_count": metrics.deadlock_count,
        # Perception / Tracking
        "detection_count": det.detection_count if det else 0,
        "confirmed_track_count": det.confirmed_track_count if det else 0,
        "track_continuity": round(det.track_continuity, 4) if det else 1.0,
        "tracking_position_rmse_m": round(det.tracking_position_rmse_m, 4) if det else 0.0,
        "tracking_velocity_rmse_m_s": round(det.tracking_velocity_rmse_m_s, 4) if det else 0.0,
        # Prediction
        "prediction_rmse_0_50s": round(det.prediction_rmse_0_50s, 4) if det else 0.0,
        "prediction_rmse_1_00s": round(det.prediction_rmse_1_00s, 4) if det else 0.0,
        "prediction_rmse_2_00s": round(det.prediction_rmse_2_00s, 4) if det else 0.0,
        # Planning
        "local_planner_cycles": metrics.local_planner_cycles,
        "global_replan_count": metrics.global_replan_count,
        "failed_global_replan_count": metrics.failed_global_replan_count,
        "avg_valid_candidates": round(avg_valid_cands, 1),
        "avg_rejected_candidates": round(avg_rej_cands, 1),
        # Safety
        "emergency_stop_count": metrics.emergency_stop_count,
        "near_collision_count": det.near_collision_count if det else 0,
        "unsafe_command_count": det.unsafe_command_count if det else 0,
        # Localization
        "localization_position_rmse_m": round(loc_pos_rmse, 4),
        "localization_yaw_rmse_rad": round(loc_yaw_rmse, 4),
        "localization_max_pos_error_m": round(loc_max_pos, 4),
        "localization_max_yaw_error_rad": round(loc_max_yaw, 4),
        # Performance Timing (ms)
        "step_latency_mean_ms": round(lat_mean, 3),
        "step_latency_p50_ms": round(lat_p50, 3),
        "step_latency_p95_ms": round(lat_p95, 3),
        "step_latency_p99_ms": round(lat_p99, 3),
        "step_latency_max_ms": round(lat_max, 3),
        "planner_latency_mean_ms": round(metrics.planner_latency_mean_s * 1000.0, 3),
        "planner_latency_max_ms": round(metrics.planner_latency_max_s * 1000.0, 3),
    }

    # Save per-run telemetry JSON
    os.makedirs(TELEMETRY_DIR, exist_ok=True)
    telem_file = os.path.join(TELEMETRY_DIR, f"{cfg.scenario_id}_seed{seed}.json")
    with open(telem_file, "w", encoding="utf-8") as f:
        json.dump({
            "scenario_metadata": run_record,
            "telemetry_points": runner.telemetry,
        }, f, indent=2)

    print(f"[{scenario_key} | Seed {seed}] Result: {metrics.mission_outcome} | Safety: {metrics.safety_outcome} | "
          f"Steps: {metrics.steps_executed} | Clearance: {metrics.minimum_clearance_m:.3f}m | "
          f"P95 Latency: {lat_p95:.1f}ms | RTF: {metrics.real_time_factor:.2f}")

    return run_record


def calculate_seed_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics for repeated scenario runs."""
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_scenario.setdefault(r["scenario_key"], []).append(r)

    stats: Dict[str, Any] = {}
    for sk, group in by_scenario.items():
        total_runs = len(group)
        success_runs = sum(1 for r in group if r["mission_outcome"] == "GOAL_REACHED")
        collision_runs = sum(1 for r in group if r["safety_outcome"] == "COLLISION")
        safe_incomplete_runs = sum(1 for r in group if r["mission_outcome"] == "INCOMPLETE" and r["safety_outcome"] == "SAFE")

        clearances = [r["minimum_clearance_m"] for r in group]
        goal_errors = [r["goal_position_error_m"] for r in group]
        latencies = [r["step_latency_p95_ms"] for r in group]

        stats[sk] = {
            "scenario_id": group[0]["scenario_id"],
            "total_runs": total_runs,
            "success_count": success_runs,
            "collision_count": collision_runs,
            "safe_incomplete_count": safe_incomplete_runs,
            "mission_success_rate": round(success_runs / total_runs, 4),
            "collision_rate": round(collision_runs / total_runs, 4),
            "safe_incomplete_rate": round(safe_incomplete_runs / total_runs, 4),
            "clearance_min_m": round(float(np.min(clearances)), 3),
            "clearance_mean_m": round(float(np.mean(clearances)), 3),
            "clearance_p95_m": round(float(np.percentile(clearances, 95)), 3),
            "goal_error_mean_m": round(float(np.mean(goal_errors)), 4),
            "goal_error_median_m": round(float(np.median(goal_errors)), 4),
            "p95_latency_mean_ms": round(float(np.mean(latencies)), 2),
            "p95_latency_max_ms": round(float(np.max(latencies)), 2),
        }
    return stats


def run_benchmark_suite(batch_arg: str = "all") -> bool:
    """Execute designated benchmark batch or complete suite."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TELEMETRY_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    env_info = get_environment_info()

    all_batches = define_batches()
    if batch_arg == "all":
        selected_runs = [run for b in all_batches.values() for run in b]
    elif batch_arg in all_batches:
        selected_runs = all_batches[batch_arg]
    else:
        print(f"[-] Invalid batch '{batch_arg}'. Choose 1, 2, 3, 4, 5, or 'all'.")
        return False

    print("=" * 75)
    print(f"MILESTONE 13.16: FULL CLOSED-LOOP BENCHMARK (Batch: {batch_arg.upper()})")
    print(f"Environment: OS: {env_info['os']} | CPU: {env_info['cpu_processor']} | GPU: {env_info['gpu_name']}")
    print(f"Scheduled runs: {len(selected_runs)}")
    print("=" * 75)

    existing_results: List[Dict[str, Any]] = []
    results_path = os.path.join(OUTPUT_DIR, "mission_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = []

    # Map existing by (scenario_key, seed) to update / append
    results_map = {(r["scenario_key"], r["seed"]): r for r in existing_results}

    t0_suite = time.perf_counter()
    for sk, seed in selected_runs:
        max_steps = 120 if (batch_arg == "5" or seed == 999) else None
        rec = execute_single_run(sk, seed, max_steps_override=max_steps)
        results_map[(sk, seed)] = rec

    all_results = list(results_map.values())
    total_wall = time.perf_counter() - t0_suite

    # Compute aggregate seed statistics
    seed_stats = calculate_seed_statistics(all_results)

    # 1. benchmark_manifest.json
    manifest = {
        "benchmark_version": "13.16.0",
        "benchmark_name": "Phase 13 Full Closed-Loop Benchmark",
        "timestamp_utc": env_info["timestamp_utc"],
        "environment": env_info,
        "batch_executed": batch_arg,
        "total_runs_recorded": len(all_results),
        "total_wall_clock_s": round(total_wall, 2),
    }
    with open(os.path.join(OUTPUT_DIR, "benchmark_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 2. scenario_matrix.json
    matrix = [
        {
            "scenario_key": sk,
            "scenario_id": builder(42).scenario_id,
            "scenario_name": builder(42).name,
            "category": builder(42).benchmark_category,
            "description": builder(42).description,
            "start_pose": list(builder(42).start_pose),
            "goal_pose": list(builder(42).goal_pose),
            "max_duration_s": builder(42).max_duration_s,
            "obstacle_count": len(builder(42).obstacles),
            "static_obstacle_count": len(builder(42).static_obstacles),
        }
        for sk, builder in BENCHMARK_SCENARIO_REGISTRY.items()
    ]
    with open(os.path.join(OUTPUT_DIR, "scenario_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)

    # 3. mission_results.json
    with open(os.path.join(OUTPUT_DIR, "mission_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # 4. perception_results.json
    perception_data = [
        {
            "scenario_key": r["scenario_key"],
            "seed": r["seed"],
            "detection_count": r["detection_count"],
            "confirmed_track_count": r["confirmed_track_count"],
            "track_continuity": r["track_continuity"],
            "tracking_position_rmse_m": r["tracking_position_rmse_m"],
            "tracking_velocity_rmse_m_s": r["tracking_velocity_rmse_m_s"],
            "prediction_rmse_0_50s": r["prediction_rmse_0_50s"],
            "prediction_rmse_1_00s": r["prediction_rmse_1_00s"],
            "prediction_rmse_2_00s": r["prediction_rmse_2_00s"],
        }
        for r in all_results
    ]
    with open(os.path.join(OUTPUT_DIR, "perception_results.json"), "w", encoding="utf-8") as f:
        json.dump(perception_data, f, indent=2)

    # 5. planning_results.json
    planning_data = [
        {
            "scenario_key": r["scenario_key"],
            "seed": r["seed"],
            "local_planner_cycles": r["local_planner_cycles"],
            "global_replan_count": r["global_replan_count"],
            "failed_global_replan_count": r["failed_global_replan_count"],
            "path_length_m": r["path_length_m"],
            "path_efficiency": r["path_efficiency"],
            "avg_valid_candidates": r["avg_valid_candidates"],
            "avg_rejected_candidates": r["avg_rejected_candidates"],
            "planner_latency_mean_ms": r["planner_latency_mean_ms"],
            "planner_latency_max_ms": r["planner_latency_max_ms"],
        }
        for r in all_results
    ]
    with open(os.path.join(OUTPUT_DIR, "planning_results.json"), "w", encoding="utf-8") as f:
        json.dump(planning_data, f, indent=2)

    # 6. safety_results.json
    safety_data = [
        {
            "scenario_key": r["scenario_key"],
            "seed": r["seed"],
            "safety_outcome": r["safety_outcome"],
            "collision_count": r["collision_count"],
            "minimum_clearance_m": r["minimum_clearance_m"],
            "emergency_stop_count": r["emergency_stop_count"],
            "near_collision_count": r["near_collision_count"],
            "unsafe_command_count": r["unsafe_command_count"],
        }
        for r in all_results
    ]
    with open(os.path.join(OUTPUT_DIR, "safety_results.json"), "w", encoding="utf-8") as f:
        json.dump(safety_data, f, indent=2)

    # 7. performance_results.json
    performance_data = [
        {
            "scenario_key": r["scenario_key"],
            "seed": r["seed"],
            "steps_executed": r["steps_executed"],
            "actual_simulation_time_s": r["actual_simulation_time_s"],
            "wall_clock_duration_s": r["wall_clock_duration_s"],
            "real_time_factor": r["real_time_factor"],
            "step_latency_mean_ms": r["step_latency_mean_ms"],
            "step_latency_p50_ms": r["step_latency_p50_ms"],
            "step_latency_p95_ms": r["step_latency_p95_ms"],
            "step_latency_p99_ms": r["step_latency_p99_ms"],
            "step_latency_max_ms": r["step_latency_max_ms"],
        }
        for r in all_results
    ]
    with open(os.path.join(OUTPUT_DIR, "performance_results.json"), "w", encoding="utf-8") as f:
        json.dump(performance_data, f, indent=2)

    # 8. aggregate_results.json
    total_completed = len(all_results)
    total_success = sum(1 for r in all_results if r["mission_outcome"] == "GOAL_REACHED")
    total_collision = sum(1 for r in all_results if r["safety_outcome"] == "COLLISION")
    total_safe_inc = sum(1 for r in all_results if r["mission_outcome"] == "INCOMPLETE" and r["safety_outcome"] == "SAFE")

    aggregate_summary = {
        "total_missions_evaluated": total_completed,
        "overall_mission_success_count": total_success,
        "overall_mission_success_rate": round(total_success / total_completed, 4) if total_completed > 0 else 0.0,
        "overall_collision_count": total_collision,
        "overall_collision_rate": round(total_collision / total_completed, 4) if total_completed > 0 else 0.0,
        "overall_safe_incomplete_count": total_safe_inc,
        "overall_safe_incomplete_rate": round(total_safe_inc / total_completed, 4) if total_completed > 0 else 0.0,
        "overall_mean_clearance_m": round(float(np.mean([r["minimum_clearance_m"] for r in all_results])), 3) if all_results else 0.0,
        "overall_mean_p95_latency_ms": round(float(np.mean([r["step_latency_p95_ms"] for r in all_results])), 2) if all_results else 0.0,
        "overall_mean_rtf": round(float(np.mean([r["real_time_factor"] for r in all_results])), 3) if all_results else 0.0,
    }
    with open(os.path.join(OUTPUT_DIR, "aggregate_results.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2)

    # 9. seed_statistics.json
    with open(os.path.join(OUTPUT_DIR, "seed_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(seed_stats, f, indent=2)

    print("\n" + "=" * 75)
    print(f"BATCH {batch_arg.upper()} COMPLETED. Total Runs Recorded: {total_completed}")
    print(f"Overall Success Rate : {aggregate_summary['overall_mission_success_rate']*100:.1f}% ({total_success}/{total_completed})")
    print(f"Overall Collision Rate: {aggregate_summary['overall_collision_rate']*100:.1f}% ({total_collision}/{total_completed})")
    print(f"Safe Incomplete Rate : {aggregate_summary['overall_safe_incomplete_rate']*100:.1f}% ({total_safe_inc}/{total_completed})")
    print(f"Artifacts preserved in: {OUTPUT_DIR}")
    print("=" * 75 + "\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Milestone 13.16 Full Closed-Loop Benchmark Driver")
    parser.add_argument("--batch", type=str, default="all", help="Batch number to run: 1, 2, 3, 4, 5, or all")
    parser.add_argument("--scenario", type=str, default=None, help="Run single scenario (e.g. B01)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for single scenario run")

    args = parser.parse_args()

    if args.scenario:
        sk = args.scenario.upper()
        if sk not in BENCHMARK_SCENARIO_REGISTRY:
            print(f"[-] Unknown scenario '{sk}'. Registry: {list(BENCHMARK_SCENARIO_REGISTRY.keys())}")
            sys.exit(1)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        execute_single_run(sk, args.seed)
        sys.exit(0)

    success = run_benchmark_suite(batch_arg=args.batch.lower())
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
