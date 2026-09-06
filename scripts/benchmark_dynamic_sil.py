"""Formal performance benchmark suite for Phase 13.13 Dynamic SIL navigation stack.

Characterizes:
1. Static Baseline (Scenario A: 0 obstacles)
2. Single Dynamic Obstacle (Scenario C: 1 obstacle)
3. Multiple Dynamic Obstacles (Scenario H: 4 obstacles)
4. Persistent Blockage + Detour Replan (Scenario G1: 1 obstacle + replanning)
5. Local Deadlock (Scenario O: surrounding static traps)

Measures:
- Wall-clock execution time (s)
- Simulated time (s)
- Real-Time Factor (RTF = sim_time / wall_time)
- DWA Planner latency (mean, P95, max)
- Step end-to-end latency (mean, P95, max)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from simulation.runners.dynamic_sil_runner import DynamicSILRunner
from simulation.scenarios.dynamic.core_scenarios import (
    create_scenario_a_static,
    create_scenario_c_head_on,
)
from simulation.scenarios.dynamic.advanced_scenarios import (
    create_scenario_g1_persistent_blockage_alternate,
    create_scenario_h_multiple_obstacles,
    create_scenario_o_deadlock,
)

BENCHMARK_CASES = [
    ("Static Baseline (0 obs)", create_scenario_a_static, 0.60),
    ("Single Dynamic Obs (1 obs)", create_scenario_c_head_on, 0.60),
    ("Multiple Dynamic Obs (4 obs)", create_scenario_h_multiple_obstacles, 0.60),
    ("Persistent Blockage Replan", create_scenario_g1_persistent_blockage_alternate, 0.60),
    ("Local Deadlock Trap", create_scenario_o_deadlock, 0.60),
]


def run_benchmark(output_json: str = "simulation/outputs/dynamic_sil/benchmark_summary.json"):
    print("================================================================================")
    print("PHASE 13.13 DYNAMIC SIL PERFORMANCE BENCHMARK")
    print("================================================================================")
    print("Characterizing runtime performance across representative environmental loads.")
    print("NOTE: Measured in pure Python. The system is functional SIL / correctness validated,")
    print("      NOT real-time (pure-Python DWA evaluates 375 candidate trajectories per cycle).\n")

    results: List[Dict[str, Any]] = []

    for name, builder, duration in BENCHMARK_CASES:
        cfg = builder(42)
        cfg.max_duration_s = duration

        runner = DynamicSILRunner(cfg)

        step_latencies: List[float] = []

        t0_wall = time.perf_counter()
        steps = int(round(cfg.max_duration_s / cfg.control_dt_s))

        for _ in range(steps):
            t0_step = time.perf_counter()
            rec = runner.step()
            dt_step = time.perf_counter() - t0_step
            step_latencies.append(dt_step)

            if rec.get("status") == "GOAL_REACHED" or rec.get("gt_collision", False):
                break

        wall_total = time.perf_counter() - t0_wall
        sim_total = float(runner.motion.state.timestamp)
        rtf = (sim_total / wall_total) if wall_total > 0 else 0.0

        plan_lats = runner.planner_latencies
        plan_mean = float(np.mean(plan_lats)) if plan_lats else 0.0
        plan_p95 = float(np.percentile(plan_lats, 95)) if plan_lats else 0.0
        plan_max = float(np.max(plan_lats)) if plan_lats else 0.0

        step_mean = float(np.mean(step_latencies)) if step_latencies else 0.0
        step_p95 = float(np.percentile(step_latencies, 95)) if step_latencies else 0.0
        step_max = float(np.max(step_latencies)) if step_latencies else 0.0

        res_dict = {
            "case_name": name,
            "scenario_id": cfg.scenario_id,
            "obstacle_count": len(cfg.obstacles),
            "sim_duration_s": round(sim_total, 2),
            "wall_duration_s": round(wall_total, 2),
            "real_time_factor": round(rtf, 4),
            "steps_executed": len(step_latencies),
            "planner_mean_latency_s": round(plan_mean, 4),
            "planner_p95_latency_s": round(plan_p95, 4),
            "planner_max_latency_s": round(plan_max, 4),
            "step_mean_latency_s": round(step_mean, 4),
            "step_p95_latency_s": round(step_p95, 4),
            "step_max_latency_s": round(step_max, 4),
            "collision_count": len(runner.collision_events),
            "min_clearance_m": round(runner.min_clearance_observed, 3) if runner.min_clearance_observed != float("inf") else 0.0,
        }
        results.append(res_dict)

        print(f"[{name}]")
        print(f"  Sim: {sim_total:.2f}s | Wall: {wall_total:.2f}s | RTF: {rtf:.3f}x")
        print(f"  Planner Latency: Mean={plan_mean:.3f}s, P95={plan_p95:.3f}s, Max={plan_max:.3f}s")
        print(f"  Step Latency:    Mean={step_mean:.3f}s, P95={step_p95:.3f}s, Max={step_max:.3f}s")
        print(f"  Safety:          Collisions={len(runner.collision_events)}, MinClearance={runner.min_clearance_observed:.3f}m\n")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark summary JSON saved: {output_json}")

    print("\n================================================================================")
    print("BENCHMARK COMPARISON TABLE")
    print("================================================================================")
    header = f"{'Case Name':28s} | {'Obs':3s} | {'Sim(s)':6s} | {'Wall(s)':7s} | {'RTF':7s} | {'Plan Mean':9s} | {'Plan P95':8s} | {'Collisions'}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['case_name']:28s} | {r['obstacle_count']:3d} | {r['sim_duration_s']:6.2f} | "
            f"{r['wall_duration_s']:7.2f} | {r['real_time_factor']:7.4f} | {r['planner_mean_latency_s']:8.3f}s | "
            f"{r['planner_p95_latency_s']:7.3f}s | {r['collision_count']:10d}"
        )


if __name__ == "__main__":
    run_benchmark()
