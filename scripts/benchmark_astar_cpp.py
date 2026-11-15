"""Scalability & Latency Benchmark: Pure-Python Reference vs. Native C++17 Global A*.

Evaluates search and smoothing performance across canonical grid scales:
  - 50x50   (2,500 cells)
  - 100x100 (10,000 cells)
  - 120x120 (14,400 cells — Milestone 13.11 Dynamic Replanning Grid)
  - 200x200 (40,000 cells — Phase 14.5 Rolling World Map)
  - 400x400 (160,000 cells — Full Outdoor Exploration Map)

Measures Mean, Median (P50), P95, Max latencies, Speedup Factor, and verifies numerical parity.
Outputs results to reports/benchmark_astar_cpp.json and prints authoritative summary table.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import GoalState, PlannerStatus, RobotState
from ai.planning.global_planners.astar import AStarPlanner


def generate_benchmark_costmap(dim: int, res: float = 0.10, seed: int = 42) -> Tuple[BEVTerrainCostmap, RobotState, GoalState]:
    """Create a repeatable synthetic outdoor costmap of shape (dim, dim) with realistic obstacles and hazards."""
    rng = np.random.RandomState(seed)
    half_span = (dim * res) / 2.0
    x_range = (-half_span, half_span)
    y_range = (-half_span, half_span)

    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=x_range,
        y_range_m=y_range,
        resolution_m=res,
        default_cost=0.10,
    )
    costmap.unknown.fill(False)
    costmap.hazard.fill(0.0)
    costmap.slope_deg.fill(0.0)
    costmap.roughness.fill(0.0)
    costmap.height_var.fill(0.0)
    costmap.traversability.fill(0.90)

    # 1. Add rock clusters (obstacles)
    num_clusters = max(3, dim // 20)
    for _ in range(num_clusters):
        cx = rng.randint(dim // 6, 5 * dim // 6)
        cy = rng.randint(dim // 6, 5 * dim // 6)
        radius = rng.randint(2, max(3, dim // 30))
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    nr, nc = cx + dr, cy + dc
                    if 0 <= nr < dim and 0 <= nc < dim:
                        costmap.obstacle[nr, nc] = True
                        costmap.cost[nr, nc] = 1.0

    # 2. Add water hazard stripe
    haz_row = dim // 2
    costmap.hazard[haz_row - 2 : haz_row + 3, dim // 4 : 3 * dim // 4] = 0.85

    # 3. Add slope variation
    costmap.slope_deg[dim // 3 : 2 * dim // 3, dim // 3 : 2 * dim // 3] = 12.0

    # Define start and goal poses ensuring obstacle-free zones
    start_x = -half_span * 0.75
    start_y = -half_span * 0.75
    goal_x = half_span * 0.75
    goal_y = half_span * 0.75

    # Clear 1.0m radius around start and goal
    r_s, c_s = costmap.meter_to_grid(start_x, start_y)
    r_g, c_g = costmap.meter_to_grid(goal_x, goal_y)
    clear_radius = max(3, int(1.0 / res))

    for dr in range(-clear_radius, clear_radius + 1):
        for dc in range(-clear_radius, clear_radius + 1):
            if 0 <= r_s + dr < dim and 0 <= c_s + dc < dim:
                costmap.obstacle[r_s + dr, c_s + dc] = False
                costmap.hazard[r_s + dr, c_s + dc] = 0.0
                costmap.cost[r_s + dr, c_s + dc] = 0.10
            if 0 <= r_g + dr < dim and 0 <= c_g + dc < dim:
                costmap.obstacle[r_g + dr, c_g + dc] = False
                costmap.hazard[r_g + dr, c_g + dc] = 0.0
                costmap.cost[r_g + dr, c_g + dc] = 0.10

    start = RobotState(x=start_x, y=start_y, yaw=0.0)
    goal = GoalState(x=goal_x, y=goal_y)
    return costmap, start, goal


def run_benchmark():
    grid_configs = [
        {"dim": 50,  "name": "50x50 (2,500 cells)",   "n_iters": 10},
        {"dim": 100, "name": "100x100 (10,000 cells)", "n_iters": 10},
        {"dim": 120, "name": "120x120 (14,400 cells — M13.11 Replan)", "n_iters": 10},
        {"dim": 200, "name": "200x200 (40,000 cells — Phase 14.5 Rolling)", "n_iters": 10},
        {"dim": 400, "name": "400x400 (160,000 cells — Full Outdoor)", "n_iters": 5},
    ]

    print("=" * 85)
    print("SIH2026 WORKSTREAM WS-1A: NATIVE C++17 GLOBAL A* BENCHMARK SUITE")
    print("=" * 85)

    planner_py = AStarPlanner(backend="python")
    planner_scalar = AStarPlanner(backend="cpp_scalar")
    planner_simd = AStarPlanner(backend="cpp_simd")

    results_table = []
    full_data = []

    for cfg in grid_configs:
        dim = cfg["dim"]
        name = cfg["name"]
        n_iters = cfg["n_iters"]
        print(f"\n[*] Benchmarking Grid: {name} (Iterations: {n_iters})")

        costmap, start, goal = generate_benchmark_costmap(dim)

        # 1. Warmup
        planner_py.plan(costmap, start, goal)
        planner_scalar.plan(costmap, start, goal)
        planner_simd.plan(costmap, start, goal)

        # 2. Benchmark Pure-Python
        py_latencies = []
        py_nodes = 0
        py_cost = 0.0
        py_len = 0.0
        for _ in range(n_iters):
            t0 = time.perf_counter()
            path, status, diag = planner_py.plan(costmap, start, goal)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            assert status == PlannerStatus.SUCCESS
            py_latencies.append(dt_ms)
            py_nodes = diag.get("nodes_explored", 0)
            py_cost = diag.get("path_cost", 0.0)
            py_len = diag.get("path_length_m", 0.0)

        # 3. Benchmark C++ Scalar
        scalar_latencies = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            path, status, diag = planner_scalar.plan(costmap, start, goal)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            assert status == PlannerStatus.SUCCESS
            scalar_latencies.append(dt_ms)

        # 4. Benchmark C++ SIMD (AVX2)
        simd_latencies = []
        simd_nodes = 0
        simd_cost = 0.0
        simd_len = 0.0
        cpp_pure_latencies = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            path, status, diag = planner_simd.plan(costmap, start, goal)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            assert status == PlannerStatus.SUCCESS
            simd_latencies.append(dt_ms)
            simd_nodes = diag.get("nodes_explored", 0)
            simd_cost = diag.get("path_cost", 0.0)
            simd_len = diag.get("path_length_m", 0.0)
            cpp_pure_latencies.append(diag.get("cpp_total_ms", 0.0))

        # Statistical analysis
        py_mean = float(np.mean(py_latencies))
        py_p50 = float(np.median(py_latencies))
        py_p95 = float(np.percentile(py_latencies, 95))
        py_max = float(np.max(py_latencies))

        sc_mean = float(np.mean(scalar_latencies))
        sc_p50 = float(np.median(scalar_latencies))
        sc_p95 = float(np.percentile(scalar_latencies, 95))
        sc_max = float(np.max(scalar_latencies))

        simd_mean = float(np.mean(simd_latencies))
        simd_p50 = float(np.median(simd_latencies))
        simd_p95 = float(np.percentile(simd_latencies, 95))
        simd_max = float(np.max(simd_latencies))

        cpp_pure_mean = float(np.mean(cpp_pure_latencies))

        speedup_simd = py_mean / simd_mean if simd_mean > 0 else 0.0
        speedup_pure_cpp = py_mean / cpp_pure_mean if cpp_pure_mean > 0 else 0.0

        delta_cost = abs(simd_cost - py_cost)
        delta_len = abs(simd_len - py_len)

        print(f"    Python:     Mean: {py_mean:6.2f} ms | P50: {py_p50:6.2f} ms | P95: {py_p95:6.2f} ms | Max: {py_max:6.2f} ms")
        print(f"    C++ Scalar: Mean: {sc_mean:6.2f} ms | P50: {sc_p50:6.2f} ms | P95: {sc_p95:6.2f} ms | Max: {sc_max:6.2f} ms")
        print(f"    C++ SIMD:   Mean: {simd_mean:6.2f} ms | P50: {simd_p50:6.2f} ms | P95: {simd_p95:6.2f} ms | Max: {simd_max:6.2f} ms")
        print(f"    Pure C++:   Mean: {cpp_pure_mean:6.2f} ms (Excluding FFI wrapper overhead)")
        print(f"    Speedup:    {speedup_simd:.1f}x (End-to-End) / {speedup_pure_cpp:.1f}x (Pure C++ Core)")
        print(f"    Parity:     Nodes: Py={py_nodes} C++={simd_nodes} | Delta Cost: {delta_cost:.6f} | Delta Len: {delta_len:.6f} m")

        entry = {
            "grid_dim": dim,
            "grid_name": name,
            "cells": dim * dim,
            "nodes_explored": simd_nodes,
            "python": {
                "mean_ms": round(py_mean, 2),
                "p50_ms": round(py_p50, 2),
                "p95_ms": round(py_p95, 2),
                "max_ms": round(py_max, 2),
            },
            "cpp_scalar": {
                "mean_ms": round(sc_mean, 2),
                "p50_ms": round(sc_p50, 2),
                "p95_ms": round(sc_p95, 2),
                "max_ms": round(sc_max, 2),
            },
            "cpp_simd": {
                "mean_ms": round(simd_mean, 2),
                "p50_ms": round(simd_p50, 2),
                "p95_ms": round(simd_p95, 2),
                "max_ms": round(simd_max, 2),
                "cpp_pure_mean_ms": round(cpp_pure_mean, 2),
            },
            "speedup_factor": round(speedup_simd, 1),
            "pure_cpp_speedup": round(speedup_pure_cpp, 1),
            "parity_delta_cost": float(delta_cost),
            "parity_delta_length_m": float(delta_len),
        }
        full_data.append(entry)

    # Save to reports/
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_root, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "benchmark_astar_cpp.json")
    with open(out_file, "w") as f:
        json.dump({"benchmarks": full_data}, f, indent=2)

    print("\n" + "=" * 90)
    print("MASTER SUMMARY TABLE: PYTHON REFERENCE VS. NATIVE C++17 GLOBAL A*")
    print("=" * 90)
    header = f"| {'Grid Scale':<32} | {'Cells':<7} | {'Python Mean':<11} | {'C++ SIMD':<9} | {'Pure C++':<9} | {'Speedup':<8} | {'Nodes':<6} | {'Target':<9} | {'Status':<6} |"
    print(header)
    print("|" + "-" * 34 + "|" + "-" * 9 + "|" + "-" * 13 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|" + "-" * 8 + "|" + "-" * 11 + "|" + "-" * 8 + "|")

    targets = {
        50: 1.0,
        100: 1.5,
        120: 2.0,
        200: 3.5,
        400: 12.0
    }

    for row in full_data:
        dim = row["grid_dim"]
        cells = row["cells"]
        name = row["grid_name"]
        py_m = f"{row['python']['mean_ms']:.1f} ms"
        simd_m = f"{row['cpp_simd']['mean_ms']:.2f} ms"
        pure_m = f"{row['cpp_simd']['cpp_pure_mean_ms']:.2f} ms"
        spd = f"{row['speedup_factor']:.1f}x"
        nodes = row["nodes_explored"]
        tgt = f"< {targets.get(dim, 5.0):.1f} ms"
        passed = "PASS" if row['cpp_simd']['mean_ms'] < targets.get(dim, 5.0) else "WARN"
        print(f"| {name:<32} | {cells:<7} | {py_m:<11} | {simd_m:<9} | {pure_m:<9} | {spd:<8} | {nodes:<6} | {tgt:<9} | {passed:<6} |")

    print("=" * 90)
    print(f"[SUCCESS] Authoritative benchmark saved to {out_file}")


if __name__ == "__main__":
    run_benchmark()
