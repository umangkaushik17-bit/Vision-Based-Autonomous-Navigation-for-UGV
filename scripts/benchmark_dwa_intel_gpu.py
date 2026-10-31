"""Phase 13.20: Comprehensive Intel GPU DWA Validation and Comparative Benchmark.

Benchmarks:
1. CPU Reference (DWAPlanner)
2. Vectorized CPU (AcceleratedDWAPlanner on CPU)
3. Intel GPU (AcceleratedDWAPlanner on xpu:0, Intel Arc integrated GPU)

Across 5 mandatory scenarios:
- Scenario A: Open environment
- Scenario B: Static obstacle
- Scenario C: Dynamic obstacle
- Scenario D: Multiple dynamic obstacles (4 moving obstacles)
- Scenario E: Heavy collision workload (dense obstacle field)

Protocol:
- 20 warmup iterations
- 100 measured iterations
- Hardware synchronization: torch.xpu.synchronize() / time.perf_counter()
- Workload: 375 candidates (15 v x 25 w), 1.50s horizon, 0.10s dt, 15 steps
- Measures: mean, median, P95, min, max, std, candidates/sec, XPU memory
Saves machine-readable report to: reports/dwa_intel_gpu_validation.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.planning.backends import (
    CPUDWABackend,
    CUDADWABackend,
    DWABackendType,
    IntelGPUDWABackend,
    UnifiedDWAPlanner,
    VectorizedCPUDWABackend,
)
from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.path import Path
from ai.spatial_fusion.world_map import WorldTerrainMap
from scripts.benchmark_dwa_dynamic import create_world_map, make_obstacle


def build_benchmark_scenarios() -> List[Tuple[str, RobotState, Path, WorldTerrainMap, Any, Any]]:
    """Build the 5 deterministic scenarios matching Step 7 specifications."""
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    scenarios = []

    # Scenario A: Open Environment
    wmap_a = create_world_map()
    scenarios.append(("Scenario A: Open Environment", state, path, wmap_a, None, None))

    # Scenario B: Static Obstacle
    wmap_b = create_world_map()
    for y_v in np.linspace(-0.5, 0.5, 10):
        for x_v in np.linspace(1.5, 2.5, 10):
            r, c = wmap_b.world_to_grid(x_v, y_v)
            wmap_b.obstacle[r, c] = True
            wmap_b.cost[r, c] = 1.00
    scenarios.append(("Scenario B: Static Obstacle", state, path, wmap_b, None, None))

    # Scenario C: Dynamic Obstacle (Crossing Trajectory)
    wmap_c = create_world_map()
    obs_c = make_obstacle(track_id=2, start_pos=(1.5, -1.0), velocity=(0.0, 1.0))
    layer_c = DynamicOccupancyLayer()
    layer_c.update_predictions([obs_c])
    scenarios.append(("Scenario C: Dynamic Obstacle", state, path, wmap_c, layer_c, [obs_c]))

    # Scenario D: Multiple Obstacles (4 Spatiotemporally Distributed Dynamic Obstacles)
    wmap_d = create_world_map()
    obs_d = [
        make_obstacle(track_id=10, start_pos=(1.0, 2.0), velocity=(0.0, -0.8)),
        make_obstacle(track_id=11, start_pos=(2.5, -2.0), velocity=(0.0, 0.8)),
        make_obstacle(track_id=12, start_pos=(3.0, 1.5), velocity=(-0.4, -0.4)),
        make_obstacle(track_id=13, start_pos=(0.5, -1.5), velocity=(0.3, 0.3)),
    ]
    layer_d = DynamicOccupancyLayer()
    layer_d.update_predictions(obs_d)
    scenarios.append(("Scenario D: Multiple Dynamic Obstacles", state, path, wmap_d, layer_d, obs_d))

    # Scenario E: Heavy Collision Workload (Dense Obstacle Field)
    wmap_e = create_world_map()
    # Create multiple scattered obstacle clusters
    for ox, oy in [(1.0, 0.3), (1.5, -0.4), (2.0, 0.5), (2.5, -0.2), (3.0, 0.2)]:
        for y_v in np.linspace(oy - 0.2, oy + 0.2, 5):
            for x_v in np.linspace(ox - 0.2, ox + 0.2, 5):
                r, c = wmap_e.world_to_grid(x_v, y_v)
                wmap_e.obstacle[r, c] = True
                wmap_e.cost[r, c] = 1.00
    scenarios.append(("Scenario E: Heavy Collision Workload", state, path, wmap_e, None, None))

    return scenarios


def benchmark_backend_on_scenarios(
    backend_type: DWABackendType,
    scenarios: List[Tuple[str, RobotState, Path, WorldTerrainMap, Any, Any]],
    n_warmup: int = 20,
    n_measured: int = 100,
) -> Dict[str, Any]:
    cfg = PlanningConfig()
    planner = UnifiedDWAPlanner(config=cfg, backend=backend_type)
    backend_obj = planner.active_backend
    is_xpu = getattr(getattr(backend_obj, "planner", None), "backend", None) and getattr(backend_obj.planner.backend, "is_xpu", False)

    print(f"\n[{backend_type.value}] Benchmarking {backend_obj.name} ({backend_obj.device_name})...")

    results_by_scenario = []

    for name, state, path, wmap, layer, tracks in scenarios:
        # 1. Warmup (20 iterations)
        for _ in range(n_warmup):
            planner.compute_trajectory(state, path, wmap, dynamic_layer=layer, predicted_tracks=tracks)
            if is_xpu:
                torch.xpu.synchronize(0)

        # Reset memory stats if XPU
        if is_xpu:
            torch.xpu.reset_peak_memory_stats(0)

        # 2. Measured iterations (100 iterations)
        latencies_ms: List[float] = []
        last_v = 0.0
        last_w = 0.0
        last_status = PlannerStatus.SUCCESS
        last_score = 0.0

        for _ in range(n_measured):
            if is_xpu:
                torch.xpu.synchronize(0)
            t0 = time.perf_counter()
            traj, v, w, status, diag = planner.compute_trajectory(
                state, path, wmap, dynamic_layer=layer, predicted_tracks=tracks
            )
            if is_xpu:
                torch.xpu.synchronize(0)
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1000.0)
            last_v = v
            last_w = w
            last_status = status
            last_score = float(diag.get("best_score", 0.0))

        lat_arr = np.array(latencies_ms)
        mean_ms = float(np.mean(lat_arr))
        median_ms = float(np.median(lat_arr))
        p95_ms = float(np.percentile(lat_arr, 95))
        min_ms = float(np.min(lat_arr))
        max_ms = float(np.max(lat_arr))
        std_ms = float(np.std(lat_arr))
        throughput = (375.0 / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0

        peak_vram_kb = (torch.xpu.max_memory_allocated(0) / 1024.0) if is_xpu else 0.0

        record = {
            "scenario": name,
            "n_warmup": n_warmup,
            "n_measured": n_measured,
            "latency_ms": {
                "mean": round(mean_ms, 2),
                "median": round(median_ms, 2),
                "p95": round(p95_ms, 2),
                "min": round(min_ms, 2),
                "max": round(max_ms, 2),
                "std": round(std_ms, 2),
            },
            "candidates_per_second": round(throughput, 1),
            "selected_v": round(last_v, 3),
            "selected_w": round(last_w, 3),
            "score": round(last_score, 4),
            "status": last_status.name,
            "xpu_peak_memory_kb": round(peak_vram_kb, 1),
        }
        print(f"  {name:<40} | Mean: {mean_ms:6.2f} ms | P95: {p95_ms:6.2f} ms | Min/Max: {min_ms:.1f}/{max_ms:.1f} ms | Throughput: {throughput:6.1f} c/s")
        results_by_scenario.append(record)

    all_means = [r["latency_ms"]["mean"] for r in results_by_scenario]
    all_p95s = [r["latency_ms"]["p95"] for r in results_by_scenario]
    all_throughputs = [r["candidates_per_second"] for r in results_by_scenario]

    return {
        "backend_type": backend_type.value,
        "backend_name": backend_obj.name,
        "device_name": backend_obj.device_name,
        "grand_mean_ms": round(float(np.mean(all_means)), 2),
        "grand_median_ms": round(float(np.median(all_means)), 2),
        "grand_p95_ms": round(float(np.mean(all_p95s)), 2),
        "mean_candidates_per_second": round(float(np.mean(all_throughputs)), 1),
        "scenarios": results_by_scenario,
    }


def main() -> None:
    print("=" * 95)
    print("PHASE 13.20: REAL INTEL GPU (XPU) DWA VALIDATION & BENCHMARK SUITE")
    print("=" * 95)
    print(f"Host Hardware: Intel(R) Core(TM) Ultra 7 155H | GPU: Intel(R) Graphics | PyTorch: {torch.__version__}")

    scenarios = build_benchmark_scenarios()

    # 1. CPU Reference Benchmark
    # Note: Reference CPU is slow (~3.2s/iteration). We use 3 warmup and 10 measured to obtain deterministic baseline in reasonable time.
    print("\n--- [1/3] Benchmarking Reference CPU DWA ---")
    cpu_res = benchmark_backend_on_scenarios(
        DWABackendType.CPU, scenarios, n_warmup=3, n_measured=10
    )

    # 2. Vectorized CPU Benchmark (PyTorch SIMD on Host CPU)
    print("\n--- [2/3] Benchmarking Vectorized CPU DWA (100 measured iterations) ---")
    vec_res = benchmark_backend_on_scenarios(
        DWABackendType.VECTORIZED_CPU, scenarios, n_warmup=20, n_measured=100
    )

    # 3. Intel GPU Benchmark (xpu:0, Intel Arc integrated Graphics)
    print("\n--- [3/3] Benchmarking Native Intel GPU DWA (100 measured iterations on xpu:0) ---")
    intel_res = benchmark_backend_on_scenarios(
        DWABackendType.INTEL_GPU, scenarios, n_warmup=20, n_measured=100
    )

    # Summary report
    report = {
        "validation_metadata": {
            "title": "Real Intel GPU DWA Validation & Comparative Benchmark",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_model": "Intel(R) Core(TM) Ultra 7 155H",
            "gpu_model": "Intel(R) Graphics (Meteor Lake Arc integrated GPU)",
            "gpu_driver": "32.0.101.8831",
            "npu_model": "Intel(R) AI Boost (NPU 3720)",
            "pytorch_version": torch.__version__,
            "xpu_available": torch.xpu.is_available(),
            "xpu_device_name": torch.xpu.get_device_name(0),
            "workload": {
                "candidates": 375,
                "v_samples": 15,
                "omega_samples": 25,
                "horizon_s": 1.50,
                "dt_s": 0.10,
                "steps_per_trajectory": 15,
            },
        },
        "backends": {
            "CPU_Reference": cpu_res,
            "Vectorized_CPU": vec_res,
            "Intel_GPU": intel_res,
            "CUDA": {"available": False, "status": "Hardware absent on host laptop"},
            "NPU": {"available": False, "status": "NOT VALIDATED FOR DWA (Unsuitable for dynamic branching geometry)"},
        },
        "comparisons": {
            "intel_gpu_speedup_vs_cpu_ref": round(cpu_res["grand_mean_ms"] / max(0.001, intel_res["grand_mean_ms"]), 2),
            "vectorized_cpu_speedup_vs_cpu_ref": round(cpu_res["grand_mean_ms"] / max(0.001, vec_res["grand_mean_ms"]), 2),
            "intel_gpu_vs_vectorized_cpu_ratio": round(vec_res["grand_mean_ms"] / max(0.001, intel_res["grand_mean_ms"]), 2),
        },
    }

    out_file = os.path.join("reports", "dwa_intel_gpu_validation.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 105)
    print("FINAL BENCHMARK COMPARISON TABLE (100 measured iterations per scenario on GPU/Vec)")
    print("=" * 105)
    header = f"{'Scenario':<36} | {'CPU Ref (ms)':<13} | {'Vectorized (ms)':<16} | {'Intel GPU (ms)':<15} | {'GPU Speedup':<11}"
    print(header)
    print("-" * 105)
    for c_s, v_s, i_s in zip(cpu_res["scenarios"], vec_res["scenarios"], intel_res["scenarios"]):
        gpu_sp = c_s["latency_ms"]["mean"] / max(0.001, i_s["latency_ms"]["mean"])
        print(f"{c_s['scenario']:<36} | {c_s['latency_ms']['mean']:9.2f} ms | {v_s['latency_ms']['mean']:12.2f} ms | {i_s['latency_ms']['mean']:11.2f} ms | {gpu_sp:8.1f}x")
    print("=" * 105)
    print(f"Grand Mean Latency: CPU Ref = {cpu_res['grand_mean_ms']:.2f} ms | Vectorized CPU = {vec_res['grand_mean_ms']:.2f} ms | Intel GPU = {intel_res['grand_mean_ms']:.2f} ms")
    print(f"Real Intel GPU Acceleration vs Reference CPU: {report['comparisons']['intel_gpu_speedup_vs_cpu_ref']}x")
    print(f"Artifact saved to: {out_file}")
    print("=" * 105)


if __name__ == "__main__":
    main()
