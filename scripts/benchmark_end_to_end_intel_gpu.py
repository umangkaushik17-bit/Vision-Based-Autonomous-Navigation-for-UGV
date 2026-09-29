"""Phase 13.20: End-to-End Navigation Pipeline Benchmark on Intel GPU.

Measures the latency breakdown across all stages of the full navigation loop:
1. Camera input -> Perception inference (LightweightTraversabilityNet, FROZEN)
2. Costmap generation / BEV projection
3. Dynamic obstacle prediction & occupancy layer update
4. DWA trajectory search & scoring (Reference CPU vs Vectorized CPU vs Intel GPU vs CUDA)
5. Behavioral arbitration & command generation

Calculates:
- individual stage latencies (ms): perception, costmap, dynamic obstacle, DWA, behavior
- total navigation cycle latency (ms)
- navigation frequency (Hz)
- bottleneck percentage contribution

Outputs: reports/dwa_end_to_end_intel_gpu_validation.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.costmap import BEVTerrainCostmap
from ai.planning.backends import (
    CPUDWABackend,
    CUDADWABackend,
    DWABackendType,
    IntelGPUDWABackend,
    UnifiedDWAPlanner,
    VectorizedCPUDWABackend,
)
from ai.planning.base import GoalState, RobotState
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.path import Path
from ai.spatial_fusion.world_map import WorldTerrainMap
from ai.traversability.dense_predictor import DenseTraversabilityPredictor
from scripts.benchmark_dwa_dynamic import create_world_map, make_obstacle


def run_pipeline_benchmark(n_cycles: int = 10) -> Dict[str, Any]:
    print("=" * 95)
    print("STEP 10: END-TO-END AUTONOMOUS NAVIGATION PIPELINE BENCHMARK (INTEL GPU VALIDATION)")
    print("=" * 95)

    # 1. Initialize FROZEN LightweightTraversabilityNet Predictor
    checkpoint = "outputs/rellis3d/checkpoints/best.pt"
    print(f"[*] Initializing FROZEN Traversability Model ({checkpoint})...")
    predictor = DenseTraversabilityPredictor(checkpoint=checkpoint, device="cpu")

    # 2. Check Backends
    has_xpu = hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)()
    has_cuda = torch.cuda.is_available()

    print(f"[*] Hardware Status:")
    print(f"    Intel XPU Available: {has_xpu} ({torch.xpu.get_device_name(0) if has_xpu else 'N/A'})")
    print(f"    NVIDIA CUDA Available: {has_cuda}")

    # 3. Initialize Planners
    cpu_planner = UnifiedDWAPlanner(backend=DWABackendType.CPU)
    vec_planner = UnifiedDWAPlanner(backend=DWABackendType.VECTORIZED_CPU)
    intel_planner = UnifiedDWAPlanner(backend=DWABackendType.INTEL_GPU) if has_xpu else None
    cuda_planner = UnifiedDWAPlanner(backend=DWABackendType.CUDA) if has_cuda else None

    # 4. Setup Camera Frame & World Map Inputs
    img = np.full((360, 480, 3), 110, dtype=np.uint8)
    img[180:, 140:340] = [170, 160, 140]  # Drivable terrain corridor

    # 5. Global Path and Initial State
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.25, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    # Dynamic Obstacle (moving across path)
    obs = make_obstacle(track_id=1, start_pos=(2.0, 1.0), velocity=(0.0, -0.5))

    # Latency tracking dictionaries
    def make_tracker():
        return {
            "perception": [], "costmap": [], "dynamic_update": [],
            "dwa": [], "behavior_arbitration": [], "total_cycle": []
        }

    latencies_cpu = make_tracker()
    latencies_vec = make_tracker()
    latencies_intel = make_tracker() if intel_planner else None
    latencies_cuda = make_tracker() if cuda_planner else None

    # Warmup
    print("[*] Performing pipeline warmup...")
    _ = predictor.predict(img)
    wmap_warm = create_world_map()
    layer_warm = DynamicOccupancyLayer()
    layer_warm.update_predictions([obs])
    _ = cpu_planner.compute_trajectory(state, path, wmap_warm, dynamic_layer=layer_warm, predicted_tracks=[obs])
    _ = vec_planner.compute_trajectory(state, path, wmap_warm, dynamic_layer=layer_warm, predicted_tracks=[obs])
    if intel_planner:
        _ = intel_planner.compute_trajectory(state, path, wmap_warm, dynamic_layer=layer_warm, predicted_tracks=[obs])
        torch.xpu.synchronize()

    print(f"[*] Executing {n_cycles} measured navigation cycles per backend...")

    for cycle in range(n_cycles):
        # STAGE 1: Perception Inference (LightweightTraversabilityNet, FROZEN)
        t0 = time.perf_counter()
        res = predictor.predict(img)
        t_perc = (time.perf_counter() - t0) * 1000.0
        trav_prob = res.mask

        # STAGE 2: Costmap Update
        t0 = time.perf_counter()
        wmap = create_world_map()
        resized_trav = cv2.resize(trav_prob, (wmap.cost.shape[1], wmap.cost.shape[0]))
        wmap.cost = (1.0 - resized_trav).astype(np.float32)
        t_cost = (time.perf_counter() - t0) * 1000.0

        # STAGE 3: Dynamic Obstacle Update
        t0 = time.perf_counter()
        layer = DynamicOccupancyLayer()
        layer.update_predictions([obs])
        t_dyn = (time.perf_counter() - t0) * 1000.0

        # STAGE 4A: Reference CPU DWA
        t0 = time.perf_counter()
        traj_cpu, cmd_v_c, cmd_w_c, status_c, diag_c = cpu_planner.compute_trajectory(
            state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
        )
        t_dwa_cpu = (time.perf_counter() - t0) * 1000.0

        # STAGE 5A: Behavior Arbitration (CPU)
        t0 = time.perf_counter()
        decision_c = cpu_planner.behavioral_state_machine.evaluate(
            current_state=state, best_trajectory=traj_cpu, best_v=cmd_v_c, best_w=cmd_w_c,
            planner_status=status_c, dwa_diagnostics=diag_c, predicted_tracks=[obs], global_path=path,
        )
        t_beh_c = (time.perf_counter() - t0) * 1000.0
        t_total_c = t_perc + t_cost + t_dyn + t_dwa_cpu + t_beh_c

        latencies_cpu["perception"].append(t_perc)
        latencies_cpu["costmap"].append(t_cost)
        latencies_cpu["dynamic_update"].append(t_dyn)
        latencies_cpu["dwa"].append(t_dwa_cpu)
        latencies_cpu["behavior_arbitration"].append(t_beh_c)
        latencies_cpu["total_cycle"].append(t_total_c)

        # STAGE 4B: Vectorized CPU DWA
        t0 = time.perf_counter()
        traj_vec, cmd_v_v, cmd_w_v, status_v, diag_v = vec_planner.compute_trajectory(
            state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
        )
        t_dwa_vec = (time.perf_counter() - t0) * 1000.0

        # STAGE 5B: Behavior Arbitration (Vectorized)
        t0 = time.perf_counter()
        decision_v = vec_planner.behavioral_state_machine.evaluate(
            current_state=state, best_trajectory=traj_vec, best_v=cmd_v_v, best_w=cmd_w_v,
            planner_status=status_v, dwa_diagnostics=diag_v, predicted_tracks=[obs], global_path=path,
        )
        t_beh_v = (time.perf_counter() - t0) * 1000.0
        t_total_v = t_perc + t_cost + t_dyn + t_dwa_vec + t_beh_v

        latencies_vec["perception"].append(t_perc)
        latencies_vec["costmap"].append(t_cost)
        latencies_vec["dynamic_update"].append(t_dyn)
        latencies_vec["dwa"].append(t_dwa_vec)
        latencies_vec["behavior_arbitration"].append(t_beh_v)
        latencies_vec["total_cycle"].append(t_total_v)

        # STAGE 4C: Intel GPU DWA
        if intel_planner:
            t0 = time.perf_counter()
            traj_int, cmd_v_i, cmd_w_i, status_i, diag_i = intel_planner.compute_trajectory(
                state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
            )
            torch.xpu.synchronize()
            t_dwa_int = (time.perf_counter() - t0) * 1000.0

            # STAGE 5C: Behavior Arbitration (Intel GPU)
            t0 = time.perf_counter()
            decision_i = intel_planner.behavioral_state_machine.evaluate(
                current_state=state, best_trajectory=traj_int, best_v=cmd_v_i, best_w=cmd_w_i,
                planner_status=status_i, dwa_diagnostics=diag_i, predicted_tracks=[obs], global_path=path,
            )
            t_beh_i = (time.perf_counter() - t0) * 1000.0
            t_total_i = t_perc + t_cost + t_dyn + t_dwa_int + t_beh_i

            latencies_intel["perception"].append(t_perc)
            latencies_intel["costmap"].append(t_cost)
            latencies_intel["dynamic_update"].append(t_dyn)
            latencies_intel["dwa"].append(t_dwa_int)
            latencies_intel["behavior_arbitration"].append(t_beh_i)
            latencies_intel["total_cycle"].append(t_total_i)

        # STAGE 4D: CUDA DWA (if present)
        if cuda_planner:
            t0 = time.perf_counter()
            traj_cu, cmd_v_cu, cmd_w_cu, status_cu, diag_cu = cuda_planner.compute_trajectory(
                state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
            )
            torch.cuda.synchronize()
            t_dwa_cu = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            decision_cu = cuda_planner.behavioral_state_machine.evaluate(
                current_state=state, best_trajectory=traj_cu, best_v=cmd_v_cu, best_w=cmd_w_cu,
                planner_status=status_cu, dwa_diagnostics=diag_cu, predicted_tracks=[obs], global_path=path,
            )
            t_beh_cu = (time.perf_counter() - t0) * 1000.0
            t_total_cu = t_perc + t_cost + t_dyn + t_dwa_cu + t_beh_cu

            latencies_cuda["perception"].append(t_perc)
            latencies_cuda["costmap"].append(t_cost)
            latencies_cuda["dynamic_update"].append(t_dyn)
            latencies_cuda["dwa"].append(t_dwa_cu)
            latencies_cuda["behavior_arbitration"].append(t_beh_cu)
            latencies_cuda["total_cycle"].append(t_total_cu)

    def summarize(lat_dict: Dict[str, List[float]]) -> Dict[str, Any]:
        res = {}
        total_mean = float(np.mean(lat_dict["total_cycle"]))
        for k, v in lat_dict.items():
            m = float(np.mean(v))
            res[k] = {
                "mean_ms": round(m, 2),
                "p95_ms": round(float(np.percentile(v, 95)), 2),
                "pct_of_total": round((m / total_mean) * 100.0, 1),
            }
        res["effective_planner_frequency_hz"] = round(1000.0 / total_mean, 2)
        res["command_publication_frequency_hz"] = round(1000.0 / total_mean, 2)
        return res

    summary_cpu = summarize(latencies_cpu)
    summary_vec = summarize(latencies_vec)
    summary_intel = summarize(latencies_intel) if latencies_intel else None
    summary_cuda = summarize(latencies_cuda) if latencies_cuda else None

    report = {
        "benchmark_name": "Phase 13.20 End-to-End Navigation Pipeline Benchmark",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "cpu": "Intel Core Ultra 7 155H (16 cores, 22 threads)",
            "gpu": torch.xpu.get_device_name(0) if has_xpu else "None",
            "cuda_available": has_cuda,
            "xpu_available": has_xpu,
        },
        "traversability_model": "LightweightTraversabilityNet (FROZEN)",
        "pipeline_stages": [
            "Perception (LightweightTraversabilityNet)",
            "Costmap (BEV Projection & Aggregation)",
            "Dynamic Obstacle Tracking & Predictor",
            "DWA Local Trajectory Search",
            "Behavioral State Machine & Command Publishing",
        ],
        "with_reference_cpu_dwa": summary_cpu,
        "with_vectorized_cpu_dwa": summary_vec,
        "with_intel_gpu_dwa": summary_intel,
        "with_cuda_dwa": summary_cuda,
        "comparison": {
            "reference_cpu_cycle_ms": summary_cpu["total_cycle"]["mean_ms"],
            "vectorized_cpu_cycle_ms": summary_vec["total_cycle"]["mean_ms"],
            "intel_gpu_cycle_ms": summary_intel["total_cycle"]["mean_ms"] if summary_intel else None,
            "intel_gpu_speedup_over_cpu_ref": round(summary_cpu["total_cycle"]["mean_ms"] / summary_intel["total_cycle"]["mean_ms"], 2) if summary_intel else None,
            "vectorized_speedup_over_cpu_ref": round(summary_cpu["total_cycle"]["mean_ms"] / summary_vec["total_cycle"]["mean_ms"], 2),
        }
    }

    out_file = os.path.join("reports", "dwa_end_to_end_intel_gpu_validation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 105)
    header = f"{'Pipeline Stage':<28} | {'CPU Ref (ms)':<16} | {'Vectorized (ms)':<17} | {'Intel GPU (ms)':<18}"
    print(header)
    print("-" * 105)
    for stage in ["perception", "costmap", "dynamic_update", "dwa", "behavior_arbitration", "total_cycle"]:
        c_m = summary_cpu[stage]["mean_ms"]
        c_pct = summary_cpu[stage]["pct_of_total"]
        v_m = summary_vec[stage]["mean_ms"]
        v_pct = summary_vec[stage]["pct_of_total"]
        if summary_intel:
            i_m = summary_intel[stage]["mean_ms"]
            i_pct = summary_intel[stage]["pct_of_total"]
            print(f"{stage:<28} | {c_m:6.2f}ms ({c_pct:4.1f}%) | {v_m:6.2f}ms ({v_pct:4.1f}%) | {i_m:6.2f}ms ({i_pct:4.1f}%)")
        else:
            print(f"{stage:<28} | {c_m:6.2f}ms ({c_pct:4.1f}%) | {v_m:6.2f}ms ({v_pct:4.1f}%) | {'N/A':<18}")
    print("=" * 105)
    print(f"Cycle Latency & Frequency:")
    print(f"  Reference CPU:  {summary_cpu['total_cycle']['mean_ms']:7.1f} ms  ({summary_cpu['effective_planner_frequency_hz']:5.2f} Hz)")
    print(f"  Vectorized CPU: {summary_vec['total_cycle']['mean_ms']:7.1f} ms  ({summary_vec['effective_planner_frequency_hz']:5.2f} Hz)")
    if summary_intel:
        print(f"  Intel GPU:      {summary_intel['total_cycle']['mean_ms']:7.1f} ms  ({summary_intel['effective_planner_frequency_hz']:5.2f} Hz)")
        print(f"  Intel GPU Cycle Speedup: {report['comparison']['intel_gpu_speedup_over_cpu_ref']}x over CPU Reference")
    print(f"Report saved to: {out_file}")
    return report


if __name__ == "__main__":
    run_pipeline_benchmark()
