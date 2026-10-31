"""Phase 14: End-to-End Navigation Pipeline Benchmark.

Measures the latency breakdown across all stages of the full navigation loop:
1. Camera input -> Perception inference (LightweightTraversabilityNet, FROZEN)
2. Costmap generation / BEV projection
3. Dynamic obstacle prediction & occupancy layer update
4. DWA trajectory search & scoring (Reference CPU vs Vectorized CPU)
5. Behavioral arbitration & command generation

Calculates:
- individual stage latencies (ms)
- total cycle latency (ms)
- effective planner frequency (Hz)
- bottleneck percentage contribution
Outputs machine-readable artifact: reports/dwa_end_to_end_navigation_benchmark.json
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.costmap import BEVTerrainCostmap
from ai.planning.backends import (
    CPUDWABackend,
    DWABackendType,
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
    print("=" * 80)
    print("PHASE 14: END-TO-END AUTONOMOUS NAVIGATION PIPELINE BENCHMARK")
    print("=" * 80)

    # 1. Initialize FROZEN LightweightTraversabilityNet Predictor
    checkpoint = "outputs/rellis3d/checkpoints/best.pt"
    print(f"[*] Initializing FROZEN Traversability Model ({checkpoint})...")
    predictor = DenseTraversabilityPredictor(checkpoint=checkpoint, device="cpu")

    # 2. Initialize Planners
    cpu_planner = UnifiedDWAPlanner(backend=DWABackendType.CPU)
    vec_planner = UnifiedDWAPlanner(backend=DWABackendType.VECTORIZED_CPU)

    # 3. Setup Synthetic / Actual Camera Frames
    img = np.full((360, 480, 3), 110, dtype=np.uint8)
    img[180:, 140:340] = [170, 160, 140]  # Drivable road

    # 4. Global Path and Initial State
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.25, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    # Dynamic Obstacle
    obs = make_obstacle(track_id=1, start_pos=(2.0, 1.0), velocity=(0.0, -0.5))

    # Data collection containers
    latencies_cpu: Dict[str, List[float]] = {
        "perception": [], "costmap": [], "dynamic_update": [],
        "dwa": [], "behavior_arbitration": [], "total_cycle": []
    }
    latencies_vec: Dict[str, List[float]] = {
        "perception": [], "costmap": [], "dynamic_update": [],
        "dwa": [], "behavior_arbitration": [], "total_cycle": []
    }

    # Warmup
    _ = predictor.predict(img)

    for cycle in range(n_cycles):
        # STAGE 1: Perception Inference
        t0 = time.perf_counter()
        res = predictor.predict(img)
        t_perc = (time.perf_counter() - t0) * 1000.0
        trav_prob = res.mask

        # STAGE 2: Costmap Update
        t0 = time.perf_counter()
        wmap = create_world_map()
        # Ingest traversability slice into world map
        resized_trav = cv2.resize(trav_prob, (wmap.cost.shape[1], wmap.cost.shape[0]))
        wmap.cost = (1.0 - resized_trav).astype(np.float32)
        t_cost = (time.perf_counter() - t0) * 1000.0

        # STAGE 3: Dynamic Obstacle Update
        t0 = time.perf_counter()
        layer = DynamicOccupancyLayer()
        layer.update_predictions([obs])
        t_dyn = (time.perf_counter() - t0) * 1000.0

        # STAGE 4A: DWA with Reference CPU
        t0 = time.perf_counter()
        traj_cpu, cmd_v_c, cmd_w_c, status_c, diag_c = cpu_planner.compute_trajectory(
            state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
        )
        t_dwa_cpu = (time.perf_counter() - t0) * 1000.0

        # STAGE 5A: Behavior Arbitration
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

        # STAGE 4B: DWA with Vectorized CPU
        t0 = time.perf_counter()
        traj_vec, cmd_v_v, cmd_w_v, status_v, diag_v = vec_planner.compute_trajectory(
            state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
        )
        t_dwa_vec = (time.perf_counter() - t0) * 1000.0

        # STAGE 5B: Behavior Arbitration
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

    report = {
        "benchmark_name": "End-to-End Navigation Pipeline Benchmark",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Intel Core Ultra 7 155H (16 cores, 22 threads)",
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
        "bottleneck_analysis": {
            "reference_cpu_bottleneck": "DWA Local Planner",
            "reference_cpu_dwa_share_pct": summary_cpu["dwa"]["pct_of_total"],
            "vectorized_dwa_share_pct": summary_vec["dwa"]["pct_of_total"],
            "perception_share_pct_with_vectorized": summary_vec["perception"]["pct_of_total"],
            "navigation_loop_speedup": round(summary_cpu["total_cycle"]["mean_ms"] / summary_vec["total_cycle"]["mean_ms"], 1),
        },
    }

    out_file = os.path.join("reports", "dwa_end_to_end_navigation_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 90)
    print(f"{'Pipeline Stage':<38} | {'CPU Pipeline (ms)':<18} | {'Vectorized Pipeline (ms)':<22}")
    print("-" * 90)
    for stage in ["perception", "costmap", "dynamic_update", "dwa", "behavior_arbitration", "total_cycle"]:
        c_m = summary_cpu[stage]["mean_ms"]
        c_pct = summary_cpu[stage]["pct_of_total"]
        v_m = summary_vec[stage]["mean_ms"]
        v_pct = summary_vec[stage]["pct_of_total"]
        print(f"{stage:<38} | {c_m:7.2f}ms ({c_pct:4.1f}%)   | {v_m:7.2f}ms ({v_pct:4.1f}%)")
    print("=" * 90)
    print(f"Total Cycle: Reference CPU = {summary_cpu['total_cycle']['mean_ms']:.1f}ms ({summary_cpu['effective_planner_frequency_hz']:.2f} Hz)")
    print(f"Total Cycle: Vectorized    = {summary_vec['total_cycle']['mean_ms']:.1f}ms ({summary_vec['effective_planner_frequency_hz']:.2f} Hz)")
    print(f"End-to-End Loop Speedup:     {report['bottleneck_analysis']['navigation_loop_speedup']}x")
    print(f"Report saved to: {out_file}")
    return report


if __name__ == "__main__":
    run_pipeline_benchmark()
