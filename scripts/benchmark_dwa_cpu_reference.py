"""Phase 2: Establish Trusted CPU Reference Benchmark.

Runs deterministic DWA benchmarks on reference CPU DWAPlanner.
Measures:
- candidate count (375)
- trajectory points per candidate (15)
- simulation horizon (1.5s)
- simulation dt (0.10s)
- DWA latencies: mean, median (P50), P95, P99, max
- candidates/second throughput
- CPU utilization (%)
- RAM usage (MB)
- selected linear velocity (v)
- selected angular velocity (omega)
- selected trajectory waypoints
- selected trajectory score
- collision result & planner state
Outputs machine-readable artifact: reports/dwa_cpu_reference_benchmark.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import psutil

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.planning.base import PlannerStatus, RobotState
from ai.planning.dynamic.collision_checker import PredictiveCollisionChecker
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
)
from ai.planning.dynamic.dwa_filter import DynamicPlanningConfig
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path
from ai.spatial_fusion.world_map import WorldTerrainMap
from scripts.benchmark_dwa_dynamic import create_world_map, make_obstacle


def run_benchmark_scenario(
    name: str,
    planner: DWAPlanner,
    state: RobotState,
    path: Path,
    costmap: WorldTerrainMap,
    dynamic_layer: Any = None,
    predicted_tracks: Any = None,
    dynamic_config: Any = None,
    n_iterations: int = 15,
) -> Dict[str, Any]:
    process = psutil.Process(os.getpid())

    # Warmup cycle
    planner.compute_trajectory(
        state, path, costmap,
        dynamic_layer=dynamic_layer,
        predicted_tracks=predicted_tracks,
        dynamic_config=dynamic_config,
    )

    latencies_ms: List[float] = []
    cpu_percents: List[float] = []

    last_traj = None
    last_v = 0.0
    last_w = 0.0
    last_status = PlannerStatus.SUCCESS
    last_diag: Dict[str, Any] = {}

    mem_before = process.memory_info().rss / (1024.0 * 1024.0)

    for _ in range(n_iterations):
        # Measure CPU utilization and timing
        t0 = time.perf_counter()
        traj, v, w, status, diag = planner.compute_trajectory(
            state, path, costmap,
            dynamic_layer=dynamic_layer,
            predicted_tracks=predicted_tracks,
            dynamic_config=dynamic_config,
        )
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        cpu_percents.append(process.cpu_percent(interval=None))

        last_traj = traj
        last_v = v
        last_w = w
        last_status = status
        last_diag = diag

    mem_after = process.memory_info().rss / (1024.0 * 1024.0)

    lat_arr = np.array(latencies_ms)
    mean_ms = float(np.mean(lat_arr))
    median_ms = float(np.median(lat_arr))
    p95_ms = float(np.percentile(lat_arr, 95))
    p99_ms = float(np.percentile(lat_arr, 99))
    max_ms = float(np.max(lat_arr))

    candidates_count = int(last_diag.get("number_of_candidates_evaluated", planner.v_samples * planner.omega_samples))
    pts_per_cand = int(round(planner.pred_horizon / planner.sim_dt))
    candidates_per_sec = (candidates_count / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0

    selected_points = []
    if last_traj is not None:
        for pt in last_traj.points:
            selected_points.append({
                "t": round(pt.t, 3),
                "x": round(pt.x, 4),
                "y": round(pt.y, 4),
                "yaw": round(pt.yaw, 4),
                "v": round(pt.v, 3),
                "omega": round(pt.omega, 3),
                "clearance_m": round(pt.clearance_m, 4),
            })

    record = {
        "scenario_name": name,
        "n_iterations": n_iterations,
        "candidate_count": candidates_count,
        "trajectory_points_per_candidate": pts_per_cand,
        "simulation_horizon_s": planner.pred_horizon,
        "simulation_dt_s": planner.sim_dt,
        "latency_ms": {
            "mean": round(mean_ms, 3),
            "median": round(median_ms, 3),
            "p95": round(p95_ms, 3),
            "p99": round(p99_ms, 3),
            "max": round(max_ms, 3),
            "raw_samples": [round(x, 2) for x in latencies_ms],
        },
        "candidates_per_second": round(candidates_per_sec, 1),
        "cpu_utilization_percent": round(float(np.mean(cpu_percents)), 2),
        "ram_usage_mb": round(mem_after, 2),
        "selected_linear_velocity_mps": round(last_v, 3),
        "selected_angular_velocity_radps": round(last_w, 3),
        "selected_trajectory_score": round(float(last_diag.get("best_score", 0.0)), 4),
        "collision_result": bool(last_diag.get("dynamic_collision", False)),
        "planner_state": last_status.name,
        "diagnostics": {
            "statically_valid": last_diag.get("statically_valid_candidates", 0),
            "valid_trajectories": last_diag.get("valid_trajectories", 0),
            "broad_phase_cleared": last_diag.get("broad_phase_cleared_count", 0),
            "expensive_checks": last_diag.get("expensive_collision_checks", 0),
            "dynamically_rejected": last_diag.get("dynamically_rejected_candidates", 0),
            "dynamically_penalized": last_diag.get("dynamically_penalized_candidates", 0),
            "min_clearance_m": last_diag.get("min_clearance_m", 0.0),
        },
        "selected_trajectory_points": selected_points,
    }
    print(
        f"  [DONE] {name:<40} | Mean: {mean_ms:6.2f}ms | P95: {p95_ms:6.2f}ms | "
        f"Throughput: {candidates_per_sec:7.1f} cand/s | Cmd: ({last_v:.2f}, {last_w:.2f})"
    )
    return record


def main() -> None:
    print("=" * 80)
    print("PHASE 2: ESTABLISHING TRUSTED CPU DWA REFERENCE BENCHMARK")
    print("=" * 80)

    planner = DWAPlanner()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    scenarios_results: List[Dict[str, Any]] = []

    # 1. Phase 12 Static Baseline
    wmap_base = create_world_map()
    scenarios_results.append(run_benchmark_scenario("Phase 12 Static Baseline", planner, state, path, wmap_base))

    # 2. Scenario A: No Dynamic Obstacles
    wmap_a = create_world_map()
    layer_a = DynamicOccupancyLayer()
    layer_a.update_predictions([])
    scenarios_results.append(run_benchmark_scenario(
        "Scenario A: No Dynamic Obstacles", planner, state, path, wmap_a,
        dynamic_layer=layer_a, predicted_tracks=[]
    ))

    # 3. Scenario B: Moving Obstacle Outside Corridor
    wmap_b = create_world_map()
    obs_b = make_obstacle(track_id=1, start_pos=(5.0, 5.0), velocity=(0.5, 0.5))
    layer_b = DynamicOccupancyLayer()
    layer_b.update_predictions([obs_b])
    scenarios_results.append(run_benchmark_scenario(
        "Scenario B: Moving Obstacle Outside Corridor", planner, state, path, wmap_b,
        dynamic_layer=layer_b, predicted_tracks=[obs_b]
    ))

    # 4. Scenario C: Moving Obstacle Crossing Selected Trajectory
    wmap_c = create_world_map()
    obs_c = make_obstacle(track_id=2, start_pos=(1.5, -1.0), velocity=(0.0, 1.0))
    layer_c = DynamicOccupancyLayer()
    layer_c.update_predictions([obs_c])
    scenarios_results.append(run_benchmark_scenario(
        "Scenario C: Moving Obstacle Crossing Trajectory", planner, state, path, wmap_c,
        dynamic_layer=layer_c, predicted_tracks=[obs_c]
    ))

    # 5. Scenario D: Moving Obstacle Crossing Candidate Subset
    wmap_d = create_world_map()
    obs_d = make_obstacle(track_id=3, start_pos=(2.0, 1.5), velocity=(-0.5, -0.5))
    layer_d = DynamicOccupancyLayer()
    layer_d.update_predictions([obs_d])
    scenarios_results.append(run_benchmark_scenario(
        "Scenario D: Moving Obstacle Crossing Candidate Subset", planner, state, path, wmap_d,
        dynamic_layer=layer_d, predicted_tracks=[obs_d]
    ))

    # 6. Scenario E: Four Spatiotemporally Distributed Moving Obstacles
    wmap_e = create_world_map()
    obs_e1 = make_obstacle(track_id=10, start_pos=(1.0, 2.0), velocity=(0.0, -0.8))
    obs_e2 = make_obstacle(track_id=11, start_pos=(2.5, -2.0), velocity=(0.0, 0.8))
    obs_e3 = make_obstacle(track_id=12, start_pos=(3.0, 1.5), velocity=(-0.4, -0.4))
    obs_e4 = make_obstacle(track_id=13, start_pos=(0.5, -1.5), velocity=(0.3, 0.3))
    all_e = [obs_e1, obs_e2, obs_e3, obs_e4]
    layer_e = DynamicOccupancyLayer()
    layer_e.update_predictions(all_e)
    scenarios_results.append(run_benchmark_scenario(
        "Scenario E: Four Moving Obstacles", planner, state, path, wmap_e,
        dynamic_layer=layer_e, predicted_tracks=all_e
    ))

    # 7. Scenario F: Uncertain Obstacle with Risk Penalty
    wmap_f = create_world_map()
    obs_f = make_obstacle(
        track_id=5, start_pos=(2.5, 0.8), velocity=(-0.2, 0.0),
        base_unc=0.35, unc_growth=0.20
    )
    layer_f = DynamicOccupancyLayer()
    layer_f.update_predictions([obs_f])
    scenarios_results.append(run_benchmark_scenario(
        "Scenario F: Uncertain Obstacle", planner, state, path, wmap_f,
        dynamic_layer=layer_f, predicted_tracks=[obs_f]
    ))

    # 8. Scenario G: Prediction Horizon Exceeded
    wmap_g = create_world_map()
    obs_g = make_obstacle(track_id=6, start_pos=(1.0, 0.0), velocity=(0.0, 0.0), horizon_s=0.60)
    layer_g = DynamicOccupancyLayer()
    layer_g.update_predictions([obs_g])
    cfg_g = DynamicPlanningConfig(unknown_horizon_policy="PENALIZE", unknown_horizon_penalty=0.45)
    scenarios_results.append(run_benchmark_scenario(
        "Scenario G: Prediction Horizon Exceeded", planner, state, path, wmap_g,
        dynamic_layer=layer_g, predicted_tracks=[obs_g], dynamic_config=cfg_g
    ))

    # 9. Scenario H: Static + Dynamic Combination
    wmap_h = create_world_map()
    for y_v in np.linspace(0.3, 1.5, 12):
        for x_v in np.linspace(0.5, 2.0, 15):
            r, c = wmap_h.world_to_grid(x_v, y_v)
            wmap_h.obstacle[r, c] = True
            wmap_h.cost[r, c] = 1.00
    obs_h = make_obstacle(track_id=7, start_pos=(1.2, -0.2), velocity=(0.0, 0.0), size=(0.6, 0.5))
    layer_h = DynamicOccupancyLayer()
    layer_h.update_predictions([obs_h])
    scenarios_results.append(run_benchmark_scenario(
        "Scenario H: Static + Dynamic Combination", planner, state, path, wmap_h,
        dynamic_layer=layer_h, predicted_tracks=[obs_h]
    ))

    # Compute aggregate baseline metrics
    all_means = [s["latency_ms"]["mean"] for s in scenarios_results]
    all_p95s = [s["latency_ms"]["p95"] for s in scenarios_results]
    all_throughputs = [s["candidates_per_second"] for s in scenarios_results]

    report = {
        "benchmark_metadata": {
            "target": "CPU Reference DWA",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "planner_class": "DWAPlanner",
            "language": "Python 3.10 / CPython",
            "candidate_count": planner.v_samples * planner.omega_samples,
            "v_samples": planner.v_samples,
            "omega_samples": planner.omega_samples,
            "horizon_s": planner.pred_horizon,
            "dt_s": planner.sim_dt,
            "timesteps": int(round(planner.pred_horizon / planner.sim_dt)),
        },
        "overall_summary": {
            "grand_mean_latency_ms": round(float(np.mean(all_means)), 2),
            "grand_median_latency_ms": round(float(np.median(all_means)), 2),
            "grand_p95_latency_ms": round(float(np.mean(all_p95s)), 2),
            "mean_candidates_per_second": round(float(np.mean(all_throughputs)), 1),
        },
        "scenarios": scenarios_results,
    }

    out_path = os.path.join("reports", "dwa_cpu_reference_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Grand Mean Latency: {report['overall_summary']['grand_mean_latency_ms']:.2f} ms")
    print(f"Mean Throughput:    {report['overall_summary']['mean_candidates_per_second']:.1f} candidates/sec")
    print(f"Reference benchmark saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
