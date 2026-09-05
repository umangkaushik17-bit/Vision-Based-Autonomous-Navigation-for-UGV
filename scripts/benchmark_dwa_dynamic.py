"""Comprehensive Benchmark Suite for Milestone 13.9:
Dynamic Predictive Collision Integration into DWA.

Benchmarks deterministic scenarios:
  Phase 12 Baseline: Static DWA (no dynamic layer)
  Scenario A: No dynamic obstacles (empty dynamic layer)
  Scenario B: Moving obstacle outside candidate corridor (broad-phase skip validation)
  Scenario C: Moving obstacle crossing selected trajectory (evasion maneuver)
  Scenario D: Moving obstacle crossing a subset of candidates
  Scenario E: Four moving obstacles (spatiotemporally distributed)
  Scenario F: Uncertain obstacle with no physical collision (risk penalty validation)
  Scenario G: Prediction horizon exceeded (unknown horizon handling)
  Scenario H: Static + dynamic obstacle combination (conservative composition)

Measures:
  - candidate count
  - broad-phase cleared count
  - exact collision-check invocations
  - mean cycle latency (ms)
  - median cycle latency (ms)
  - P95 cycle latency (ms)
  - max cycle latency (ms)
  - selected command (v, w)
  - dynamic rejection / penalty count
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai.planning.base import PlannerStatus, RobotState
from ai.planning.dynamic.collision_checker import PredictiveCollisionChecker
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
)
from ai.planning.dynamic.dwa_filter import DynamicBroadPhaseFilter, DynamicPlanningConfig
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path
from ai.spatial_fusion.world_map import WorldTerrainMap


def make_obstacle(
    track_id: int,
    start_pos: Tuple[float, float],
    velocity: Tuple[float, float],
    size: Tuple[float, float] = (0.8, 0.6),
    yaw: float = 0.0,
    status: str = "TRACKING",
    horizon_s: float = 2.0,
    dt: float = 0.10,
    base_unc: float = 0.15,
    unc_growth: float = 0.10,
) -> PredictedObstacleTrajectory:
    cov = np.diag([0.04, 0.04, 0.01, 0.01]).astype(np.float64)
    state = TrackedObstacleState(
        track_id=track_id,
        timestamp=0.0,
        x=float(start_pos[0]),
        y=float(start_pos[1]),
        vx=float(velocity[0]),
        vy=float(velocity[1]),
        covariance=cov,
        status=status,
        is_confirmed=True if status == "TRACKING" else False,
        bbox_size=size,
        bbox_yaw=yaw,
    )

    n_steps = int(round(horizon_s / dt))
    waypoints: List[TrajectoryWaypoint] = []
    vx, vy = velocity
    x0, y0 = start_pos

    for step in range(n_steps + 1):
        t = round(step * dt, 4)
        x = x0 + vx * t
        y = y0 + vy * t
        r_unc = base_unc + unc_growth * t
        cov_2x2 = np.diag([(r_unc / 3.0) ** 2, (r_unc / 3.0) ** 2]).astype(np.float64)
        waypoints.append(
            TrajectoryWaypoint(
                time_offset=t,
                timestamp=t,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                covariance_2d=cov_2x2,
                uncertainty_radius=r_unc,
            )
        )

    polygons = [
        compute_obb_corners((wp.x, wp.y), size, yaw, margin_m=wp.uncertainty_radius)
        for wp in waypoints
    ]

    return PredictedObstacleTrajectory(
        track_id=track_id,
        current_state=state,
        horizon_s=horizon_s,
        dt=dt,
        waypoints=waypoints,
        bounding_polygon_per_step=polygons,
    )


def create_world_map(bounds: Tuple[float, float] = (-10.0, 10.0)) -> WorldTerrainMap:
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=bounds, y_bounds_m=bounds)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    return wmap


def benchmark_scenario(
    name: str,
    planner: DWAPlanner,
    state: RobotState,
    path: Path,
    costmap: WorldTerrainMap,
    dynamic_layer: Optional[DynamicOccupancyLayer] = None,
    predicted_tracks: Optional[List[PredictedObstacleTrajectory]] = None,
    dynamic_config: Optional[DynamicPlanningConfig] = None,
    n_iterations: int = 10,
) -> Dict[str, Any]:
    """Run benchmark iterations and collect performance & diagnostic statistics."""
    # Warmup
    for _ in range(1):
        planner.compute_trajectory(
            state, path, costmap,
            dynamic_layer=dynamic_layer,
            predicted_tracks=predicted_tracks,
            dynamic_config=dynamic_config,
        )

    latencies: List[float] = []
    last_diag: Dict[str, Any] = {}
    last_v: float = 0.0
    last_w: float = 0.0
    last_status = PlannerStatus.SUCCESS

    for _ in range(n_iterations):
        t0 = time.perf_counter()
        traj, v, w, status, diag = planner.compute_trajectory(
            state, path, costmap,
            dynamic_layer=dynamic_layer,
            predicted_tracks=predicted_tracks,
            dynamic_config=dynamic_config,
        )
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        last_diag = diag
        last_v = v
        last_w = w
        last_status = status

    lat_arr = np.array(latencies)
    res = {
        "name": name,
        "n_iterations": n_iterations,
        "mean_ms": float(np.mean(lat_arr)),
        "median_ms": float(np.median(lat_arr)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "max_ms": float(np.max(lat_arr)),
        "selected_v": round(last_v, 3),
        "selected_w": round(last_w, 3),
        "status": last_status.name,
        "candidates_evaluated": last_diag.get("number_of_candidates_evaluated", 375),
        "broad_phase_cleared": last_diag.get("broad_phase_cleared_count", 0),
        "exact_checks": last_diag.get("number_of_expensive_collision_checker_invocations", 0),
        "dynamically_rejected": last_diag.get("dynamically_rejected_candidates", 0),
        "dynamically_penalized": last_diag.get("dynamically_penalized_candidates", 0),
        "unknown_candidates": last_diag.get("unknown_horizon_candidates", 0),
        "dynamic_risk": last_diag.get("dynamic_risk", "FREE"),
        "best_score": last_diag.get("best_score", 0.0),
    }
    print(f"  [DONE] {name}: mean={res['mean_ms']:.1f}ms, exact_checks={res['exact_checks']}, cleared={res['broad_phase_cleared']}", flush=True)
    return res


def run_all_benchmarks() -> List[Dict[str, Any]]:
    planner = DWAPlanner()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    results: List[Dict[str, Any]] = []

    # 1. Phase 12 Static Baseline (no dynamic inputs)
    wmap_base = create_world_map()
    res_base = benchmark_scenario("Phase 12 Static Baseline", planner, state, path, wmap_base)
    results.append(res_base)

    # 2. Scenario A: No dynamic obstacles (empty dynamic layer)
    wmap_a = create_world_map()
    layer_a = DynamicOccupancyLayer()
    layer_a.update_predictions([])
    res_a = benchmark_scenario(
        "Scenario A: No Dynamic Obstacles",
        planner, state, path, wmap_a,
        dynamic_layer=layer_a, predicted_tracks=[],
    )
    results.append(res_a)

    # 3. Scenario B: Moving obstacle outside candidate corridor
    wmap_b = create_world_map()
    obs_b = make_obstacle(track_id=1, start_pos=(5.0, 5.0), velocity=(0.5, 0.5))
    layer_b = DynamicOccupancyLayer()
    layer_b.update_predictions([obs_b])
    res_b = benchmark_scenario(
        "Scenario B: Moving Outside Corridor",
        planner, state, path, wmap_b,
        dynamic_layer=layer_b, predicted_tracks=[obs_b],
    )
    results.append(res_b)

    # 4. Scenario C: Moving obstacle crossing selected trajectory
    wmap_c = create_world_map()
    obs_c = make_obstacle(track_id=2, start_pos=(1.0, 1.0), velocity=(0.0, -1.0), size=(0.7, 0.5))
    layer_c = DynamicOccupancyLayer()
    layer_c.update_predictions([obs_c])
    res_c = benchmark_scenario(
        "Scenario C: Moving Crossing Selected Traj",
        planner, state, path, wmap_c,
        dynamic_layer=layer_c, predicted_tracks=[obs_c],
    )
    results.append(res_c)

    # 5. Scenario D: Moving obstacle crossing subset of candidates
    wmap_d = create_world_map()
    obs_d = make_obstacle(track_id=3, start_pos=(1.5, -0.9), velocity=(0.1, 0.3), size=(0.6, 0.5))
    layer_d = DynamicOccupancyLayer()
    layer_d.update_predictions([obs_d])
    res_d = benchmark_scenario(
        "Scenario D: Moving Crossing Subset of Trajs",
        planner, state, path, wmap_d,
        dynamic_layer=layer_d, predicted_tracks=[obs_d],
    )
    results.append(res_d)

    # 6. Scenario E: Four moving obstacles
    wmap_e = create_world_map()
    obs_e1 = make_obstacle(track_id=10, start_pos=(1.5, 1.2), velocity=(0.0, -0.6))
    obs_e2 = make_obstacle(track_id=20, start_pos=(2.0, -1.0), velocity=(-0.2, 0.4))
    obs_e3 = make_obstacle(track_id=30, start_pos=(3.0, 0.5), velocity=(-0.5, 0.0))
    obs_e4 = make_obstacle(track_id=40, start_pos=(4.0, 4.0), velocity=(0.2, 0.2))
    layer_e = DynamicOccupancyLayer()
    layer_e.update_predictions([obs_e1, obs_e2, obs_e3, obs_e4])
    res_e = benchmark_scenario(
        "Scenario E: Four Moving Obstacles",
        planner, state, path, wmap_e,
        dynamic_layer=layer_e, predicted_tracks=[obs_e1, obs_e2, obs_e3, obs_e4],
    )
    results.append(res_e)

    # 7. Scenario F: Uncertain obstacle with no physical collision
    wmap_f = create_world_map()
    obs_f = make_obstacle(
        track_id=5, start_pos=(1.2, 0.70), velocity=(0.0, 0.0), size=(0.5, 0.4),
        base_unc=0.35, unc_growth=0.20
    )
    layer_f = DynamicOccupancyLayer()
    layer_f.update_predictions([obs_f])
    res_f = benchmark_scenario(
        "Scenario F: Uncertain Obstacle (Risk Penalty)",
        planner, state, path, wmap_f,
        dynamic_layer=layer_f, predicted_tracks=[obs_f],
    )
    results.append(res_f)

    # 8. Scenario G: Prediction horizon exceeded
    wmap_g = create_world_map()
    obs_g = make_obstacle(
        track_id=6, start_pos=(1.0, 0.0), velocity=(0.0, 0.0), horizon_s=0.60
    )
    layer_g = DynamicOccupancyLayer()
    layer_g.update_predictions([obs_g])
    cfg_g = DynamicPlanningConfig(unknown_horizon_policy="PENALIZE", unknown_horizon_penalty=0.45)
    res_g = benchmark_scenario(
        "Scenario G: Prediction Horizon Exceeded",
        planner, state, path, wmap_g,
        dynamic_layer=layer_g, predicted_tracks=[obs_g], dynamic_config=cfg_g,
    )
    results.append(res_g)

    # 9. Scenario H: Static + dynamic obstacle combination
    wmap_h = create_world_map()
    # Add static wall on left side
    for y_v in np.linspace(0.3, 1.5, 12):
        for x_v in np.linspace(0.5, 2.0, 15):
            r, c = wmap_h.world_to_grid(x_v, y_v)
            wmap_h.obstacle[r, c] = True
            wmap_h.cost[r, c] = 1.00
    obs_h = make_obstacle(track_id=7, start_pos=(1.2, -0.2), velocity=(0.0, 0.0), size=(0.6, 0.5))
    layer_h = DynamicOccupancyLayer()
    layer_h.update_predictions([obs_h])
    res_h = benchmark_scenario(
        "Scenario H: Static + Dynamic Combination",
        planner, state, path, wmap_h,
        dynamic_layer=layer_h, predicted_tracks=[obs_h],
    )
    results.append(res_h)

    return results


def print_results(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 115)
    print("MILESTONE 13.9: DWA PREDICTIVE COLLISION INTEGRATION BENCHMARK RESULTS")
    print("=" * 115)
    header = (
        f"{'Scenario':<36} | {'Mean (ms)':<9} | {'P95 (ms)':<8} | {'Max (ms)':<8} | "
        f"{'Exact Checks':<12} | {'BP Cleared':<10} | {'Dyn Rej':<7} | {'Cmd (v, w)':<12}"
    )
    print(header)
    print("-" * 115)
    for r in results:
        cmd_str = f"({r['selected_v']:.2f}, {r['selected_w']:.2f})"
        line = (
            f"{r['name']:<36} | {r['mean_ms']:<9.2f} | {r['p95_ms']:<8.2f} | {r['max_ms']:<8.2f} | "
            f"{r['exact_checks']:<12} | {r['broad_phase_cleared']:<10} | {r['dynamically_rejected']:<7} | {cmd_str:<12}"
        )
        print(line)
    print("=" * 115 + "\n")


if __name__ == "__main__":
    results = run_all_benchmarks()
    print_results(results)
