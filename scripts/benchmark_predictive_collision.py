"""Validation and performance benchmark harness for Milestone 13.8 Predictive Collision Checking.

Executes:
1. Validation Benchmark across 5 canonical scenarios:
   - Scenario A: UGV trajectory clearly avoids obstacle.
   - Scenario B: UGV trajectory physically intersects obstacle.
   - Scenario C: UGV approaches uncertainty region but does not physically collide.
   - Scenario D: Trajectory extends beyond prediction horizon (UNKNOWN_HORIZON).
   - Scenario E: Multiple obstacles with one eventually causing collision.
2. Latency and Scalability Performance Profiling:
   - 1 trajectory / 1 obstacle
   - 1 trajectory / 4 obstacles
   - Trajectory length scalability (5, 10, 20, 30 samples)
   - Obstacle count scalability (1, 2, 4, 8 obstacles)
   - Metrics: Mean, Median, P95, and Maximum latency
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionConfig,
    PredictiveCollisionResult,
    TrajectorySample,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
)
from ai.planning.dynamic.occupancy_layer import (
    DynamicOccupancyConfig,
    DynamicOccupancyLayer,
    DynamicRiskLevel,
)


def create_benchmark_obstacle(
    track_id: int,
    start_pos: Tuple[float, float],
    velocity: Tuple[float, float],
    size: Tuple[float, float] = (1.5, 0.8),
    yaw: float = 0.0,
    status: str = "TRACKING",
    horizon_s: float = 2.0,
    dt: float = 0.25,
    base_unc: float = 0.20,
    unc_growth: float = 0.15,
) -> PredictedObstacleTrajectory:
    """Construct deterministic obstacle trajectory for benchmarking."""
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
        t = step * dt
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


def generate_ugv_trajectory(
    speed: float = 1.0,
    duration_s: float = 2.0,
    n_samples: int = 20,
    yaw: float = 0.0,
) -> List[TrajectorySample]:
    """Generate linear UGV trajectory samples."""
    samples: List[TrajectorySample] = []
    times = np.linspace(0.0, duration_s, n_samples)
    for t in times:
        t_f = float(t)
        x = speed * math.cos(yaw) * t_f
        y = speed * math.sin(yaw) * t_f
        samples.append(TrajectorySample(time_offset=t_f, x=x, y=y, yaw=yaw, v=speed))
    return samples


def run_validation_scenarios() -> List[Dict[str, Any]]:
    """Execute Scenarios A through E deterministically and record telemetry."""
    results: List[Dict[str, Any]] = []
    checker = PredictiveCollisionChecker()

    # -------------------------------------------------------------------------
    # Scenario A: Clearly avoids obstacle
    # -------------------------------------------------------------------------
    obs_a = create_benchmark_obstacle(track_id=1, start_pos=(1.0, 3.0), velocity=(0.0, 0.0))
    layer_a = DynamicOccupancyLayer()
    layer_a.update_predictions([obs_a])
    traj_a = generate_ugv_trajectory(speed=1.0, duration_s=2.0, n_samples=20)

    t0 = time.perf_counter()
    res_a = checker.check_trajectory(traj_a, layer_a, [obs_a])
    lat_a_us = (time.perf_counter() - t0) * 1e6

    results.append({
        "scenario": "Scenario A (Clear Avoidance)",
        "physical_collision": res_a.collision,
        "risk_level": res_a.risk_classification.name,
        "unknown_horizon": res_a.unknown_horizon_encountered,
        "collision_time": res_a.collision_time,
        "min_clearance_m": round(res_a.minimum_clearance, 4),
        "status": res_a.status.value,
        "latency_us": round(lat_a_us, 1),
    })

    # -------------------------------------------------------------------------
    # Scenario B: Physical intersection
    # -------------------------------------------------------------------------
    obs_b = create_benchmark_obstacle(track_id=2, start_pos=(1.2, 0.0), velocity=(0.0, 0.0), size=(1.0, 1.0))
    layer_b = DynamicOccupancyLayer()
    layer_b.update_predictions([obs_b])
    traj_b = generate_ugv_trajectory(speed=1.0, duration_s=2.0, n_samples=20)

    t0 = time.perf_counter()
    res_b = checker.check_trajectory(traj_b, layer_b, [obs_b])
    lat_b_us = (time.perf_counter() - t0) * 1e6

    results.append({
        "scenario": "Scenario B (Physical Intersection)",
        "physical_collision": res_b.collision,
        "risk_level": res_b.risk_classification.name,
        "unknown_horizon": res_b.unknown_horizon_encountered,
        "collision_time": round(res_b.collision_time, 3) if res_b.collision_time else None,
        "min_clearance_m": round(res_b.minimum_clearance, 4),
        "status": res_b.status.value,
        "latency_us": round(lat_b_us, 1),
    })

    # -------------------------------------------------------------------------
    # Scenario C: Approaches uncertainty region without physical collision
    # -------------------------------------------------------------------------
    obs_c = create_benchmark_obstacle(track_id=3, start_pos=(1.0, 0.0), velocity=(0.0, 0.0), size=(1.0, 1.0), base_unc=0.8)
    layer_c = DynamicOccupancyLayer()
    layer_c.update_predictions([obs_c])
    # UGV runs along y = 0.95m (clearance ~ 0.175m, inside [0, 1*sigma] -> High Risk, clearance > 0)
    traj_c = [TrajectorySample(time_offset=float(t), x=float(t), y=0.95, yaw=0.0, v=1.0) for t in np.linspace(0, 2.0, 20)]

    t0 = time.perf_counter()
    res_c = checker.check_trajectory(traj_c, layer_c, [obs_c])
    lat_c_us = (time.perf_counter() - t0) * 1e6

    results.append({
        "scenario": "Scenario C (Uncertainty Penetration)",
        "physical_collision": res_c.collision,
        "risk_level": res_c.risk_classification.name,
        "unknown_horizon": res_c.unknown_horizon_encountered,
        "collision_time": round(res_c.collision_time, 3) if res_c.collision_time else None,
        "min_clearance_m": round(res_c.minimum_clearance, 4),
        "status": res_c.status.value,
        "latency_us": round(lat_c_us, 1),
    })

    # -------------------------------------------------------------------------
    # Scenario D: Trajectory extends beyond prediction horizon (2.5s > 2.0s)
    # -------------------------------------------------------------------------
    layer_d = DynamicOccupancyLayer(DynamicOccupancyConfig(max_horizon_s=2.0))
    traj_d = generate_ugv_trajectory(speed=1.0, duration_s=2.5, n_samples=25)

    t0 = time.perf_counter()
    res_d = checker.check_trajectory(traj_d, layer_d)
    lat_d_us = (time.perf_counter() - t0) * 1e6

    results.append({
        "scenario": "Scenario D (Beyond Horizon)",
        "physical_collision": res_d.collision,
        "risk_level": res_d.risk_classification.name,
        "unknown_horizon": res_d.unknown_horizon_encountered,
        "collision_time": round(res_d.collision_time, 3) if res_d.collision_time else None,
        "min_clearance_m": round(res_d.minimum_clearance, 4),
        "status": res_d.status.value,
        "latency_us": round(lat_d_us, 1),
    })

    # -------------------------------------------------------------------------
    # Scenario E: Multiple obstacles with one eventually causing collision
    # -------------------------------------------------------------------------
    obs_e1 = create_benchmark_obstacle(track_id=10, start_pos=(5.0, 5.0), velocity=(0.0, 0.0))
    obs_e2 = create_benchmark_obstacle(track_id=20, start_pos=(0.0, -3.0), velocity=(1.0, 0.0))
    obs_e3 = create_benchmark_obstacle(track_id=30, start_pos=(1.2, -2.0), velocity=(0.0, 2.0), size=(0.8, 0.8))  # Collides at t~1.0s
    obs_e4 = create_benchmark_obstacle(track_id=40, start_pos=(-3.0, 0.0), velocity=(-1.0, 0.0))

    layer_e = DynamicOccupancyLayer()
    layer_e.update_predictions([obs_e1, obs_e2, obs_e3, obs_e4])
    traj_e = generate_ugv_trajectory(speed=1.0, duration_s=2.0, n_samples=20)

    t0 = time.perf_counter()
    res_e = checker.check_trajectory(traj_e, layer_e, [obs_e1, obs_e2, obs_e3, obs_e4])
    lat_e_us = (time.perf_counter() - t0) * 1e6

    results.append({
        "scenario": "Scenario E (Multi-Obstacle Collision)",
        "physical_collision": res_e.collision,
        "risk_level": res_e.risk_classification.name,
        "unknown_horizon": res_e.unknown_horizon_encountered,
        "collision_time": round(res_e.collision_time, 3) if res_e.collision_time else None,
        "min_clearance_m": round(res_e.minimum_clearance, 4),
        "status": res_e.status.value,
        "latency_us": round(lat_e_us, 1),
    })

    return results


def profile_latencies(
    checker: PredictiveCollisionChecker,
    traj: List[TrajectorySample],
    layer: DynamicOccupancyLayer,
    tracks: List[PredictedObstacleTrajectory],
    n_iterations: int = 100,
) -> Dict[str, float]:
    """Measure mean, median, P95, and max latency over repeated runs."""
    # Warmup
    for _ in range(10):
        _ = checker.check_trajectory(traj, layer, tracks)

    times_us: List[float] = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = checker.check_trajectory(traj, layer, tracks)
        times_us.append((time.perf_counter() - t0) * 1e6)

    arr = np.array(times_us)
    return {
        "mean_us": float(np.mean(arr)),
        "median_us": float(np.median(arr)),
        "p95_us": float(np.percentile(arr, 95)),
        "max_us": float(np.max(arr)),
    }


def run_performance_benchmarks() -> Dict[str, Any]:
    """Execute comprehensive latency benchmarks across obstacle counts and trajectory lengths."""
    checker = PredictiveCollisionChecker()
    benchmarks: Dict[str, Any] = {}

    # Setup obstacles
    obs_list: List[PredictedObstacleTrajectory] = [
        create_benchmark_obstacle(1, (1.5, 0.5), (0.2, 0.1)),
        create_benchmark_obstacle(2, (2.0, -1.0), (-0.1, 0.3)),
        create_benchmark_obstacle(3, (3.0, 1.5), (0.0, -0.2)),
        create_benchmark_obstacle(4, (0.5, 2.0), (0.3, -0.1)),
        create_benchmark_obstacle(5, (4.0, -2.0), (0.1, 0.1)),
        create_benchmark_obstacle(6, (2.5, 3.0), (-0.2, -0.2)),
        create_benchmark_obstacle(7, (1.0, -3.0), (0.0, 0.4)),
        create_benchmark_obstacle(8, (3.5, 0.0), (-0.3, 0.0)),
    ]

    # Layer with 1, 2, 4, 8 obstacles
    layer_1 = DynamicOccupancyLayer()
    layer_1.update_predictions(obs_list[:1])

    layer_2 = DynamicOccupancyLayer()
    layer_2.update_predictions(obs_list[:2])

    layer_4 = DynamicOccupancyLayer()
    layer_4.update_predictions(obs_list[:4])

    layer_8 = DynamicOccupancyLayer()
    layer_8.update_predictions(obs_list[:8])

    # Trajectories of varying sample lengths
    traj_5 = generate_ugv_trajectory(duration_s=0.5, n_samples=5)
    traj_10 = generate_ugv_trajectory(duration_s=1.0, n_samples=10)
    traj_20 = generate_ugv_trajectory(duration_s=2.0, n_samples=20)
    traj_30 = generate_ugv_trajectory(duration_s=2.0, n_samples=30)

    # 1. Standard Conditions
    benchmarks["1_traj_1_obs_20_samples"] = profile_latencies(
        checker, traj_20, layer_1, obs_list[:1]
    )
    benchmarks["1_traj_4_obs_20_samples"] = profile_latencies(
        checker, traj_20, layer_4, obs_list[:4]
    )

    # 2. Scalability across trajectory lengths (with 4 obstacles)
    benchmarks["traj_length_5_samples"] = profile_latencies(
        checker, traj_5, layer_4, obs_list[:4]
    )
    benchmarks["traj_length_10_samples"] = profile_latencies(
        checker, traj_10, layer_4, obs_list[:4]
    )
    benchmarks["traj_length_20_samples"] = profile_latencies(
        checker, traj_20, layer_4, obs_list[:4]
    )
    benchmarks["traj_length_30_samples"] = profile_latencies(
        checker, traj_30, layer_4, obs_list[:4]
    )

    # 3. Scalability across obstacle counts (with 20 samples)
    benchmarks["obs_count_1"] = profile_latencies(
        checker, traj_20, layer_1, obs_list[:1]
    )
    benchmarks["obs_count_2"] = profile_latencies(
        checker, traj_20, layer_2, obs_list[:2]
    )
    benchmarks["obs_count_4"] = profile_latencies(
        checker, traj_20, layer_4, obs_list[:4]
    )
    benchmarks["obs_count_8"] = profile_latencies(
        checker, traj_20, layer_8, obs_list[:8]
    )

    return benchmarks


def print_validation_table(val_results: List[Dict[str, Any]]) -> None:
    """Format and print Validation Benchmark table."""
    print("=" * 115)
    print("MILESTONE 13.8 — VALIDATION BENCHMARK RESULTS (SCENARIOS A through E)")
    print("=" * 115)
    header = (
        f"{'Scenario':<34} | {'Physical':<8} | {'Risk':<11} | {'Unknown':<7} | "
        f"{'t_coll (s)':<10} | {'Clear (m)':<9} | {'Status':<22} | {'Latency':<9}"
    )
    print(header)
    print("-" * 115)
    for r in val_results:
        t_str = f"{r['collision_time']:.2f}" if r["collision_time"] is not None else "None"
        row = (
            f"{r['scenario']:<34} | {str(r['physical_collision']):<8} | {r['risk_level']:<11} | "
            f"{str(r['unknown_horizon']):<7} | {t_str:<10} | {r['min_clearance_m']:<9.4f} | "
            f"{r['status']:<22} | {r['latency_us']:>6.1f} us"
        )
        print(row)
    print("=" * 115)


def print_performance_table(perf_results: Dict[str, Any]) -> None:
    """Format and print Performance Latency table."""
    print("\n" + "=" * 85)
    print("MILESTONE 13.8 — PERFORMANCE & SCALABILITY LATENCY PROFILING (100 runs)")
    print("=" * 85)
    header = f"{'Configuration':<35} | {'Mean':<10} | {'Median':<10} | {'P95':<10} | {'Max':<10}"
    print(header)
    print("-" * 85)
    for name, metrics in perf_results.items():
        m_us = f"{metrics['mean_us']:.1f} us"
        med_us = f"{metrics['median_us']:.1f} us"
        p95_us = f"{metrics['p95_us']:.1f} us"
        max_us = f"{metrics['max_us']:.1f} us"
        print(f"{name:<35} | {m_us:<10} | {med_us:<10} | {p95_us:<10} | {max_us:<10}")
    print("=" * 85)


if __name__ == "__main__":
    val = run_validation_scenarios()
    print_validation_table(val)

    perf = run_performance_benchmarks()
    print_performance_table(perf)
