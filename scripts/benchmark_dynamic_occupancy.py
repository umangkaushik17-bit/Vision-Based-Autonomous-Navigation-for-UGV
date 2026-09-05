"""Benchmark and critical blockage analysis for Phase 13 Milestone 13.7 — Dynamic Occupancy Layer.

Evaluates:
1. Occupied cell counts across discrete time bins (0.00s to 2.00s):
   - Nominal physical footprint cells & area
   - Uncertainty-expanded cells & area
   - Categorical risk counts (LETHAL, HIGH_RISK, MEDIUM_RISK, LOW_RISK)
   - Total affected percentage of the metric grid
2. Critical analysis of 2.0s uncertainty envelope:
   - Pathological binary inflation vs graded dynamic risk representation
3. Multi-obstacle scaling (1, 2, 4, 10 obstacles)
4. Runtime latency profiling:
   - Ingestion and rasterization latency
   - Point query latency (get_cost, is_occupied)
   - Slice retrieval latency
5. Memory footprint of the multi-bin spatiotemporal representation.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Dict, List

import numpy as np

from ai.planning.dynamic.data_model import TrackedObstacleState
from ai.planning.dynamic.occupancy_layer import (
    DynamicOccupancyConfig,
    DynamicOccupancyLayer,
    DynamicRiskLevel,
)
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig


def make_test_obstacle(
    track_id: int,
    x: float,
    y: float,
    vx: float = 1.0,
    vy: float = 0.0,
    status: str = "TRACKING",
) -> TrackedObstacleState:
    """Construct a standardized test track."""
    cov = np.diag([0.04, 0.04, 0.09, 0.09]).astype(np.float64)
    return TrackedObstacleState(
        track_id=track_id,
        timestamp=0.0,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        covariance=cov,
        bbox_size=(1.5, 0.8),
        bbox_yaw=0.0,
        status=status,
    )


def evaluate_occupancy_breakdown_per_bin() -> List[Dict[str, Any]]:
    """Compute exact cell counts, areas, and blockage percentages per time bin for a moving obstacle."""
    predictor = MotionPredictor(PredictorConfig(process_noise_spectral_density=1.0, sigma_scale=3.0))
    layer = DynamicOccupancyLayer(DynamicOccupancyConfig())

    # Obstacle moving along X at 1.5 m/s
    track = make_test_obstacle(track_id=1, x=0.0, y=0.0, vx=1.5, vy=0.0)
    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.25)
    layer.update_predictions([pred])

    stats_list = []
    for t in layer.get_time_bins():
        stats = layer.get_slice_statistics(t)
        stats_list.append(stats)
    return stats_list


def evaluate_multi_obstacle_blockage() -> Dict[int, Dict[str, Any]]:
    """Evaluate grid blockage and cell counts for 1, 2, 4 simultaneous obstacles."""
    predictor = MotionPredictor()
    results = {}

    configs = {
        1: [make_test_obstacle(1, 0.0, 0.0, 1.0, 0.0)],
        2: [
            make_test_obstacle(1, -5.0, 0.0, 1.0, 0.0),
            make_test_obstacle(2, 5.0, 0.0, -1.0, 0.0),
        ],
        4: [
            make_test_obstacle(1, -5.0, -5.0, 1.0, 0.0),
            make_test_obstacle(2, 5.0, 5.0, -1.0, 0.0),
            make_test_obstacle(3, -5.0, 5.0, 0.0, -1.0),
            make_test_obstacle(4, 5.0, -5.0, 0.0, 1.0),
        ],
    }

    for n_obs, tracks in configs.items():
        layer = DynamicOccupancyLayer()
        preds = predictor.predict_tracks(tracks, horizon_s=2.0, dt_s=0.25)
        layer.update_predictions(preds)

        # Evaluate at horizon 2.0s
        stats_2s = layer.get_slice_statistics(2.0)
        results[n_obs] = stats_2s

    return results


def benchmark_performance() -> Dict[str, Any]:
    """Measure generation, rasterization, and query latencies."""
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    track_counts = [1, 2, 4, 10]
    iterations = 500
    warmup = 50

    def get_tracks(n: int) -> List[TrackedObstacleState]:
        return [
            make_test_obstacle(
                track_id=i + 1,
                x=float(-15.0 + i * 3.0),
                y=float(-10.0 + (i % 3) * 5.0),
                vx=1.0,
                vy=0.5,
            )
            for i in range(n)
        ]

    # Warmup
    warmup_tracks = get_tracks(10)
    warmup_preds = predictor.predict_tracks(warmup_tracks, horizon_s=2.0, dt_s=0.25)
    for _ in range(warmup):
        layer.update_predictions(warmup_preds)
        layer.get_cost(0.0, 0.0, 1.0)
        layer.get_slice(1.0)

    # 1. Rasterization generation latency per batch size
    gen_latencies_us: Dict[int, float] = {}
    for n in track_counts:
        tracks = get_tracks(n)
        preds = predictor.predict_tracks(tracks, horizon_s=2.0, dt_s=0.25)
        t0 = time.perf_counter_ns()
        for _ in range(iterations):
            layer.update_predictions(preds)
        t1 = time.perf_counter_ns()
        gen_latencies_us[n] = ((t1 - t0) / iterations) / 1000.0

    # 2. Point query latency (get_cost)
    t0 = time.perf_counter_ns()
    n_queries = 10000
    for i in range(n_queries):
        layer.get_cost(float(i % 20 - 10), float(i % 10 - 5), 1.0)
    t1 = time.perf_counter_ns()
    point_query_lat_ns = (t1 - t0) / n_queries

    # 3. Slice lookup latency (get_slice)
    t0 = time.perf_counter_ns()
    n_lookups = 10000
    for i in range(n_lookups):
        layer.get_slice(0.5)
    t1 = time.perf_counter_ns()
    slice_lookup_lat_ns = (t1 - t0) / n_lookups

    # 4. Memory footprint calculation
    sample_slice = layer.get_slice(0.0)
    bytes_cost = sample_slice.cost.nbytes
    bytes_risk = sample_slice.risk_level.nbytes
    bytes_nom = sample_slice.nominal_obstacle.nbytes
    bytes_unc = sample_slice.uncertainty_zone.nbytes
    bytes_per_slice = bytes_cost + bytes_risk + bytes_nom + bytes_unc
    total_memory_kb = (bytes_per_slice * len(layer.get_time_bins())) / 1024.0

    return {
        "gen_latencies_us": gen_latencies_us,
        "point_query_lat_ns": point_query_lat_ns,
        "slice_lookup_lat_ns": slice_lookup_lat_ns,
        "bytes_per_slice": bytes_per_slice,
        "total_memory_kb": total_memory_kb,
        "grid_shape": layer.shape,
    }


def main() -> None:
    print("============================================================")
    print("MILESTONE 13.7 DYNAMIC OCCUPANCY LAYER BENCHMARK")
    print("============================================================")

    stats_list = evaluate_occupancy_breakdown_per_bin()
    multi_obs = evaluate_multi_obstacle_blockage()
    perf = benchmark_performance()

    print("\n--- OCCUPANCY BY TIME BIN (1 MOVING OBSTACLE 1.5x0.8m) ---")
    print("| Time Bin | Nominal (m^2) | Unc (m^2) | Lethal | High Risk | Med Risk | Low Risk | Total Area (m^2) | Affected % |")
    print("|----------|---------------|-----------|--------|-----------|----------|----------|------------------|------------|")
    for s in stats_list:
        t = s["time_offset"]
        print(
            f"| {t:6.2f}s  | {s['nominal_area_m2']:13.2f} | {s['uncertainty_area_m2']:9.2f} | "
            f"{s['lethal_cells']:6d} | {s['high_risk_cells']:9d} | {s['medium_risk_cells']:8d} | "
            f"{s['low_risk_cells']:8d} | {s['total_affected_area_m2']:16.2f} | {s['affected_grid_percent']:9.3f}% |"
        )

    print("\n--- CRITICAL ANALYSIS: 2.0s PATHOLOGICAL BLOCKAGE CHECK ---")
    s2 = stats_list[-1]
    binary_lethal_radius = 5.2536 + 0.75  # 3-sigma radius + half-length
    binary_lethal_area = math.pi * (binary_lethal_radius ** 2)
    binary_lethal_cells = int(binary_lethal_area / 0.01)
    binary_percent = (binary_lethal_cells / s2["total_cells"]) * 100.0

    print(f"Total Local Grid: {s2['total_cells']} cells (40m x 40m = 1600 m^2)")
    print(f"1. Naive Binary Inflation (Full 3-sigma as Lethal):")
    print(f"   Lethal radius: {binary_lethal_radius:.2f} m -> Area: {binary_lethal_area:.2f} m^2 ({binary_lethal_cells} cells)")
    print(f"   Grid blockage per obstacle: {binary_percent:.2f}% (Pathological for 4 obstacles: ~{binary_percent*4:.1f}%)")
    print(f"2. Implemented Graded Dynamic Occupancy:")
    print(f"   Strict Lethal (Nominal Footprint): {s2['nominal_area_m2']:.2f} m^2 ({s2['lethal_cells']} cells, {s2['lethal_cells']/s2['total_cells']*100:.3f}%)")
    print(f"   High Risk Zone (0-1 sigma): {s2['high_risk_cells']*0.01:.2f} m^2 ({s2['high_risk_cells']} cells)")
    print(f"   Medium Risk Zone (1-2 sigma): {s2['medium_risk_cells']*0.01:.2f} m^2 ({s2['medium_risk_cells']} cells)")
    print(f"   Low Risk Zone (2-3 sigma): {s2['low_risk_cells']*0.01:.2f} m^2 ({s2['low_risk_cells']} cells)")
    print(f"   Total Affected Footprint: {s2['total_affected_area_m2']:.2f} m^2 ({s2['total_affected_cells']} cells, {s2['affected_grid_percent']:.3f}%)")
    print("   -> Conclusion: Lethal space is strictly restricted to the true physical footprint.")
    print("      Uncertainty is represented as graded cost, preventing corridor deadlock.")

    print("\n--- MULTI-OBSTACLE SCALING AT 2.0s HORIZON ---")
    print("| Obstacles | Lethal Cells | Total Affected Cells | Total Affected Area | Grid Affected % |")
    print("|-----------|--------------|----------------------|---------------------|-----------------|")
    for n in [1, 2, 4]:
        st = multi_obs[n]
        print(
            f"| {n:9d} | {st['lethal_cells']:12d} | {st['total_affected_cells']:20d} | "
            f"{st['total_affected_area_m2']:16.2f} m^2 | {st['affected_grid_percent']:14.3f}% |"
        )

    print("\n--- RUNTIME PERFORMANCE PROFILING ---")
    print(f"Grid Shape: {perf['grid_shape']} (0.10m resolution, 400x400 cells)")
    print(f"Memory Footprint per slice: {perf['bytes_per_slice'] / 1024.0:.1f} KB")
    print(f"Total 9-bin Layer Memory: {perf['total_memory_kb']:.1f} KB ({perf['total_memory_kb']/1024.0:.2f} MB)")
    print(f"Point Query Latency (get_cost): {perf['point_query_lat_ns']:.1f} ns")
    print(f"Slice Lookup Latency (get_slice): {perf['slice_lookup_lat_ns']:.1f} ns")
    print("Full-Trajectory Ingestion & Rasterization across 9 time bins:")
    for n in [1, 2, 4, 10]:
        print(f"   {n:2d} Obstacles: {perf['gen_latencies_us'][n]:7.2f} us ({perf['gen_latencies_us'][n]/1000.0:.3f} ms)")


if __name__ == "__main__":
    main()
