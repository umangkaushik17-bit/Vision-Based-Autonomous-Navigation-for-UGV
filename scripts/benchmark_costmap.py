"""Benchmark evaluating costmap latency, throughput, and memory consumption across grid resolutions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np

from ai.costmap import (
    BEVCostmapFuser,
    CostmapMetadata,
    LayerGenerator,
    MotionCompensator,
    ObstacleInflator,
    TemporalCostmapAccumulator,
)
from ai.perception_fusion import SafetyAwarePerceptionResult
from ai.traversability.geometry import CameraExtrinsics, CameraIntrinsics

OUTPUT_DIR = Path("outputs/costmap/benchmarks")


def benchmark_resolution(res_m: float, dummy_perception: SafetyAwarePerceptionResult, rgb_frame: np.ndarray, iterations: int = 15) -> Dict[str, Any]:
    """Benchmark costmap operations for a specific cell resolution."""
    print(f"\n--- Testing Resolution: {res_m * 100:.0f} cm/cell ({res_m} m) ---")

    layer_gen = LayerGenerator(resolution_m=res_m)
    metadata = CostmapMetadata(
        resolution_m=res_m,
        shape=(layer_gen.n_x, layer_gen.n_y),
        x_range_m=layer_gen.x_range_m,
        y_range_m=layer_gen.y_range_m
    )
    fuser = BEVCostmapFuser()
    inflator = ObstacleInflator(resolution_m=res_m)
    compensator = MotionCompensator(metadata)
    temporal = TemporalCostmapAccumulator(metadata)

    # 1. Warmup
    layers = layer_gen.generate_layers(dummy_perception, rgb_image=rgb_frame)
    costmap = fuser.fuse_layers(layers, metadata)
    inflated = inflator.inflate_costmap(costmap)
    _ = compensator.warp_costmap(inflated, 0.1, 0.05, 0.02)
    _ = temporal.update(inflated, (0.1, 0.05, 0.02))

    # 2. Benchmark Component Latencies
    t_proj_list = []
    t_fuse_list = []
    t_infl_list = []
    t_warp_list = []
    t_temp_list = []
    t_total_list = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        layers = layer_gen.generate_layers(dummy_perception, rgb_image=rgb_frame)
        t1 = time.perf_counter()

        costmap = fuser.fuse_layers(layers, metadata)
        t2 = time.perf_counter()

        inflated = inflator.inflate_costmap(costmap)
        t3 = time.perf_counter()

        warped = compensator.warp_costmap(inflated, 0.1, 0.02, 0.01)
        t4 = time.perf_counter()

        accum = temporal.update(inflated, (0.1, 0.02, 0.01))
        t5 = time.perf_counter()

        t_proj_list.append((t1 - t0) * 1000.0)
        t_fuse_list.append((t2 - t1) * 1000.0)
        t_infl_list.append((t3 - t2) * 1000.0)
        t_warp_list.append((t4 - t3) * 1000.0)
        t_temp_list.append((t5 - t4) * 1000.0)
        t_total_list.append((t5 - t0) * 1000.0)

    # Memory measurement
    approx_memory_bytes = sum(arr.nbytes for arr in [
        costmap.cost, costmap.traversability, costmap.obstacle, costmap.hazard,
        costmap.unknown, costmap.confidence, costmap.slope_deg, costmap.roughness,
        costmap.height_var, costmap.observation_count
    ])

    mean_total = float(np.mean(t_total_list))
    fps = 1000.0 / mean_total if mean_total > 0 else 0.0

    print(f"  Grid Size         : {metadata.shape[0]}x{metadata.shape[1]} ({metadata.shape[0]*metadata.shape[1]} cells)")
    print(f"  Memory Footprint  : {approx_memory_bytes / 1024:.1f} KB")
    print(f"  Layer Projection  : {np.mean(t_proj_list):.2f} ms")
    print(f"  Multi-Layer Fusion: {np.mean(t_fuse_list):.2f} ms")
    print(f"  Obstacle Inflation: {np.mean(t_infl_list):.2f} ms")
    print(f"  Motion Alignment  : {np.mean(t_warp_list):.2f} ms")
    print(f"  Temporal Fusion   : {np.mean(t_temp_list):.2f} ms")
    print(f"  Total Costmap Cycle: {mean_total:.2f} ms ({fps:.1f} FPS)")

    return {
        "resolution_m": res_m,
        "grid_shape": list(metadata.shape),
        "total_cells": metadata.shape[0] * metadata.shape[1],
        "memory_kb": round(approx_memory_bytes / 1024.0, 2),
        "latencies_ms": {
            "projection_mean": round(float(np.mean(t_proj_list)), 2),
            "fusion_mean": round(float(np.mean(t_fuse_list)), 2),
            "inflation_mean": round(float(np.mean(t_infl_list)), 2),
            "motion_compensation_mean": round(float(np.mean(t_warp_list)), 2),
            "temporal_update_mean": round(float(np.mean(t_temp_list)), 2),
            "total_mean": round(mean_total, 2),
            "total_p50": round(float(np.percentile(t_total_list, 50)), 2),
            "total_p95": round(float(np.percentile(t_total_list, 95)), 2),
            "fps": round(fps, 2)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark costmap resolutions and pipeline latency")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("      SIH26126 PHASE 5: COSTMAP RESOLUTION & LATENCY BENCHMARK")
    print("=" * 70)

    # Create synthetic mock perception result (480, 640)
    h, w = 480, 640
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    trav = np.full((h, w), 0.85, dtype=np.float32)
    haz_mask = np.zeros((h, w), dtype=bool)
    haz_mask[300:360, 250:350] = True
    risk = np.zeros((h, w), dtype=np.float32)
    risk[haz_mask] = 1.0
    unk = np.zeros((h, w), dtype=bool)
    conf = np.full((h, w), 0.90, dtype=np.float32)

    dummy_perception = SafetyAwarePerceptionResult(
        final_traversability=trav,
        hazard_mask=haz_mask,
        hazard_classes=["WATER"],
        risk_map=risk,
        unknown_mask=unk,
        confidence_map=conf,
        detections=[],
        overlay=rgb,
        latency_ms=10.0
    )

    resolutions = [0.05, 0.10, 0.20]  # 5cm, 10cm, 20cm
    results = {}

    for res in resolutions:
        res_key = f"{int(res*100)}cm"
        results[res_key] = benchmark_resolution(res, dummy_perception, rgb, iterations=args.iterations)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Host AMD Ryzen CPU",
        "resolutions_tested": results,
        "recommended_default": "10cm (Balance of sub-decimeter fidelity and high-speed planning)"
    }

    report_path = OUTPUT_DIR / "costmap_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[OK] Costmap resolution benchmark report saved to: {report_path}")


if __name__ == "__main__":
    main()
