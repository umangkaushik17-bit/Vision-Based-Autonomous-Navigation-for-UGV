"""Evidence consolidation script for Phase 13.13 SIL validation.

Generates:
1. aggregate_scenario_comparison.png - Multi-metric summary across all scenarios.
2. planner_latency_benchmark.png - Benchmark planner and step latency vs obstacle load.
3. prediction_error_vs_horizon.png - Prediction RMSE across horizons (0.25s to 2.00s).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "simulation/outputs/dynamic_sil"


def generate_aggregate_comparison(metrics_map: Dict[str, Dict[str, Any]]) -> str:
    """Generate a multi-panel comparison chart across all scenarios."""
    scenarios = sorted(metrics_map.keys())
    clearances = []
    tracking_rmses = []
    planner_latencies = []
    labels = []

    for sc in scenarios:
        m = metrics_map[sc]
        short_name = sc.replace("scenario_", "").replace("_metrics", "")
        labels.append(short_name)
        clearances.append(min(5.0, m.get("min_clearance_m", 5.0)))
        trk_rmse = m.get("detailed_metrics", {}).get("tracking", {}).get("position_rmse_m", 0.0)
        tracking_rmses.append(trk_rmse)
        plan_lat = m.get("planner_latency_mean_s", 0.0)
        planner_latencies.append(plan_lat)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("Phase 13.13 Dynamic SIL — Aggregate Scenario Metrics", fontsize=14, fontweight="bold")

    x = np.arange(len(scenarios))

    # Minimum Clearance
    ax1.bar(x, clearances, color="teal", alpha=0.85)
    ax1.axhline(0.4, color="red", linestyle="--", label="Critical Safety Threshold (0.4m)")
    ax1.set_ylabel("Min Clearance (m)", fontsize=10)
    ax1.set_title("Minimum Clearance to Obstacles (Capped at 5.0m)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    # Tracking Position RMSE
    ax2.bar(x, tracking_rmses, color="darkorange", alpha=0.85)
    ax2.set_ylabel("Tracking RMSE (m)", fontsize=10)
    ax2.set_title("Tracking Position RMSE (Timestamp-Aligned)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Planner Mean Latency
    ax3.bar(x, planner_latencies, color="purple", alpha=0.85)
    ax3.set_ylabel("Mean Latency (s)", fontsize=10)
    ax3.set_title("DWA Planner Mean Latency per Cycle (Pure Python)", fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "aggregate_scenario_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def generate_benchmark_plot(benchmark_json_path: str) -> str:
    """Generate planner latency vs load plot from benchmark summary."""
    if not os.path.exists(benchmark_json_path):
        print(f"Warning: {benchmark_json_path} does not exist.")
        return ""

    with open(benchmark_json_path, "r") as f:
        results = json.load(f)

    names = [r["case_name"] for r in results]
    obs = [r["obstacle_count"] for r in results]
    plan_means = [r["planner_mean_latency_s"] for r in results]
    plan_p95s = [r["planner_p95_latency_s"] for r in results]
    step_means = [r["step_mean_latency_s"] for r in results]
    rtfs = [r["real_time_factor"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase 13.13 Dynamic SIL — Performance Benchmark Characterization", fontsize=13, fontweight="bold")

    x = np.arange(len(names))
    width = 0.25

    # Latencies
    ax1.bar(x - width, plan_means, width, label="Planner Mean", color="royalblue")
    ax1.bar(x, plan_p95s, width, label="Planner P95", color="navy")
    ax1.bar(x + width, step_means, width, label="Step Mean", color="mediumpurple")
    ax1.set_ylabel("Latency (seconds)", fontsize=10)
    ax1.set_title("Planner and SIL Step Latencies (Pure Python)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Real Time Factor
    ax2.bar(x, rtfs, width=0.4, color="forestgreen", alpha=0.85)
    ax2.set_ylabel("Real-Time Factor (Sim Time / Wall Time)", fontsize=10)
    ax2.set_title("Real-Time Factor across Scenarios", fontsize=11)
    ax2.axhline(1.0, color="red", linestyle="--", label="Real-Time Boundary (1.0x)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "planner_latency_benchmark.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def generate_prediction_error_plot(metrics_map: Dict[str, Dict[str, Any]]) -> str:
    """Generate prediction error vs prediction horizon curve."""
    horizons = [0.25, 0.50, 1.00, 1.50, 2.00]
    horizon_keys = ["rmse_0_25s", "rmse_0_50s", "rmse_1_00s", "rmse_1_50s", "rmse_2_00s"]

    # Select dynamic scenarios with active predictions
    target_scenarios = [
        "scenario_b_parallel",
        "scenario_c_head_on",
        "scenario_d1_crossing_left_to_right",
        "scenario_d2_crossing_right_to_left",
        "scenario_e1_accelerating",
        "scenario_e2_decelerating",
        "scenario_h_multiple_obstacles",
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Obstacle Motion Prediction: Error vs. Horizon", fontsize=13, fontweight="bold")

    for sc_name in target_scenarios:
        for fname, data in metrics_map.items():
            if sc_name in fname:
                pred_metrics = data.get("detailed_metrics", {}).get("prediction", {})
                errors = [pred_metrics.get(k, 0.0) for k in horizon_keys]
                label = sc_name.replace("scenario_", "")
                ax.plot(horizons, errors, marker="o", linewidth=2.0, label=label)
                break

    ax.set_xlabel("Prediction Horizon (seconds)", fontsize=11)
    ax.set_ylabel("Prediction RMSE (meters)", fontsize=11)
    ax.set_title("Ground Truth Aligned Evaluation at t + h", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "prediction_error_vs_horizon.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main():
    metrics_files = glob.glob(os.path.join(OUTPUT_DIR, "*_metrics.json"))
    metrics_map = {}
    for mf in metrics_files:
        with open(mf, "r") as f:
            data = json.load(f)
            key = os.path.basename(mf).replace(".json", "")
            metrics_map[key] = data

    print(f"Loaded {len(metrics_map)} scenario metrics.")
    generate_aggregate_comparison(metrics_map)

    benchmark_json = os.path.join(OUTPUT_DIR, "benchmark_summary.json")
    generate_benchmark_plot(benchmark_json)

    generate_prediction_error_plot(metrics_map)
    print("All evidence plots generated successfully.")


if __name__ == "__main__":
    main()
