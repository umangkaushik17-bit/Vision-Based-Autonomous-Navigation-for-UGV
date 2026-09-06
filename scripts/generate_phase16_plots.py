"""Diagnostic plotting suite for Milestone 13.16 Full Closed-Loop Benchmark.

Generates the 8 mandatory visualization plots from telemetry data:
1. B01 baseline trajectory (X-Y plane, start, goal, path)
2. B04 static detour (UGV path, static wall, lateral detour)
3. B06 multi-dynamic obstacle trajectories (UGV path, 3 dynamic obstacle tracks, clearances)
4. B07 mixed static/dynamic trajectory (static gate pillars, crossing dynamic obstacles)
5. B08 dynamic bottleneck timeline (time vs UGV speed, behavior state, bottleneck clearance)
6. B11 sensor recovery timeline (pre-fault, blackout, recovery, post-recovery navigation)
7. B12 compound stress timeline (multi-panel timeline of pose, tracks, behavior, faults, clearance)
8. Performance step latency distribution (histogram, P50, P95, P99)
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join("simulation", "outputs", "phase16")
TELEMETRY_DIR = os.path.join(OUTPUT_DIR, "telemetry")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def load_telemetry(scenario_id: str, seed: int = 42) -> Optional[Dict[str, Any]]:
    path = os.path.join(TELEMETRY_DIR, f"{scenario_id}_seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_b01_baseline():
    data = load_telemetry("B01_flat_baseline", 42)
    if not data:
        return
    pts = data["telemetry_points"]
    xs = [p["robot_x"] for p in pts]
    ys = [p["robot_y"] for p in pts]

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, "b-", linewidth=2.5, label="UGV Path")
    plt.plot(xs[0], ys[0], "go", markersize=10, label="Start (0,0)")
    plt.plot(4.0, 0.0, "r*", markersize=14, label="Goal (4.0, 0)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("B01: Flat Open Terrain Baseline Trajectory (Nominal)", fontsize=12, fontweight="bold")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.xlim(-0.5, 5.0)
    plt.ylim(-1.5, 1.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "B01_baseline_trajectory.png"), dpi=200)
    plt.close()
    print("[+] Generated B01_baseline_trajectory.png")


def plot_b04_static_detour():
    data = load_telemetry("B04_static_detour", 42)
    if not data:
        return
    pts = data["telemetry_points"]
    xs = [p["robot_x"] for p in pts]
    ys = [p["robot_y"] for p in pts]

    plt.figure(figsize=(9, 5))
    # Plot static wall obstacles at x=2.0
    for y_pos in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 0.8]:
        circle = plt.Circle((2.0, y_pos), 0.35, color="red", alpha=0.5)
        plt.gca().add_patch(circle)

    plt.plot(xs, ys, "b-", linewidth=2.5, label="UGV Detour Path")
    plt.plot(xs[0], ys[0], "go", markersize=10, label="Start (0,0)")
    plt.plot(4.0, 0.0, "r*", markersize=14, label="Goal (4.0, 0)")
    plt.plot([2.0], [1.5], "m^", markersize=10, label="Bypass Corridor")

    # Annotate replan point if replan triggered
    for p in pts:
        if p.get("replan_triggered"):
            plt.plot(p["robot_x"], p["robot_y"], "ks", markersize=8, label="Global Replan Point")
            break

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("B04: Static Blockage & Lateral Detour Trajectory", fontsize=12, fontweight="bold")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.xlim(-0.5, 5.5)
    plt.ylim(-2.5, 3.0)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "B04_static_detour.png"), dpi=200)
    plt.close()
    print("[+] Generated B04_static_detour.png")


def plot_b06_multi_dynamic():
    data = load_telemetry("B06_multi_dynamic", 42)
    if not data:
        return
    pts = data["telemetry_points"]
    xs = [p["robot_x"] for p in pts]
    ys = [p["robot_y"] for p in pts]

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, "b-", linewidth=2.5, label="UGV Path")
    plt.plot(xs[0], ys[0], "go", markersize=10, label="Start (0,0)")
    plt.plot(4.5, 0.0, "r*", markersize=14, label="Goal (4.5, 0)")

    # Extract dynamic obstacle tracks from telemetry
    obs_tracks: Dict[str, List[Tuple[float, float]]] = {}
    for p in pts:
        for t in p.get("tracked_details", []):
            tid = str(t["track_id"])
            obs_tracks.setdefault(tid, []).append((t["x"], t["y"]))

    colors = ["orange", "purple", "brown"]
    for i, (tid, o_pts) in enumerate(obs_tracks.items()):
        if o_pts:
            oxs, oys = zip(*o_pts)
            c = colors[i % len(colors)]
            plt.plot(oxs, oys, "--", color=c, linewidth=1.8, label=f"Obstacle Track {tid}")
            plt.plot(oxs[-1], oys[-1], "o", color=c, markersize=7)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("B06: Multi-Dynamic Obstacles Trajectory (Crossing, Parallel, Approaching)", fontsize=12, fontweight="bold")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.xlim(-0.5, 5.5)
    plt.ylim(-2.5, 2.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "B06_multidynamic_trajectories.png"), dpi=200)
    plt.close()
    print("[+] Generated B06_multidynamic_trajectories.png")


def plot_b07_mixed_static_dynamic():
    data = load_telemetry("B07_mixed_static_dynamic", 42)
    if not data:
        return
    pts = data["telemetry_points"]
    xs = [p["robot_x"] for p in pts]
    ys = [p["robot_y"] for p in pts]

    plt.figure(figsize=(10, 5))
    # Static gate pillars
    circle1 = plt.Circle((2.5, 1.1), 0.40, color="gray", alpha=0.7, label="Static Gate Pillars")
    circle2 = plt.Circle((2.5, -1.1), 0.40, color="gray", alpha=0.7)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)

    plt.plot(xs, ys, "b-", linewidth=2.5, label="UGV Path")
    plt.plot(xs[0], ys[0], "go", markersize=10, label="Start (0,0)")
    plt.plot(4.5, 0.0, "r*", markersize=14, label="Goal (4.5, 0)")

    obs_tracks: Dict[str, List[Tuple[float, float]]] = {}
    for p in pts:
        for t in p.get("tracked_details", []):
            tid = str(t["track_id"])
            obs_tracks.setdefault(tid, []).append((t["x"], t["y"]))

    for tid, o_pts in obs_tracks.items():
        if o_pts:
            oxs, oys = zip(*o_pts)
            plt.plot(oxs, oys, "m--", linewidth=1.8, label=f"Dynamic Traffic ({tid})")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("B07: Mixed Static Gate + Dynamic Traffic Navigation", fontsize=12, fontweight="bold")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.xlim(-0.5, 5.5)
    plt.ylim(-2.5, 2.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "B07_mixed_static_dynamic.png"), dpi=200)
    plt.close()
    print("[+] Generated B07_mixed_static_dynamic.png")


def plot_b08_bottleneck_timeline():
    data = load_telemetry("B08_dynamic_bottleneck", 42)
    if not data:
        return
    pts = data["telemetry_points"]
    times = [p["sim_time"] for p in pts]
    speeds = [p["robot_v"] for p in pts]
    behaviors = [p["behavior_state"] for p in pts]
    clearances = [p["min_clearance_m"] for p in pts]

    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    ax1.plot(times, speeds, "b-", linewidth=2, label="UGV Linear Velocity (m/s)")
    ax1.plot(times, clearances, "g--", linewidth=1.5, label="Clearance to Obstacle (m)")
    ax1.set_xlabel("Simulation Time (s)")
    ax1.set_ylabel("Speed / Clearance", color="b")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Highlight behavior states
    state_map = {"PROCEED": 0, "AVOID": 1, "WAIT": 2, "STOP": 3}
    b_vals = [state_map.get(b, 0) for b in behaviors]

    ax2 = ax1.twinx()
    ax2.step(times, b_vals, "r:", where="post", linewidth=2, label="Behavior State")
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(["PROCEED", "AVOID", "WAIT", "STOP"])
    ax2.set_ylabel("Behavioral State", color="r")

    plt.title("B08: Dynamic Bottleneck Behavioral State Transition Timeline", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "B08_dynamic_bottleneck_timeline.png"), dpi=200)
    plt.close()
    print("[+] Generated B08_dynamic_bottleneck_timeline.png")


def plot_b11_recovery_timeline():
    data = load_telemetry("B11_sensor_recovery", 42)
    if not data:
        return
    pts = data["telemetry_points"]
    times = [p["sim_time"] for p in pts]
    track_counts = [p["tracked_count"] for p in pts]
    speeds = [p["robot_v"] for p in pts]

    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    # Fault window shading
    ax1.axvspan(3.5, 7.0, color="salmon", alpha=0.3, label="Blackout Fault Window (3.5s - 7.0s)")
    ax1.axvspan(7.0, 9.5, color="lightgreen", alpha=0.3, label="Track Reacquisition / Recovery (7.0s - 9.5s)")

    ax1.plot(times, speeds, "b-", linewidth=2, label="UGV Linear Velocity (m/s)")
    ax1.set_xlabel("Simulation Time (s)")
    ax1.set_ylabel("Speed (m/s)", color="b")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.step(times, track_counts, "m-", linewidth=2.2, where="post", label="Confirmed Track Count")
    ax2.set_ylabel("Track Count", color="m")
    ax2.set_ylim(-0.5, 2.5)

    plt.title("B11: Full-Horizon Sensor Blackout and Track Recovery Timeline", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "B11_recovery_timeline.png"), dpi=200)
    plt.close()
    print("[+] Generated B11_recovery_timeline.png")


def plot_b12_compound_timeline():
    data = load_telemetry("B12_compound_stress", 42)
    if not data:
        return
    pts = data["telemetry_points"]
    times = [p["sim_time"] for p in pts]
    xs = [p["robot_x"] for p in pts]
    track_counts = [p["tracked_count"] for p in pts]
    clearances = [p["min_clearance_m"] for p in pts]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Shading for replanner failure window
    ax1.axvspan(3.0, 5.5, color="gold", alpha=0.3, label="Replanner Failure Fault Window (3.0s - 5.5s)")
    ax1.plot(times, xs, "b-", linewidth=2, label="UGV X Position (Progress to Goal)")
    ax1.set_ylabel("X Position (m)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    ax2.plot(times, clearances, "g-", linewidth=1.8, label="Minimum Clearance (m)")
    ax2.axhline(0.35, color="red", linestyle="--", label="Critical Margin (0.35m)")
    ax2.set_xlabel("Simulation Time (s)")
    ax2.set_ylabel("Clearance (m)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.suptitle("B12: Compound Stress Mission Telemetry Timeline (Faults, Progress, Safety)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "B12_compound_stress_timeline.png"), dpi=200)
    plt.close()
    print("[+] Generated B12_compound_stress_timeline.png")


def plot_performance_distribution():
    perf_file = os.path.join(OUTPUT_DIR, "performance_results.json")
    if not os.path.exists(perf_file):
        return
    with open(perf_file, "r", encoding="utf-8") as f:
        perf_data = json.load(f)

    p95_lats = [r["step_latency_p95_ms"] for r in perf_data]
    mean_lats = [r["step_latency_mean_ms"] for r in perf_data]
    max_lats = [r["step_latency_max_ms"] for r in perf_data]

    plt.figure(figsize=(9, 4.5))
    bins = np.linspace(min(mean_lats) * 0.8, max(max_lats) * 1.1, 20)
    plt.hist(mean_lats, bins=bins, alpha=0.6, color="blue", label="Step Mean Latency (ms)")
    plt.hist(p95_lats, bins=bins, alpha=0.6, color="orange", label="Step P95 Latency (ms)")

    overall_p50 = float(np.median(p95_lats))
    overall_p95 = float(np.percentile(p95_lats, 95))
    plt.axvline(overall_p50, color="green", linestyle="--", linewidth=2, label=f"Median P95: {overall_p50:.1f}ms")
    plt.axvline(overall_p95, color="red", linestyle="--", linewidth=2, label=f"P95 of P95s: {overall_p95:.1f}ms")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("Control Cycle Step Latency Distribution (P50, P95, Max)", fontsize=12, fontweight="bold")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Scenario Run Count")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "performance_latency_distribution.png"), dpi=200)
    plt.close()
    print("[+] Generated performance_latency_distribution.png")


def generate_all_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n[*] Generating Phase 13.16 Diagnostic Visualizations...")
    plot_b01_baseline()
    plot_b04_static_detour()
    plot_b06_multi_dynamic()
    plot_b07_mixed_static_dynamic()
    plot_b08_bottleneck_timeline()
    plot_b11_recovery_timeline()
    plot_b12_compound_timeline()
    plot_performance_distribution()
    print("[+] Diagnostic plots generated in:", PLOTS_DIR)


if __name__ == "__main__":
    generate_all_plots()
