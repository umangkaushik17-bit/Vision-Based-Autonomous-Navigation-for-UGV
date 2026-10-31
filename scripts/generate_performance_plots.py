"""Generate diagnostic visualization plots for the CPU performance baseline and profiling results."""

from __future__ import annotations

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join("simulation", "outputs", "performance")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load JSON data
with open(os.path.join(OUTPUT_DIR, "performance_summary.json"), "r", encoding="utf-8") as f:
    summary = json.load(f)

with open(os.path.join(OUTPUT_DIR, "cpu_profile.json"), "r", encoding="utf-8") as f:
    profiles = json.load(f)

with open(os.path.join(OUTPUT_DIR, "memory_profile.json"), "r", encoding="utf-8") as f:
    memories = json.load(f)

scenarios = ["B01", "B06", "B07", "B12"]
sc_names = ["B01 (Nominal)", "B06 (Multi-Dyn)", "B07 (Mixed)", "B12 (Stress)"]

# ---------------------------------------------------------------------------
# Plot 1: Scenario Step Latencies & DWA Share
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

means = [summary["scenario_comparisons"][s]["mean_step_latency_ms"] / 1000.0 for s in scenarios]
p50s = [summary["scenario_comparisons"][s]["p50_step_latency_ms"] / 1000.0 for s in scenarios]
p95s = [summary["scenario_comparisons"][s]["p95_step_latency_ms"] / 1000.0 for s in scenarios]
maxs = [summary["scenario_comparisons"][s]["max_step_latency_ms"] / 1000.0 for s in scenarios]

x = np.arange(len(scenarios))
width = 0.20

ax1.bar(x - 1.5 * width, p50s, width, label="P50 Latency", color="#3498db")
ax1.bar(x - 0.5 * width, means, width, label="Mean Latency", color="#2ecc71")
ax1.bar(x + 0.5 * width, p95s, width, label="P95 Latency", color="#e67e22")
ax1.bar(x + 1.5 * width, maxs, width, label="Max Latency", color="#e74c3c")

ax1.axhline(0.10, color="red", linestyle="--", linewidth=1.5, label="10 Hz Real-Time Target (100 ms)")
ax1.set_ylabel("Step Latency (seconds)", fontsize=11, fontweight="bold")
ax1.set_title("CPU Step Latency Distribution across Canonical Scenarios", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(sc_names, fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper left")

# Sub-plot 2: DWA vs Non-DWA Share in B06
b06_subs = profiles["profiles"]["B06"]["subtimers_mean_ms"]
labels = ["DWA Local Planner", "Dynamic Occupancy", "Motion Prediction", "Kalman Tracking", "Perception / Replanner"]
values = [
    b06_subs["dwa_total_ms"],
    b06_subs["dynamic_occupancy_layer_ms"],
    b06_subs["motion_prediction_ms"],
    b06_subs["tracking_kalman_ms"],
    b06_subs["perception_generator_ms"] + b06_subs["global_replan_check_ms"] + b06_subs["behavioral_arbitration_ms"]
]
colors = ["#e74c3c", "#f39c12", "#f1c40f", "#3498db", "#95a5a6"]
explode = (0.08, 0.15, 0.2, 0.25, 0.3)

ax2.pie(values, labels=labels, autopct="%1.2f%%", startangle=140, colors=colors, explode=explode)
ax2.set_title("B06 Control Step Time Breakdown (Total: 19.86s)", fontsize=12, fontweight="bold")

plt.tight_layout()
fig_path1 = os.path.join(PLOTS_DIR, "latency_cpu_distribution.png")
plt.savefig(fig_path1, dpi=200)
plt.close()
print(f"[Plot Saved] {fig_path1}")

# ---------------------------------------------------------------------------
# Plot 2: Memory Footprint & Tracemalloc Stability
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

rss_before = [memories["memory_profiles"][s]["rss_before_mb"] for s in scenarios]
rss_10 = [memories["memory_profiles"][s]["rss_after_10_cycles_mb"] for s in scenarios]
rss_20 = [memories["memory_profiles"][s]["rss_after_20_cycles_mb"] for s in scenarios]
rss_peak = [memories["memory_profiles"][s]["peak_rss_mb"] for s in scenarios]

x = np.arange(len(scenarios))
width = 0.20

ax1.bar(x - 1.5 * width, rss_before, width, label="Before Run", color="#9b59b6")
ax1.bar(x - 0.5 * width, rss_10, width, label="After 10 Cycles", color="#3498db")
ax1.bar(x + 0.5 * width, rss_20, width, label="After 20 Cycles", color="#1abc9c")
ax1.bar(x + 1.5 * width, rss_peak, width, label="Peak RSS", color="#e74c3c")

ax1.set_ylabel("Process RSS (MB)", fontsize=11, fontweight="bold")
ax1.set_ylim([280, 350])
ax1.set_title("Memory Stability: Process RSS across Execution Cycles", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(sc_names, fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="lower right")

tm_curr = [memories["memory_profiles"][s]["tracemalloc_current_mb"] for s in scenarios]
tm_peak = [memories["memory_profiles"][s]["tracemalloc_peak_mb"] for s in scenarios]

ax2.bar(x - 0.15, tm_curr, 0.30, label="Tracemalloc Current", color="#27ae60")
ax2.bar(x + 0.15, tm_peak, 0.30, label="Tracemalloc Peak", color="#d35400")
ax2.set_ylabel("Tracemalloc Memory (MB)", fontsize=11, fontweight="bold")
ax2.set_title("Python Heap Allocation Bounds (Tracemalloc)", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(sc_names, fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper left")

plt.tight_layout()
fig_path2 = os.path.join(PLOTS_DIR, "memory_usage_stability.png")
plt.savefig(fig_path2, dpi=200)
plt.close()
print(f"[Plot Saved] {fig_path2}")

# ---------------------------------------------------------------------------
# Plot 3: Detailed DWA Bottleneck Breakdown
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# In B06, DWA inner functions
func_names = [
    "Footprint Collision\n(collision.py)",
    "Point Collision\n(collision.py)",
    "Polygon Clearance\n(collision_checker.py)",
    "SAT Polygon Intersect\n(data_model.py)",
    "Trajectory Rollout\n(dwa.py)",
    "Grid Coords / In-Bounds\n(world_map.py)",
    "All Other Ops\n(Scoring, Kinematics)",
]

# Total cumulative times from cProfile B06
func_times = [
    145.09,
    72.90,
    88.38,
    57.40,
    159.61 - 145.09,  # Rollout pure overhead
    29.20 + 17.31,
    496.55 - (145.09 + 72.90 + 88.38 + 57.40 + 14.52 + 46.51)
]

y_pos = np.arange(len(func_names))
bars = ax.barh(y_pos, func_times, color="#2980b9", edgecolor="#1a5276", alpha=0.85)

for bar, val in zip(bars, func_times):
    ax.text(val + 3.0, bar.get_y() + bar.get_height() / 2.0, f"{val:.1f}s ({val/496.55*100:.1f}%)",
            va="center", fontsize=9, fontweight="bold")

ax.set_yticks(y_pos)
ax.set_yticklabels(func_names, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Cumulative Execution Time over 25 Steps (Seconds)", fontsize=11, fontweight="bold")
ax.set_title("Root-Cause Bottleneck Breakdown inside DWA & Collision Checker (B06)", fontsize=12, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
ax.set_xlim([0, 200])

plt.tight_layout()
fig_path3 = os.path.join(PLOTS_DIR, "dwa_bottleneck_breakdown.png")
plt.savefig(fig_path3, dpi=200)
plt.close()
print(f"[Plot Saved] {fig_path3}")
