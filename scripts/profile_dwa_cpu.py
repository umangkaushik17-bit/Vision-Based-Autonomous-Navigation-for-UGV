"""Phase 3: CPU DWA Profiling Script.

Uses cProfile and pstats to collect exact function-level wall-clock time breakdown
for DWAPlanner across:
1. Static scenario (footprint collision, unicycle rollout, costmap lookup, scoring)
2. Dynamic scenario (predictive dynamic collision, SAT polygon check, polygon clearance)
Saves profile stats and generates hotspot classification.
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.planning.base import RobotState
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path
from scripts.benchmark_dwa_dynamic import create_world_map, make_obstacle


def profile_dwa() -> None:
    planner = DWAPlanner()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    # Scenario 1: Static only (3 iterations)
    wmap_base = create_world_map()
    # Add some static obstacles
    for y_v in np.linspace(0.3, 1.5, 10):
        for x_v in np.linspace(0.5, 2.0, 10):
            r, c = wmap_base.world_to_grid(x_v, y_v)
            wmap_base.obstacle[r, c] = True
            wmap_base.cost[r, c] = 1.00

    print("Profiling Static DWA...")
    pr_static = cProfile.Profile()
    pr_static.enable()
    for _ in range(3):
        planner.compute_trajectory(state, path, wmap_base)
    pr_static.disable()

    s_out = io.StringIO()
    ps_static = pstats.Stats(pr_static, stream=s_out).sort_stats("cumulative")
    ps_static.print_stats(30)
    static_summary = s_out.getvalue()
    print(static_summary[:2000])

    # Scenario 2: Dynamic Obstacle (Scenario C)
    wmap_dyn = create_world_map()
    obs_c = make_obstacle(track_id=2, start_pos=(1.5, -1.0), velocity=(0.0, 1.0))
    layer_c = DynamicOccupancyLayer()
    layer_c.update_predictions([obs_c])

    print("\nProfiling Dynamic DWA...")
    pr_dyn = cProfile.Profile()
    pr_dyn.enable()
    for _ in range(3):
        planner.compute_trajectory(
            state, path, wmap_dyn,
            dynamic_layer=layer_c, predicted_tracks=[obs_c]
        )
    pr_dyn.disable()

    d_out = io.StringIO()
    ps_dyn = pstats.Stats(pr_dyn, stream=d_out).sort_stats("cumulative")
    ps_dyn.print_stats(35)
    dyn_summary = d_out.getvalue()
    print(dyn_summary[:2500])

    # Save outputs to reports
    out_txt = os.path.join("reports", "dwa_cpu_profiling_results.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=== STATIC DWA PROFILE (cumulative time) ===\n")
        f.write(static_summary)
        f.write("\n\n=== DYNAMIC DWA PROFILE (cumulative time) ===\n")
        f.write(dyn_summary)

    print(f"\nProfiling report saved to {out_txt}")


if __name__ == "__main__":
    profile_dwa()
