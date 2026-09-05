"""Benchmark harness comparing A*, PF-RRT*, and Hybrid A* across 8 test scenarios."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning import (
    AStarPlanner,
    GoalState,
    HybridAStarPlanner,
    PathMetricsCalculator,
    PFRRTStarPlanner,
    PlanningVisualizer,
    RobotState,
)

OUTPUT_DIR = Path("outputs/planning/benchmarks")
SCENARIO_DIR = Path("outputs/planning/scenarios")


def build_scenario_costmap(scenario_id: int) -> Tuple[BEVTerrainCostmap, RobotState, GoalState, str]:
    """Construct deterministic test costmaps matching prompt specifications."""
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0),
        resolution_m=0.10, default_cost=0.10
    )
    costmap.unknown.fill(False)
    start = RobotState(x=0.8, y=0.0, yaw=0.0)
    goal = GoalState(x=4.8, y=0.0)

    if scenario_id == 1:
        name = "scenario1_flat_terrain"
        # Pure flat terrain: baseline nominal shortest path

    elif scenario_id == 2:
        name = "scenario2_rough_terrain_detour"
        # High roughness patch directly in the middle: cost=0.70, roughness=0.85
        r_mid, c_mid = costmap.meter_to_grid(2.8, 0.0)
        costmap.roughness[r_mid-6:r_mid+7, c_mid-8:c_mid+9] = 0.85
        costmap.cost[r_mid-6:r_mid+7, c_mid-8:c_mid+9] = 0.60

    elif scenario_id == 3:
        name = "scenario3_steep_slope_avoidance"
        # Critical slope ridge across center: slope=32 degrees >= 28 degrees (lethal)
        r_mid, c_mid = costmap.meter_to_grid(2.8, 0.0)
        costmap.slope_deg[r_mid-4:r_mid+5, c_mid-12:c_mid+13] = 32.0
        costmap.cost[r_mid-4:r_mid+5, c_mid-12:c_mid+13] = 1.00
        costmap.obstacle[r_mid-4:r_mid+5, c_mid-12:c_mid+13] = True

    elif scenario_id == 4:
        name = "scenario4_creek_water_hazard"
        # Lethal creek water pool: hazard=1.0, cost=1.00
        r_mid, c_mid = costmap.meter_to_grid(2.8, 0.0)
        costmap.hazard[r_mid-5:r_mid+6, c_mid-10:c_mid+11] = 1.00
        costmap.cost[r_mid-5:r_mid+6, c_mid-10:c_mid+11] = 1.00
        costmap.obstacle[r_mid-5:r_mid+6, c_mid-10:c_mid+11] = True

    elif scenario_id == 5:
        name = "scenario5_negative_ditch"
        # Ditch washout depression: step height=0.30m >= 0.20m (lethal)
        r_mid, c_mid = costmap.meter_to_grid(2.8, 0.0)
        costmap.height_var[r_mid-4:r_mid+5, c_mid-15:c_mid+16] = 0.30
        costmap.cost[r_mid-4:r_mid+5, c_mid-15:c_mid+16] = 1.00
        costmap.obstacle[r_mid-4:r_mid+5, c_mid-15:c_mid+16] = True

    elif scenario_id == 6:
        name = "scenario6_unknown_region"
        # High uncertainty region in middle: unknown=True, cost=0.65
        r_mid, c_mid = costmap.meter_to_grid(2.8, 0.0)
        costmap.unknown[r_mid-8:r_mid+9, c_mid-10:c_mid+11] = True
        costmap.cost[r_mid-8:r_mid+9, c_mid-10:c_mid+11] = 0.65

    elif scenario_id == 7:
        name = "scenario7_narrow_safe_corridor"
        # Two large obstacles with a 1.0m wide corridor between them (rover width 0.45m)
        r_mid, c_mid = costmap.meter_to_grid(2.8, 0.0)
        costmap.obstacle[r_mid-5:r_mid+6, :c_mid-5] = True
        costmap.cost[r_mid-5:r_mid+6, :c_mid-5] = 1.00
        costmap.obstacle[r_mid-5:r_mid+6, c_mid+6:] = True
        costmap.cost[r_mid-5:r_mid+6, c_mid+6:] = 1.00

    elif scenario_id == 8:
        name = "scenario8_completely_blocked_corridor"
        # Complete lethal barrier across entire corridor: NO_PATH expected
        r_mid, _ = costmap.meter_to_grid(2.8, 0.0)
        costmap.obstacle[r_mid-3:r_mid+4, :] = True
        costmap.cost[r_mid-3:r_mid+4, :] = 1.00

    else:
        raise ValueError(f"Unknown scenario ID: {scenario_id}")

    return costmap, start, goal, name


def main():
    parser = argparse.ArgumentParser(description="Benchmark global planners across 8 scenarios")
    parser.add_argument("--trials", type=int, default=3, help="Number of benchmark trials per planner")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("      SIH26126 PHASE 6: GLOBAL PLANNER BENCHMARK & COMPARISON")
    print("=" * 78)

    planners = [
        AStarPlanner(),
        PFRRTStarPlanner(),
        HybridAStarPlanner()
    ]
    metrics_calc = PathMetricsCalculator()
    viz = PlanningVisualizer()

    benchmark_summary: Dict[str, Any] = {}

    for s_id in range(1, 9):
        costmap, start, goal, scenario_name = build_scenario_costmap(s_id)
        print(f"\n[{s_id}/8] Evaluating Scenario: {scenario_name}")
        benchmark_summary[scenario_name] = {}

        for planner in planners:
            times = []
            success_count = 0
            best_path = None
            last_diag: Dict[str, Any] = {}

            for trial in range(args.trials):
                t0 = time.perf_counter()
                path, status, diag = planner.plan(costmap, start, goal)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                times.append(elapsed_ms)
                last_diag = diag

                if path is not None and status.value == "SUCCESS":
                    success_count += 1
                    if best_path is None or path.length < best_path.length:
                        best_path = path

            success_rate = (success_count / args.trials) * 100.0
            mean_time = float(np.mean(times))
            p50_time = float(np.percentile(times, 50))
            p95_time = float(np.percentile(times, 95))

            if best_path is not None:
                m = metrics_calc.evaluate_path(best_path, costmap)
                metrics_dict = m.to_dict()
                # Save scenario visualization for trial 0
                hud = viz.render_plan(costmap, start, goal, global_path=best_path, title=f"{planner.name} on {scenario_name}")
                cv2.imwrite(str(SCENARIO_DIR / f"{scenario_name}_{planner.name.replace('*', 'star').replace(' ', '_').lower()}.png"), cv2.cvtColor(hud, cv2.COLOR_RGB2BGR))
            else:
                metrics_dict = {
                    "path_length_m": None, "mean_cost": None, "min_clearance_m": None,
                    "total_risk": None, "smoothness_score": None
                }

            print(f"  {planner.name:<12}: Success={success_rate:>5.1f}% | Time={mean_time:>6.1f}ms (P95={p95_time:>6.1f}ms) | Length={str(metrics_dict['path_length_m']):>5}m | Clearance={str(metrics_dict['min_clearance_m'])}m")

            benchmark_summary[scenario_name][planner.name] = {
                "success_rate_percent": round(success_rate, 1),
                "mean_time_ms": round(mean_time, 2),
                "p50_time_ms": round(p50_time, 2),
                "p95_time_ms": round(p95_time, 2),
                "nodes_explored": last_diag.get("nodes_explored", 0),
                "metrics": metrics_dict
            }

    report_path = OUTPUT_DIR / "global_planners_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n" + "=" * 78)
    print(f"[OK] Master Global Planner Benchmark Report saved to: {report_path}")


if __name__ == "__main__":
    main()
