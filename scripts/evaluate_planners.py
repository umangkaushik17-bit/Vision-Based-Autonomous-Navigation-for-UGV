"""Comprehensive evaluation suite: Ablation study, Terrain detour experiment, and Dynamic obstacle response."""

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
    DWAPlanner,
    GoalState,
    PathMetricsCalculator,
    PlanningConfig,
    PlanningVisualizer,
    RobotState,
)

OUTPUT_DIR = Path("outputs/planning/comparisons")


def run_ablation_study() -> Dict[str, Any]:
    """Execute Ablations A through D to evaluate the impact of layered terrain awareness."""
    print("\n[1/3] Running Multi-Layer Planning Ablation Study (Section 59)...")

    # Base costmap with terrain, hazard, and unknown regions
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0),
        resolution_m=0.10, default_cost=0.10
    )
    costmap.unknown.fill(False)

    # Moderate rough patch
    r_rgh, c_rgh = costmap.meter_to_grid(2.5, 0.0)
    costmap.roughness[r_rgh-4:r_rgh+5, c_rgh-4:c_rgh+5] = 0.80
    costmap.cost[r_rgh-4:r_rgh+5, c_rgh-4:c_rgh+5] = 0.50

    # Explicit hazard patch
    r_haz, c_haz = costmap.meter_to_grid(3.5, 0.5)
    costmap.hazard[r_haz-3:r_haz+4, c_haz-3:c_haz+4] = 0.90
    costmap.cost[r_haz-3:r_haz+4, c_haz-3:c_haz+4] = 0.90
    costmap.obstacle[r_haz-3:r_haz+4, c_haz-3:c_haz+4] = True

    # Unknown region
    r_unk, c_unk = costmap.meter_to_grid(2.0, -1.0)
    costmap.unknown[r_unk-4:r_unk+5, c_unk-4:c_unk+5] = True
    costmap.cost[r_unk-4:r_unk+5, c_unk-4:c_unk+5] = 0.65

    start = RobotState(x=0.8, y=0.0, yaw=0.0)
    goal = GoalState(x=4.8, y=0.0)
    metrics_calc = PathMetricsCalculator()

    # Ablation A: Distance/Occupancy-only planning (weights zeroed out except distance)
    cfg_a = PlanningConfig()
    cfg_a.cost_weights = {
        "w_distance": 1.0, "w_traversability": 0.0, "w_cost": 0.0,
        "w_hazard": 0.0, "w_slope": 0.0, "w_roughness": 0.0, "w_unknown": 0.0
    }
    planner_a = AStarPlanner(config=cfg_a)
    path_a, _, _ = planner_a.plan(costmap, start, goal)
    m_a = metrics_calc.evaluate_path(path_a, costmap)

    # Ablation B: Distance + Terrain-aware planning
    cfg_b = PlanningConfig()
    cfg_b.cost_weights = {
        "w_distance": 1.0, "w_traversability": 0.8, "w_cost": 1.5,
        "w_hazard": 0.0, "w_slope": 1.2, "w_roughness": 0.6, "w_unknown": 0.0
    }
    planner_b = AStarPlanner(config=cfg_b)
    path_b, _, _ = planner_b.plan(costmap, start, goal)
    m_b = metrics_calc.evaluate_path(path_b, costmap)

    # Ablation C: Terrain + Hazard-aware planning
    cfg_c = PlanningConfig()
    cfg_c.cost_weights = {
        "w_distance": 1.0, "w_traversability": 0.8, "w_cost": 1.5,
        "w_hazard": 2.0, "w_slope": 1.2, "w_roughness": 0.6, "w_unknown": 0.0
    }
    planner_c = AStarPlanner(config=cfg_c)
    path_c, _, _ = planner_c.plan(costmap, start, goal)
    m_c = metrics_calc.evaluate_path(path_c, costmap)

    # Ablation D: Full Fusion (Distance + Terrain + Hazard + Unknown aware)
    cfg_d = PlanningConfig()
    planner_d = AStarPlanner(config=cfg_d)
    path_d, _, _ = planner_d.plan(costmap, start, goal)
    m_d = metrics_calc.evaluate_path(path_d, costmap)

    print(f"  Ablation A (Occupancy Only)  : Length={m_a.path_length_m:.2f}m | Mean Cost={m_a.mean_cost:.3f} | Min Clear={m_a.min_clearance_m:.2f}m")
    print(f"  Ablation B (+Terrain Aware)  : Length={m_b.path_length_m:.2f}m | Mean Cost={m_b.mean_cost:.3f} | Min Clear={m_b.min_clearance_m:.2f}m")
    print(f"  Ablation C (+Hazard Aware)   : Length={m_c.path_length_m:.2f}m | Mean Cost={m_c.mean_cost:.3f} | Min Clear={m_c.min_clearance_m:.2f}m")
    print(f"  Ablation D (Full Fusion)     : Length={m_d.path_length_m:.2f}m | Mean Cost={m_d.mean_cost:.3f} | Min Clear={m_d.min_clearance_m:.2f}m")

    return {
        "ablation_a_occupancy_only": m_a.to_dict(),
        "ablation_b_terrain_aware": m_b.to_dict(),
        "ablation_c_hazard_aware": m_c.to_dict(),
        "ablation_d_full_fusion": m_d.to_dict()
    }


def run_terrain_detour_experiment() -> Dict[str, Any]:
    """Verify Section 61: Planner selects a longer, safer path when the direct route is hazardous."""
    print("\n[2/3] Running Terrain Detour Experiment (Section 61)...")

    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0),
        resolution_m=0.10, default_cost=0.10
    )
    costmap.unknown.fill(False)

    # Place a steep / high-cost mountain ridge directly along the straight centerline
    r_mid, c_mid = costmap.meter_to_grid(2.8, 0.0)
    costmap.slope_deg[r_mid-4:r_mid+5, c_mid-8:c_mid+9] = 30.0  # Lethal slope >= 28 deg
    costmap.cost[r_mid-4:r_mid+5, c_mid-8:c_mid+9] = 1.00
    costmap.obstacle[r_mid-4:r_mid+5, c_mid-8:c_mid+9] = True

    # Lateral bypass on the left (y = +1.5m) is smooth, flat, and safe (cost = 0.10)
    start = RobotState(x=0.8, y=0.0, yaw=0.0)
    goal = GoalState(x=4.8, y=0.0)

    planner = AStarPlanner()
    path, status, diag = planner.plan(costmap, start, goal)

    direct_dist = float(np.hypot(goal.x - start.x, goal.y - start.y))
    actual_length = float(path.length) if path else 0.0
    detour_ratio = actual_length / max(direct_dist, 1e-4)

    # Check maximum lateral diversion
    max_lateral_m = float(np.max(np.abs(path.waypoints_xy[:, 1]))) if path else 0.0

    print(f"  Direct Geometric Distance   : {direct_dist:.2f} m")
    print(f"  Selected Path Length         : {actual_length:.2f} m (Detour Ratio: {detour_ratio:.2f}x)")
    print(f"  Max Lateral Detour Clearance : {max_lateral_m:.2f} m")
    print(f"  Detour Verification          : {'PASSED (Safe Detour Chosen)' if max_lateral_m > 0.8 else 'FAILED'}")

    viz = PlanningVisualizer()
    hud = viz.render_plan(costmap, start, goal, global_path=path, title="Section 61 Terrain Detour Experiment")
    cv2.imwrite(str(OUTPUT_DIR / "terrain_detour_experiment.png"), cv2.cvtColor(hud, cv2.COLOR_RGB2BGR))

    return {
        "direct_geometric_distance_m": round(direct_dist, 3),
        "actual_path_length_m": round(actual_length, 3),
        "detour_ratio": round(detour_ratio, 2),
        "max_lateral_detour_m": round(max_lateral_m, 2),
        "safe_detour_selected": bool(max_lateral_m > 0.8)
    }


def run_dynamic_obstacle_experiment() -> Dict[str, Any]:
    """Verify Section 60: Local planner reactions to unexpected obstacles and corridor blockages."""
    print("\n[3/3] Running Dynamic Obstacle Reaction & Emergency Stop Experiment (Section 60)...")

    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0),
        resolution_m=0.10, default_cost=0.10
    )
    costmap.unknown.fill(False)

    start = RobotState(x=1.0, y=0.0, yaw=0.0, v=0.50, omega=0.0)
    goal = GoalState(x=4.5, y=0.0)

    astar = AStarPlanner()
    path, _, _ = astar.plan(costmap, start, goal)

    dwa = DWAPlanner()

    # Step 1: Nominal tracking (clear corridor)
    _, v1, _, s1, _ = dwa.compute_trajectory(start, path, costmap)

    # Step 2: Partial dynamic obstacle enters path -> planner navigates around
    r_dyn, c_dyn = costmap.meter_to_grid(1.8, 0.0)
    costmap.obstacle[r_dyn-1:r_dyn+2, c_dyn-1:c_dyn+2] = True
    costmap.cost[r_dyn-1:r_dyn+2, c_dyn-1:c_dyn+2] = 1.00
    _, v2, w2, s2, _ = dwa.compute_trajectory(start, path, costmap)

    # Step 3: Complete corridor blockage -> Emergency Stop triggered (v=0, w=0)
    costmap.obstacle[r_dyn-2:r_dyn+3, :] = True
    costmap.cost[r_dyn-2:r_dyn+3, :] = 1.00
    _, v3, w3, s3, diag3 = dwa.compute_trajectory(start, path, costmap)

    # Step 4: Obstacle removed -> Navigation resumes
    costmap.obstacle[r_dyn-2:r_dyn+3, :] = False
    costmap.cost[r_dyn-2:r_dyn+3, :] = 0.10
    _, v4, _, s4, _ = dwa.compute_trajectory(start, path, costmap)

    print(f"  Step 1 (Clear Corridor)      : Velocity={v1:.2f} m/s | Status={s1.value}")
    print(f"  Step 2 (Partial Obstacle)    : Velocity={v2:.2f} m/s, w={w2:.2f} rad/s (Steering around)")
    print(f"  Step 3 (Complete Blockage)   : Velocity={v3:.2f} m/s | Emergency Stop Triggered: {diag3.get('emergency_stop_triggered')}")
    print(f"  Step 4 (Obstacle Cleared)    : Velocity={v4:.2f} m/s | Status={s4.value} (Navigation Resumed)")

    return {
        "nominal_velocity_mps": round(v1, 3),
        "evasive_steering_w_radps": round(w2, 3),
        "emergency_stop_velocity_mps": round(v3, 3),
        "emergency_stop_passed": bool(v3 == 0.0 and s3.value == "NO_SAFE_TRAJECTORY"),
        "recovery_resumed_passed": bool(v4 > 0.0 and s4.value == "SUCCESS")
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("      SIH26126 PHASE 6: MASTER EVALUATION & ABLATION HARNESS")
    print("=" * 78)

    ablation_res = run_ablation_study()
    detour_res = run_terrain_detour_experiment()
    dyn_res = run_dynamic_obstacle_experiment()

    master_eval = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ablation_study": ablation_res,
        "terrain_detour_experiment": detour_res,
        "dynamic_obstacle_experiment": dyn_res
    }

    report_path = OUTPUT_DIR / "planning_evaluation_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(master_eval, f, indent=2)

    print("\n" + "=" * 78)
    print(f"[OK] Master Planning Evaluation Report saved to: {report_path}")


if __name__ == "__main__":
    main()
