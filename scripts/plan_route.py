"""CLI tool to plan a global path on a BEV costmap using A*, PF-RRT*, or Hybrid A*."""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

from ai.costmap import BEVTerrainCostmap, CostmapSerializer
from ai.planning import (
    AStarPlanner,
    GoalState,
    HybridAStarPlanner,
    PFRRTStarPlanner,
    PlanningVisualizer,
    RobotState,
)

OUTPUT_DIR = Path("outputs/planning/global")


def main():
    parser = argparse.ArgumentParser(description="Plan a route on a BEVTerrainCostmap")
    parser.add_argument("--planner", type=str, default="astar", choices=["astar", "pf_rrt_star", "hybrid_astar"], help="Planner algorithm")
    parser.add_argument("--start-x", type=float, default=1.0, help="Start X (meters)")
    parser.add_argument("--start-y", type=float, default=0.0, help="Start Y (meters)")
    parser.add_argument("--start-yaw", type=float, default=0.0, help="Start Yaw (radians)")
    parser.add_argument("--goal-x", type=float, default=4.5, help="Goal X (meters)")
    parser.add_argument("--goal-y", type=float, default=1.0, help="Goal Y (meters)")
    parser.add_argument("--output-name", type=str, default="sample_plan", help="Output filename prefix")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Create or load costmap
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0),
        resolution_m=0.10, default_cost=0.10
    )
    costmap.unknown.fill(False)

    # Place sample obstacle
    r_o, c_o = costmap.meter_to_grid(2.8, 0.5)
    costmap.obstacle[r_o-2:r_o+3, c_o-2:c_o+3] = True
    costmap.cost[r_o-2:r_o+3, c_o-2:c_o+3] = 1.00

    start = RobotState(x=args.start_x, y=args.start_y, yaw=args.start_yaw)
    goal = GoalState(x=args.goal_x, y=args.goal_y)

    # 2. Select planner
    if args.planner == "astar":
        planner = AStarPlanner()
    elif args.planner == "pf_rrt_star":
        planner = PFRRTStarPlanner()
    else:
        planner = HybridAStarPlanner()

    print(f"\n[PLANNING] Running {planner.name} from ({start.x}, {start.y}) to ({goal.x}, {goal.y})...")
    path, status, diag = planner.plan(costmap, start, goal)

    print(f"  Status        : {status.value}")
    print(f"  Planning Time : {diag.get('planning_time_ms', 0.0)} ms")
    if path is not None:
        print(f"  Path Length   : {path.length:.2f} m ({len(path)} waypoints)")
        print(f"  Min Clearance : {diag.get('min_clearance_m', 0.0)} m")

        # 3. Render and save
        viz = PlanningVisualizer()
        rendered = viz.render_plan(costmap, start, goal, global_path=path, title=f"{planner.name} Plan")
        out_img = OUTPUT_DIR / f"{args.output_name}.png"
        cv2.imwrite(str(out_img), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        print(f"  Saved Image   : {out_img}")
    else:
        print(f"  Failure Reason: {diag.get('reason', 'Search terminated with no valid path')}")


if __name__ == "__main__":
    main()
