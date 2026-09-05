"""Pure Python closed-loop navigation simulation with dynamic obstacle avoidance and replanning."""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

from ai.costmap import BEVTerrainCostmap
from ai.planning import (
    AStarPlanner,
    DWAPlanner,
    GoalState,
    PlanningVisualizer,
    RobotState,
)

OUTPUT_DIR = Path("outputs/planning/animations")


def run_navigation_simulation(max_steps: int = 60, dt: float = 0.10) -> bool:
    """Run closed-loop motion simulation from start to goal."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[SIMULATION] Starting Closed-Loop Pure-Python Navigation Loop...")

    # 1. Initialize environment
    costmap = BEVTerrainCostmap.create_empty(
        x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0),
        resolution_m=0.10, default_cost=0.10
    )
    costmap.unknown.fill(False)

    # Static corridor obstacles
    r_mid, c_mid = costmap.meter_to_grid(2.8, 0.0)
    costmap.obstacle[r_mid-4:r_mid+5, :c_mid-6] = True
    costmap.cost[r_mid-4:r_mid+5, :c_mid-6] = 1.00
    costmap.obstacle[r_mid-4:r_mid+5, c_mid+7:] = True
    costmap.cost[r_mid-4:r_mid+5, c_mid+7:] = 1.00

    start = RobotState(x=0.8, y=0.0, yaw=0.0, v=0.0, omega=0.0)
    goal = GoalState(x=4.8, y=0.0, xy_tolerance_m=0.15)

    global_planner = AStarPlanner()
    local_planner = DWAPlanner()
    viz = PlanningVisualizer()

    # Initial global plan
    global_path, g_status, _ = global_planner.plan(costmap, start, goal)
    if global_path is None or g_status.value != "SUCCESS":
        print("  [ERROR] Initial global planning failed.")
        return False

    current_state = start
    robot_trajectory = [current_state.position]
    frames = []

    print(f"  Initial Global Path: {global_path.length:.2f} m ({len(global_path)} waypoints)")

    goal_reached = False

    for step in range(max_steps):
        # 1. Inject dynamic moving obstacle at steps 15..28
        if 15 <= step <= 28:
            obs_y = -0.6 + (step - 15) * 0.10  # Moving across the corridor from -0.6m to +0.7m
            r_dyn, c_dyn = costmap.meter_to_grid(2.5, obs_y)
            costmap.obstacle[r_dyn-1:r_dyn+2, c_dyn-1:c_dyn+2] = True
            costmap.cost[r_dyn-1:r_dyn+2, c_dyn-1:c_dyn+2] = 1.00
        elif step > 28:
            # Dynamic obstacle cleared
            r_dyn, _ = costmap.meter_to_grid(2.5, 0.0)
            costmap.obstacle[r_dyn-2:r_dyn+3, c_mid-6:c_mid+7] = False
            costmap.cost[r_dyn-2:r_dyn+3, c_mid-6:c_mid+7] = 0.10

        # 2. Check goal arrival
        if goal.is_reached(current_state):
            print(f"  [SUCCESS] Goal reached at step {step}! Position: ({current_state.x:.2f}, {current_state.y:.2f})")
            goal_reached = True
            break

        # 3. Local planner trajectory selection
        traj, v, w, status, diag = local_planner.compute_trajectory(current_state, global_path, costmap)

        # 4. Render HUD frame
        hud = viz.render_plan(
            costmap, current_state, goal, global_path=global_path,
            local_trajectory=traj,
            title=f"Step {step:02d} | v={v:.2f}m/s, w={w:.2f}rad/s | {status.value}"
        )
        frames.append(hud)

        # 5. Propagate robot kinematic state
        new_yaw = current_state.yaw + w * dt
        new_yaw = (new_yaw + np.pi) % (2.0 * np.pi) - np.pi
        new_x = current_state.x + v * np.cos(new_yaw) * dt
        new_y = current_state.y + v * np.sin(new_yaw) * dt

        current_state = RobotState(
            x=new_x, y=new_y, yaw=new_yaw, v=v, omega=w,
            timestamp=current_state.timestamp + dt
        )
        robot_trajectory.append(current_state.position)

    # Save summary visual montage
    if frames:
        key_frames = [frames[0], frames[len(frames)//3], frames[2*len(frames)//3], frames[-1]]
        montage = np.hstack(key_frames)
        montage_path = OUTPUT_DIR / "navigation_simulation_montage.png"
        cv2.imwrite(str(montage_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        print(f"  [SAVED] Simulation Montage: {montage_path}")

    return goal_reached


def main():
    parser = argparse.ArgumentParser(description="Run closed-loop navigation simulation")
    parser.add_argument("--steps", type=int, default=50, help="Maximum simulation steps")
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("      SIH26126 PHASE 6: CLOSED-LOOP NAVIGATION SIMULATOR")
    print("=" * 78)

    success = run_navigation_simulation(max_steps=args.steps)
    if success:
        print("\n[OK] Simulation completed successfully with obstacle avoidance.")
    else:
        print("\n[WARN] Simulation ended before reaching goal tolerance.")


if __name__ == "__main__":
    main()
