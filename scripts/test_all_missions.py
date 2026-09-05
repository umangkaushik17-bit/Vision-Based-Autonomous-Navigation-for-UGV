"""Test script to evaluate all 4 closed-loop missions with the corrected navigation pipeline."""

import math
from pathlib import Path
import sys
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "ros2_ws" / "src" / "ugv_navigation"))
import ugv_navigation

from ai.planning.base import PlannerStatus
from simulation.sim_engine import GazeboSimEngine
from ugv_navigation.adapters.planner_adapter import PoseStampedMsg
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.nodes.localization_node import LocalizationNode
from ugv_navigation.nodes.perception_node import PerceptionNode
from ugv_navigation.nodes.planner_node import PlannerNode
from ugv_navigation.nodes.spatial_fusion_node import SpatialFusionNode
from ugv_navigation.nodes.static_tf_node import StaticTfNode
from ugv_navigation.tf.tf_tree import TfTreeBuffer


def run_mission(m_id, m_name, goal, max_steps=150, obs_type=None, blocked=False):
    print(f"\n========================================================")
    print(f"  Starting: {m_name}")
    print(f"========================================================")

    sim = GazeboSimEngine()
    sim.gt_registry["obstacles"] = []
    tf_buf = TfTreeBuffer()
    StaticTfNode(tf_buffer=tf_buf)

    loc_node = LocalizationNode(tf_buffer=tf_buf)
    perc_node = PerceptionNode()
    spatial_node = SpatialFusionNode(tf_buffer=tf_buf)
    planner_node = PlannerNode(tf_buffer=tf_buf)

    gx, gy = goal
    goal_msg = PoseStampedMsg.from_xy_yaw(gx, gy, None, frame_id="odom", timestamp_s=0.0)
    planner_node.on_goal_received(goal_msg)

    if blocked:
        sim.gt_registry["obstacles"].append({
            "id": "wall",
            "type": "rock",
            "position": [gx, gy, 0.0],
            "bounding_radius_m": 0.5,
            "x": gx,
            "y": gy,
            "radius": 0.5
        })
    elif obs_type == "rock":
        sim.gt_registry["obstacles"].append({
            "id": "rock_mid",
            "type": "rock",
            "position": [1.2, 0.0, 0.0],
            "bounding_radius_m": 0.35,
            "x": 1.2,
            "y": 0.0,
            "radius": 0.35
        })
    elif obs_type == "hazard":
        sim.gt_registry["obstacles"].append({
            "id": "hazard_mid",
            "type": "puddle",
            "position": [1.2, 0.0, 0.0],
            "bounding_radius_m": 0.35,
            "x": 1.2,
            "y": 0.0,
            "radius": 0.35
        })

    gt_traj = []
    est_traj = []
    cmd_history = []
    status_history = []
    timestamps = []
    clearances = []
    collision_count = 0
    min_clearance = float("inf")
    emergency_stops = 0

    for step in range(max_steps):
        t_sim = sim.sim_time_s
        stream = sim.get_next_sensor_cycle()

        # 1. Localization
        loc_msg, _ = loc_node.process_frame(
            img_left=stream.image_left,
            timestamp_s=stream.timestamp,
            img_right=stream.image_right,
        )
        spatial_node.on_localization_status(loc_msg)
        planner_node.on_localization_status(loc_msg)

        # 2. Perception
        bev = perc_node.process_frame(
            image_rgb=stream.image_left,
            timestamp_s=stream.timestamp,
        )

        # 3. Spatial Fusion
        fused, map_msg, _ = spatial_node.on_costmap_received(bev)
        if map_msg is None:
            map_msg = WorldTerrainMapMsg.from_world_terrain_map(
                spatial_node.adapter.world_map, timestamp_s=stream.timestamp
            )

        if blocked:
            w_m = map_msg.to_world_terrain_map()
            r_b, c_b = w_m.world_to_grid(gx, gy)
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    if (dr * 0.10)**2 + (dc * 0.10)**2 <= (0.50**2):
                        w_m.obstacle[r_b + dr, c_b + dc] = True
            map_msg = WorldTerrainMapMsg.from_world_terrain_map(w_m, timestamp_s=stream.timestamp)
        elif obs_type == "rock":
            w_m = map_msg.to_world_terrain_map()
            r_o, c_o = w_m.world_to_grid(1.2, 0.0)
            for dr in range(-4, 5):
                for dc in range(-4, 5):
                    if (dr * 0.10)**2 + (dc * 0.10)**2 <= (0.35**2):
                        w_m.obstacle[r_o + dr, c_o + dc] = True
            map_msg = WorldTerrainMapMsg.from_world_terrain_map(w_m, timestamp_s=stream.timestamp)
        elif obs_type == "hazard":
            w_m = map_msg.to_world_terrain_map()
            r_h, c_h = w_m.world_to_grid(1.2, 0.0)
            for dr in range(-4, 5):
                for dc in range(-4, 5):
                    if (dr * 0.10)**2 + (dc * 0.10)**2 <= (0.35**2):
                        w_m.hazard[r_h + dr, c_h + dc] = 0.85
            map_msg = WorldTerrainMapMsg.from_world_terrain_map(w_m, timestamp_s=stream.timestamp)

        planner_node.on_map_received(map_msg)

        # 4. Planner cycle
        cmd, path, status, diag = planner_node.process_cycle(sim_time_s=stream.timestamp, dt_s=0.10)

        # 5. Physics step
        sim.set_command(cmd.linear.x, cmd.angular.z)
        sim.step_physics(dt_s=0.05)

        gt_p = stream.ground_truth_pose[:2, 3]
        gt_traj.append(gt_p.copy())
        est_traj.append([loc_msg.pose.pose.position.x, loc_msg.pose.pose.position.y])
        cmd_history.append((cmd.linear.x, cmd.angular.z))
        status_history.append(status.value)
        timestamps.append(stream.timestamp)

        if status == PlannerStatus.EMERGENCY_STOP:
            emergency_stops += 1

        step_clearance = float("inf")
        for obs in sim.gt_registry.get("obstacles", []):
            ox = obs["position"][0] if "position" in obs else obs.get("x", 0.0)
            oy = obs["position"][1] if "position" in obs else obs.get("y", 0.0)
            d_obs = math.hypot(gt_p[0] - ox, gt_p[1] - oy)
            r_coll = obs.get("bounding_radius_m", obs.get("radius", 0.35)) + 0.225
            clearance = d_obs - r_coll
            if clearance < 0:
                collision_count += 1
            step_clearance = min(step_clearance, clearance)

        min_clearance = min(min_clearance, step_clearance)
        clearances.append(step_clearance)

        d_goal = math.hypot(gt_p[0] - gx, gt_p[1] - gy)
        if step % 10 == 0 or status == PlannerStatus.GOAL_REACHED:
            print(f"Step {step:3d} ({stream.timestamp:.2f}s) | GT:({gt_p[0]:.2f}, {gt_p[1]:.2f}) | Dist:{d_goal:.2f}m | Status:{status.value:15s} | V:{cmd.linear.x:.2f}")

        if status == PlannerStatus.GOAL_REACHED:
            print(f"==> Goal reached successfully at step {step} ({stream.timestamp:.2f}s)!")
            break
        elif blocked and status in (PlannerStatus.NO_PATH, PlannerStatus.GOAL_BLOCKED):
            print(f"==> Correctly stopped due to blocked goal: {status.value}")
            break

    gt_arr = np.array(gt_traj)
    final_dist = math.hypot(gt_arr[-1, 0] - gx, gt_arr[-1, 1] - gy)
    path_len = float(np.sum(np.hypot(np.diff(gt_arr[:, 0]), np.diff(gt_arr[:, 1])))) if len(gt_arr) > 1 else 0.0
    safe_min_clr = float(max(0.0, min_clearance)) if min_clearance != float("inf") else 2.0

    print(f"--- Summary for {m_name} ---")
    print(f"Goal: ({gx}, {gy})")
    print(f"Final Pos: ({gt_arr[-1, 0]:.2f}, {gt_arr[-1, 1]:.2f})")
    print(f"Goal Error: {final_dist:.3f} m")
    print(f"Path Length: {path_len:.2f} m")
    print(f"Min Clearance: {safe_min_clr:.3f} m")
    print(f"Collisions: {collision_count}")
    print(f"Emergency Stops: {emergency_stops}")
    print(f"Final Status: {status_history[-1]}")

    return {
        "id": m_id,
        "name": m_name,
        "goal": (gx, gy),
        "final_pos": (float(gt_arr[-1, 0]), float(gt_arr[-1, 1])),
        "goal_error": round(final_dist, 3),
        "path_length": round(path_len, 2),
        "min_clearance": round(safe_min_clr, 3),
        "collisions": collision_count,
        "emergency_stops": emergency_stops,
        "final_status": status_history[-1],
        "gt_traj": gt_traj,
        "est_traj": est_traj,
        "cmd_history": cmd_history,
        "status_history": status_history,
        "timestamps": timestamps,
        "clearances": clearances,
    }


if __name__ == "__main__":
    results = []
    results.append(run_mission("flat", "Flat Open Navigation", (2.0, 0.0), max_steps=100))
    results.append(run_mission("obstacle", "Rock Obstacle Detour", (2.5, 0.0), max_steps=140, obs_type="rock"))
    results.append(run_mission("hazard", "Hazard Area Bypass", (2.5, 0.0), max_steps=140, obs_type="hazard"))
    results.append(run_mission("failure", "Blocked Goal Failure", (2.0, 0.0), max_steps=25, blocked=True))

    print("\n==========================================================================================")
    print("FINAL MISSION RESULTS TABLE")
    print("==========================================================================================")
    print(f"{'Mission':<22} | {'Goal':<10} | {'Final Pos':<14} | {'Goal Err':<9} | {'Path Len':<9} | {'Min Clr':<8} | {'Coll':<5} | {'E-Stops':<7} | {'Result'}")
    print("-" * 110)
    for r in results:
        res_str = "SUCCESS" if (r["goal_error"] <= 0.25 or (r["id"] == "failure" and r["final_status"] in ("NO_PATH", "GOAL_BLOCKED"))) else "FAIL"
        fx, fy = r['final_pos'][0], r['final_pos'][1]
        f_pos_str = f"({fx:.2f}, {fy:.2f})"
        print(f"{r['name']:<22} | {str(r['goal']):<10} | {f_pos_str:<14} | {r['goal_error']:<9.3f} | {r['path_length']:<9.2f} | {r['min_clearance']:<8.3f} | {r['collisions']:<5d} | {r['emergency_stops']:<7d} | {res_str}")
