"""Targeted diagnostic script to trace the Flat Navigation Mission step-by-step."""

import math
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "ros2_ws" / "src" / "ugv_navigation"))
import ugv_navigation

from ai.planning.base import PlannerStatus
from simulation.sim_engine import GazeboSimEngine
from ugv_navigation.adapters.planner_adapter import PoseStampedMsg, WorldTerrainMapMsg
from ugv_navigation.nodes.localization_node import LocalizationNode
from ugv_navigation.nodes.perception_node import PerceptionNode
from ugv_navigation.nodes.planner_node import PlannerNode
from ugv_navigation.nodes.spatial_fusion_node import SpatialFusionNode
from ugv_navigation.nodes.static_tf_node import StaticTfNode
from ugv_navigation.tf.tf_tree import TfTreeBuffer


def run_flat_diagnostic():
    sim = GazeboSimEngine()
    tf_buf = TfTreeBuffer()
    StaticTfNode(tf_buffer=tf_buf)

    loc_node = LocalizationNode(tf_buffer=tf_buf)
    perc_node = PerceptionNode()
    spatial_node = SpatialFusionNode(tf_buffer=tf_buf)
    planner_node = PlannerNode(tf_buffer=tf_buf)

    goal_x, goal_y = 2.0, 0.0
    goal_msg = PoseStampedMsg.from_xy_yaw(goal_x, goal_y, None, frame_id="odom", timestamp_s=0.0)
    planner_node.on_goal_received(goal_msg)

    print(f"Goal set to ({goal_x}, {goal_y})")
    print(f"{'Step':<5} | {'Time(s)':<7} | {'Robot X':<8} | {'Robot Y':<8} | {'Dist(m)':<8} | {'Status':<15} | {'Cmd V':<7} | {'Cmd W':<7} | {'Reason'}")
    print("-" * 105)

    for step in range(160):
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
        planner_node.on_map_received(map_msg)

        # 4. Planner cycle
        cmd, path, status, diag = planner_node.process_cycle(sim_time_s=stream.timestamp, dt_s=0.10)

        # 5. Step physics
        sim.set_command(cmd.linear.x, cmd.angular.z)
        sim.step_physics(dt_s=0.05)

        gt_p = stream.ground_truth_pose[:2, 3]
        dist_to_goal = math.hypot(gt_p[0] - goal_x, gt_p[1] - goal_y)
        loc_x = loc_msg.pose.pose.position.x
        loc_y = loc_msg.pose.pose.position.y
        loc_dist = math.hypot(loc_x - goal_x, loc_y - goal_y)
        reason = diag.get("safety_fault_reason", diag.get("replan_reason", "nominal"))

        print(f"{step:<5} | {stream.timestamp:<7.3f} | GT:({gt_p[0]:.2f}, {gt_p[1]:.2f}, d={dist_to_goal:.2f}) | LOC:({loc_x:.2f}, {loc_y:.2f}, d={loc_dist:.2f}) | {status.value:<15} | V:{cmd.linear.x:.2f} | {reason}")

        if status == PlannerStatus.GOAL_REACHED:
            print(f"Goal reached successfully at step {step} ({stream.timestamp:.2f}s)!")
            break


if __name__ == "__main__":
    run_flat_diagnostic()
