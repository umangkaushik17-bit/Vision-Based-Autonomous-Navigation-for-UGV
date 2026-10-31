#!/usr/bin/env python3
"""Unified Presentation Launch Orchestrator for Phase 14.7.3.

Provides single-command startup for the complete presentation environment:
1. Gazebo Sim loading the 100m outdoor presentation world ('outdoor_presentation_100m.sdf')
2. ROS-GZ Parameter Bridge connecting cmd_vel, odometry, clock, IMU, camera, and joints
3. Interactive UGV Navigation Node (A*, DWA, EKF, RELLIS-3D, MissionExecutive, and Viz Adapters)
4. Presentation Dynamic Obstacle Controller (smooth cross-corridor patrol near W3)
5. RViz2 pre-configured with the Hero Presentation profile ('hero_presentation.rviz')

Design Invariants:
- Failure Isolation: RViz is non-fatal; closing or restarting RViz does not stop navigation.
- Simulation Clock: use_sim_time is propagated across all processes.
- Zero Algorithm Modifications: Autonomy decisions remain strictly downstream and unmodified.
"""

from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_dir = Path(__file__).resolve().parent.parent
    workspace_dir = pkg_dir.parent

    default_world = pkg_dir / "worlds" / "outdoor_presentation_100m.sdf"
    bridge_config_path = pkg_dir / "config" / "bridge_params.yaml"
    default_rviz_config = pkg_dir / "rviz" / "hero_presentation.rviz"
    nav_node_path = pkg_dir / "nodes" / "interactive_ugv_navigation_node.py"
    obstacle_node_path = pkg_dir / "nodes" / "presentation_obstacle_controller.py"
    default_mission = workspace_dir / "configs" / "mission_phase14_5_100m.yaml"
    default_obstacles_file = pkg_dir / "config" / "ground_truth_obstacles_100m.json"

    # Launch Configurations
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    headless = LaunchConfiguration("headless", default="false")
    world = LaunchConfiguration("world", default=str(default_world))
    rviz = LaunchConfiguration("rviz", default="true")
    rviz_config = LaunchConfiguration("rviz_config", default=str(default_rviz_config))
    enable_dynamic_obstacle = LaunchConfiguration("enable_dynamic_obstacle", default="true")
    mission_file = LaunchConfiguration("mission_file", default=str(default_mission))
    obstacles_file = LaunchConfiguration("obstacles_file", default=str(default_obstacles_file))
    enable_vision_costmap = LaunchConfiguration("enable_vision_costmap", default="false")

    # Declare Arguments
    decl_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (/clock) time across ROS2 nodes",
    )
    decl_headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo Sim in headless mode without GUI",
    )
    decl_world = DeclareLaunchArgument(
        "world",
        default_value=str(default_world),
        description="Full path to Gazebo world SDF file",
    )
    decl_rviz = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz2 with Hero Presentation profile",
    )
    decl_rviz_config = DeclareLaunchArgument(
        "rviz_config",
        default_value=str(default_rviz_config),
        description="Full path to RViz2 configuration file",
    )
    decl_enable_dynamic_obstacle = DeclareLaunchArgument(
        "enable_dynamic_obstacle",
        default_value="true",
        description="Enable autonomous cross-corridor patrol for dynamic obstacle",
    )
    decl_mission_file = DeclareLaunchArgument(
        "mission_file",
        default_value=str(default_mission),
        description="Full path to canonical mission YAML configuration",
    )
    decl_enable_vision_costmap = DeclareLaunchArgument(
        "enable_vision_costmap",
        default_value="false",
        description="Enable neural vision costmap ground projection into local planner",
    )
    decl_obstacles_file = DeclareLaunchArgument(
        "obstacles_file",
        default_value=str(default_obstacles_file),
        description="Full path to ground truth static obstacles JSON",
    )

    # Resolve Display & Resource paths for WSLg / Linux
    display_val = os.environ.get("DISPLAY", ":0") or ":0"
    xdg_runtime = (
        "/run/user/1000"
        if os.path.isdir("/run/user/1000")
        else os.environ.get("XDG_RUNTIME_DIR", "/mnt/wslg/runtime-dir")
    )
    resource_path = (
        f"{pkg_dir}/worlds:{pkg_dir}/models:{pkg_dir}/models/outdoor_assets:"
        f"{os.environ.get('GZ_SIM_RESOURCE_PATH', '')}"
    )

    gui_env = {
        "DISPLAY": display_val,
        "XDG_RUNTIME_DIR": xdg_runtime,
        "GZ_SIM_RESOURCE_PATH": resource_path,
    }

    set_pythonpath = SetEnvironmentVariable(
        name="PYTHONPATH",
        value=f"{workspace_dir}:{os.environ.get('PYTHONPATH', '')}",
    )
    set_display = SetEnvironmentVariable(name="DISPLAY", value=display_val)
    set_xdg = SetEnvironmentVariable(name="XDG_RUNTIME_DIR", value=xdg_runtime)
    set_resource = SetEnvironmentVariable(name="GZ_SIM_RESOURCE_PATH", value=resource_path)

    # 1. Gazebo Sim Process (GUI mode)
    gz_gui = ExecuteProcess(
        cmd=["gz", "sim", "-r", "-v", "3", world],
        name="gz_sim_gui",
        output="screen",
        additional_env=gui_env,
        condition=UnlessCondition(headless),
    )

    # 1b. Gazebo Sim Process (Headless mode)
    gz_headless = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", world],
        name="gz_sim_headless",
        output="screen",
        condition=IfCondition(headless),
    )

    # 2. ROS-GZ Parameter Bridge Node
    bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        output="screen",
        parameters=[{
            "config_file": str(bridge_config_path),
            "use_sim_time": use_sim_time,
        }],
    )

    # 3. Interactive UGV Navigation Node (100m Corridor Bounds & Visualization)
    nav_proc = ExecuteProcess(
        cmd=[
            "python3",
            str(nav_node_path),
            "--ros-args",
            "-p", "use_sim_time:=true",
            "-p", f"mission_file:={str(default_mission)}",
            "-p", "use_ekf_fusion:=false",
            "-p", "rock_x:=-100.0",
            "-p", "rock_radius:=0.0",
            "-p", "map_x_min:=-5.0",
            "-p", "map_x_max:=115.0",
            "-p", "map_y_min:=-15.0",
            "-p", "map_y_max:=15.0",
            "-p", "enable_visualization_adapter:=true",
            "-p", "costmap_publish_rate_hz:=5.0",
            "-p", "publish_robot_description:=true",
            "-p", "enable_vision_costmap:=false",
            "-p", f"obstacles_file:={str(default_obstacles_file)}",
        ],
        name="interactive_ugv_navigation",
        output="screen",
    )

    # 4. Presentation Dynamic Obstacle Controller (Autonomous Cross-Corridor Patrol)
    obstacle_proc = ExecuteProcess(
        cmd=[
            "python3",
            str(obstacle_node_path),
            "--ros-args",
            "-p", "use_sim_time:=true",
            "-p", "enable_patrol:=true",
            "-p", "patrol_speed_mps:=0.40",
            "-p", "y_min:=-7.5",
            "-p", "y_max:=1.5",
        ],
        name="presentation_obstacle_controller",
        output="screen",
        condition=IfCondition(enable_dynamic_obstacle),
    )

    # 5. RViz2 Hero Presentation Process
    # Note: on_exit=Shutdown() is intentionally NOT used here to preserve failure isolation.
    rviz_proc = ExecuteProcess(
        cmd=[
            "rviz2",
            "-d", rviz_config,
            "--ros-args",
            "-p", "use_sim_time:=true",
        ],
        name="rviz2_hero",
        output="screen",
        additional_env=gui_env,
        condition=IfCondition(rviz),
    )

    banner_log = LogInfo(
        msg="\n"
            "======================================================================\n"
            "  SIH2026 UGV AUTONOMOUS NAVIGATION — HERO PROTOTYPE PRESENTATION    \n"
            "======================================================================\n"
            "  - World:        outdoor_presentation_100m.sdf (104.65m Corridor)    \n"
            "  - Mission:      START -> W1 -> W2 -> W3 -> W4 -> W5 (100m)         \n"
            "  - Perception:   RELLIS-3D Dense Traversability Net                  \n"
            "  - Telemetry:    /costmap/occupancy, /mission/waypoint_markers       \n"
            "  - RViz Profile: hero_presentation.rviz (Target: base_link)          \n"
            "  - Dynamic Obs:  5 Corridor Obstacles (W1-W5 Oncoming/Crossing/Chicane)\n"
            "======================================================================\n"
    )

    return LaunchDescription([
        decl_use_sim_time,
        decl_headless,
        decl_world,
        decl_rviz,
        decl_rviz_config,
        decl_enable_dynamic_obstacle,
        decl_mission_file,
        decl_enable_vision_costmap,
        decl_obstacles_file,
        set_pythonpath,
        set_display,
        set_xdg,
        set_resource,
        banner_log,
        gz_gui,
        gz_headless,
        bridge_node,
        nav_proc,
        obstacle_proc,
        rviz_proc,
    ])
