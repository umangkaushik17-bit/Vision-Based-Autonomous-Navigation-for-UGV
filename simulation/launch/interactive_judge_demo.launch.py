#!/usr/bin/env python3
"""Unified Launch Orchestrator for DEMO-01: Interactive UGV Demonstration.

Launches the complete local judge-facing stack in one single command:
1. Gazebo Sim Harmonic loading 'demo_outdoor.sdf'
2. ROS-GZ Parameter Bridge connecting cmd_vel, odometry, clock, and sensors
3. Interactive UGV Navigation Node (A*, DWA, Kalman tracking, predictions, HUD overlay)
4. RViz2 pre-configured with judge visualization profile ('judge_demo.rviz')
5. Dynamic Obstacle Teleoperation node
"""

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

    world_path = pkg_dir / "worlds" / "demo_outdoor.sdf"
    bridge_config_path = pkg_dir / "config" / "bridge_params.yaml"
    rviz_config_path = pkg_dir / "rviz" / "judge_demo.rviz"
    nav_node_path = pkg_dir / "nodes" / "interactive_ugv_navigation_node.py"
    teleop_node_path = pkg_dir / "nodes" / "teleop_obstacle_keyboard.py"

    # Launch Configurations
    headless = LaunchConfiguration("headless", default="false")
    rviz = LaunchConfiguration("rviz", default="true")
    teleop = LaunchConfiguration("teleop", default="false")

    # Launch Arguments
    decl_headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo Sim in headless mode (true) or with GUI (false)",
    )
    decl_rviz = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz2 with judge display configuration",
    )
    decl_teleop = DeclareLaunchArgument(
        "teleop",
        default_value="false",
        description="Launch dynamic obstacle keyboard teleoperation node in this process",
    )

    # Environment
    set_pythonpath = SetEnvironmentVariable(
        name="PYTHONPATH",
        value=f"{workspace_dir}:{os.environ.get('PYTHONPATH', '')}",
    )

    # 1. Gazebo Sim Process (GUI mode)
    gz_gui = ExecuteProcess(
        cmd=["gz", "sim", "-r", str(world_path)],
        name="gz_sim_gui",
        output="screen",
        condition=UnlessCondition(headless),
    )

    # 1b. Gazebo Sim Process (Headless mode)
    gz_headless = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", str(world_path)],
        name="gz_sim_headless",
        output="screen",
        condition=IfCondition(headless),
    )

    # 2. ROS-GZ Bridge Node
    bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        output="screen",
        parameters=[{"config_file": str(bridge_config_path)}],
    )

    # 3. Interactive Navigation Node
    nav_node = ExecuteProcess(
        cmd=["python3", str(nav_node_path)],
        name="interactive_ugv_navigation",
        output="screen",
    )

    # 4. RViz2 Process
    rviz_proc = ExecuteProcess(
        cmd=["rviz2", "-d", str(rviz_config_path)],
        name="rviz2_judge",
        output="screen",
        condition=IfCondition(rviz),
    )

    # 5. Dynamic Obstacle Teleoperation (Optional background)
    teleop_proc = ExecuteProcess(
        cmd=["python3", str(teleop_node_path)],
        name="teleop_obstacle",
        output="screen",
        condition=IfCondition(teleop),
    )

    info_log = LogInfo(
        msg="\n"
            "==============================================================\n"
            "  DEMO-01 INTERACTIVE UGV JUDGE DEMONSTRATION LAUNCHED        \n"
            "==============================================================\n"
            "  Controls:                                                   \n"
            "    - To control dynamic obstacle: run in a separate terminal:\n"
            "        python3 simulation/nodes/teleop_obstacle_keyboard.py  \n"
            "    - Keys: W/S (X-axis), A/D (Y-axis), SPACE (stop), R (reset)\n"
            "==============================================================\n"
    )

    return LaunchDescription([
        decl_headless,
        decl_rviz,
        decl_teleop,
        set_pythonpath,
        info_log,
        gz_gui,
        gz_headless,
        bridge_node,
        nav_node,
        rviz_proc,
        teleop_proc,
    ])
