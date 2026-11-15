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
from typing import Any, List, Optional

try:
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
except Exception:
    class LaunchDescription:  # type: ignore[no-redef]
        def __init__(self, entities: Optional[List[Any]] = None) -> None:
            self.entities = list(entities or [])

    class DeclareLaunchArgument:  # type: ignore[no-redef]
        def __init__(self, name: str, default_value: Any = None, description: str = "", **kwargs: Any) -> None:
            self.name = name
            self.default_value = default_value
            self.description = description

    class ExecuteProcess:  # type: ignore[no-redef]
        def __init__(self, cmd: List[Any], name: Optional[str] = None, output: str = "screen",
                     additional_env: Any = None, condition: Any = None, on_exit: Any = None, **kwargs: Any) -> None:
            self.cmd = cmd
            self.name = name
            self.output = output
            self.additional_env = additional_env
            self.condition = condition
            self.on_exit = on_exit

    class LogInfo:  # type: ignore[no-redef]
        def __init__(self, msg: Any = "", **kwargs: Any) -> None:
            self.msg = msg

    class SetEnvironmentVariable:  # type: ignore[no-redef]
        def __init__(self, name: str, value: Any = "", **kwargs: Any) -> None:
            self.name = name
            self.value = value

    class IfCondition:  # type: ignore[no-redef]
        def __init__(self, predicate: Any) -> None:
            self.predicate = predicate

    class UnlessCondition:  # type: ignore[no-redef]
        def __init__(self, predicate: Any) -> None:
            self.predicate = predicate

    class LaunchConfiguration:  # type: ignore[no-redef]
        def __init__(self, variable_name: str, default: Any = None) -> None:
            self.variable_name = variable_name
            self.default = default
            self.text = str(default) if default is not None else f"$({variable_name})"

        def __str__(self) -> str:
            return self.text

    class Node:  # type: ignore[no-redef]
        def __init__(self, package: str = "", executable: str = "", name: Optional[str] = None,
                     output: str = "screen", parameters: Optional[List[Any]] = None, **kwargs: Any) -> None:
            self.package = package
            self.executable = executable
            self.name = name
            self._Node__node_name = name
            self.output = output
            self.parameters = parameters or []


def generate_launch_description() -> LaunchDescription:
    pkg_dir = Path(__file__).resolve().parent.parent
    workspace_dir = pkg_dir.parent

    default_world = pkg_dir / "worlds" / "outdoor_uneven_terrain_100m.sdf"
    if not default_world.exists():
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
    use_ekf_fusion = LaunchConfiguration("use_ekf_fusion", default="true")
    enable_2_5d_terrain = LaunchConfiguration("enable_2_5d_terrain", default="true")
    terrain_world_mode = LaunchConfiguration("terrain_world_mode", default="uneven_100m")

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
    decl_use_ekf_fusion = DeclareLaunchArgument(
        "use_ekf_fusion",
        default_value="true",
        description="Enable 15-state GPS-denied EKF multi-sensor localization fusion",
    )
    decl_obstacles_file = DeclareLaunchArgument(
        "obstacles_file",
        default_value=str(default_obstacles_file),
        description="Full path to ground truth static obstacles JSON",
    )
    decl_enable_2_5d_terrain = DeclareLaunchArgument(
        "enable_2_5d_terrain",
        default_value="true",
        description="Enable Phase 15/16 2.5D elevation, slope, step, ditch, and belly clearance pipeline",
    )
    decl_terrain_world_mode = DeclareLaunchArgument(
        "terrain_world_mode",
        default_value="uneven_100m",
        description="2.5D terrain world mode ('uneven_100m' or 'flat_100m')",
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
            "-p", ["use_ekf_fusion:=", use_ekf_fusion],
            "-p", "rock_x:=-100.0",
            "-p", "rock_radius:=0.0",
            "-p", "map_x_min:=-5.0",
            "-p", "map_x_max:=115.0",
            "-p", "map_y_min:=-15.0",
            "-p", "map_y_max:=15.0",
            "-p", "enable_visualization_adapter:=true",
            "-p", "costmap_publish_rate_hz:=5.0",
            "-p", "publish_robot_description:=true",
            "-p", ["enable_vision_costmap:=", enable_vision_costmap],
            "-p", ["enable_2_5d_terrain:=", enable_2_5d_terrain],
            "-p", ["terrain_world_mode:=", terrain_world_mode],
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
            "  - World:        outdoor_uneven_terrain_100m.sdf (2.5D 100m Corridor)\n"
            "  - Mission:      START -> W1 -> W2 -> W3 -> W4 -> W5 (100m)         \n"
            "  - Perception:   RELLIS-3D Dense Traversability + 2.5D Elevation    \n"
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
        decl_use_ekf_fusion,
        decl_obstacles_file,
        decl_enable_2_5d_terrain,
        decl_terrain_world_mode,
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
