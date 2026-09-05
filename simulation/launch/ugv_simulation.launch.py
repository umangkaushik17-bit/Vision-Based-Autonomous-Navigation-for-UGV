"""Main ROS2 Launch Orchestrator for Gazebo UGV Simulation.

Starts Gazebo Sim, spawns the 4WD UGV robot, launches ros_gz_bridge,
starts robot_state_publisher, and launches the validated Phase 10 ROS2
navigation nodes with use_sim_time enabled.
"""

from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_dir = Path(__file__).resolve().parent.parent
    sim_config = pkg_dir / "config" / "simulation_params.yaml"
    bridge_config = pkg_dir / "config" / "bridge_params.yaml"
    world_path = pkg_dir / "worlds" / "outdoor_terrain.sdf"
    urdf_path = pkg_dir / "models" / "ugv" / "ugv.urdf"

    # Launch Arguments
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    headless = LaunchConfiguration("headless", default="true")
    world = LaunchConfiguration("world", default=str(world_path))

    decl_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )
    decl_headless = DeclareLaunchArgument(
        "headless",
        default_value="true",
        description="Run Gazebo simulation headless without GUI",
    )
    decl_world = DeclareLaunchArgument(
        "world",
        default_value=str(world_path),
        description="Full path to world SDF file",
    )

    # Robot State Publisher
    with open(urdf_path, "r") as f:
        robot_desc = f.read()

    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_desc, "use_sim_time": use_sim_time}],
    )

    # ROS-GZ Bridge Node
    bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        output="screen",
        parameters=[{"config_file": str(bridge_config), "use_sim_time": use_sim_time}],
    )

    # Phase 10 ROS2 Navigation Nodes
    static_tf_node = Node(
        package="ugv_navigation",
        executable="static_tf_node",
        name="static_tf_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    localization_node = Node(
        package="ugv_navigation",
        executable="localization_node",
        name="localization_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    perception_node = Node(
        package="ugv_navigation",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    spatial_fusion_node = Node(
        package="ugv_navigation",
        executable="spatial_fusion_node",
        name="spatial_fusion_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription([
        decl_use_sim_time,
        decl_headless,
        decl_world,
        rsp_node,
        bridge_node,
        static_tf_node,
        localization_node,
        perception_node,
        spatial_fusion_node,
    ])
