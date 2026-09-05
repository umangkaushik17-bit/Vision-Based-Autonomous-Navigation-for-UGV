"""ROS2 Launch file for Phase 13 Dynamic Obstacle Navigation Pipeline."""

import os
from typing import List

try:
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    LAUNCH_AVAILABLE = True
except ImportError:
    LAUNCH_AVAILABLE = False


def generate_launch_description():
    """Generates launch description for dynamic navigation stack."""
    if not LAUNCH_AVAILABLE:
        return None

    try:
        pkg_dir = get_package_share_directory("ugv_navigation")
        default_param_file = os.path.join(pkg_dir, "config", "ugv_params.yaml")
    except Exception:
        default_param_file = "config/ugv_params.yaml"

    param_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_param_file,
        description="Full path to the ROS2 parameters YAML file",
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )

    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        param_file_arg,
        use_sim_time_arg,

        # 1. Static TF Broadcaster (odom -> base_link -> camera_link -> camera_optical)
        Node(
            package="ugv_navigation",
            executable="static_tf_node",
            name="static_tf_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen",
        ),

        # 2. Localization Node (Visual Odometry & dynamic TF2 broadcast)
        Node(
            package="ugv_navigation",
            executable="localization_node",
            name="localization_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen",
        ),

        # 3. Spatial Fusion Node (Phase 9 World Terrain Map)
        Node(
            package="ugv_navigation",
            executable="spatial_fusion_node",
            name="spatial_fusion_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen",
        ),

        # 4. Dynamic Detector Node (Emits /perception/dynamic_obstacles)
        Node(
            package="ugv_navigation",
            executable="dynamic_detector_node",
            name="dynamic_detector_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen",
        ),

        # 5. Dynamic Tracker Node (Emits /tracking/dynamic_obstacles)
        Node(
            package="ugv_navigation",
            executable="dynamic_tracker_node",
            name="dynamic_tracker_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen",
        ),

        # 6. Dynamic Predictor Node (Emits /prediction/obstacle_trajectories)
        Node(
            package="ugv_navigation",
            executable="dynamic_predictor_node",
            name="dynamic_predictor_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen",
        ),

        # 7. Dynamic Occupancy Node (Emits /occupancy/dynamic_slices)
        Node(
            package="ugv_navigation",
            executable="dynamic_occupancy_node",
            name="dynamic_occupancy_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen",
        ),

        # 8. Dynamic Planner Node (Synchronous DWA + Behavioral SM + Global Replanning + Safety Clamp)
        Node(
            package="ugv_navigation",
            executable="dynamic_planner_node",
            name="dynamic_planner_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen",
        ),
    ])
