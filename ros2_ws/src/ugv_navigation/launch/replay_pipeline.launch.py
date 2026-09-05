"""ROS2 Launch file for Offline Replay Pipeline."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory("ugv_navigation")
    param_file = os.path.join(pkg_dir, "config", "ugv_params.yaml")

    return LaunchDescription([
        # 1. Offline Replay Node
        Node(
            package="ugv_navigation",
            executable="replay_node",
            name="replay_node",
            parameters=[param_file],
            output="screen",
        ),
        # 2. Static TF Broadcaster
        Node(
            package="ugv_navigation",
            executable="static_tf_node",
            name="static_tf_node",
            parameters=[param_file],
            output="screen",
        ),
        # 3. Perception Node
        Node(
            package="ugv_navigation",
            executable="perception_node",
            name="perception_node",
            parameters=[param_file],
            output="screen",
        ),
        # 4. Localization Node
        Node(
            package="ugv_navigation",
            executable="localization_node",
            name="localization_node",
            parameters=[param_file],
            output="screen",
        ),
        # 5. Spatial Fusion Node
        Node(
            package="ugv_navigation",
            executable="spatial_fusion_node",
            name="spatial_fusion_node",
            parameters=[param_file],
            output="screen",
        ),
    ])
