"""ROS2 Launch file for UGV Navigation Core Pipeline."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory("ugv_navigation")
    param_file = os.path.join(pkg_dir, "config", "ugv_params.yaml")

    return LaunchDescription([
        # 1. Static TF Broadcaster (Camera extrinsics)
        Node(
            package="ugv_navigation",
            executable="static_tf_node",
            name="static_tf_node",
            parameters=[param_file],
            output="screen",
        ),
        # 2. Perception Node (Phase 4/5 models)
        Node(
            package="ugv_navigation",
            executable="perception_node",
            name="perception_node",
            parameters=[param_file],
            output="screen",
        ),
        # 3. Localization Node (Phase 8 Visual Odometry + Dynamic TF)
        Node(
            package="ugv_navigation",
            executable="localization_node",
            name="localization_node",
            parameters=[param_file],
            output="screen",
        ),
        # 4. Spatial Fusion Node (Phase 9 World Grid)
        Node(
            package="ugv_navigation",
            executable="spatial_fusion_node",
            name="spatial_fusion_node",
            parameters=[param_file],
            output="screen",
        ),
    ])
