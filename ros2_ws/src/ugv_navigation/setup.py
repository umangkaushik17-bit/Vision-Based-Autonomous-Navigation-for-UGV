from setuptools import find_packages, setup
import os
from glob import glob

package_name = "ugv_navigation"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SIH26126 Team",
    maintainer_email="shres@local.host",
    description="ROS2 Middleware, TF2 Integration, and Dynamic Obstacle Navigation for Autonomous Outdoor UGV",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "replay_node = ugv_navigation.nodes.replay_node:main",
            "perception_node = ugv_navigation.nodes.perception_node:main",
            "localization_node = ugv_navigation.nodes.localization_node:main",
            "spatial_fusion_node = ugv_navigation.nodes.spatial_fusion_node:main",
            "static_tf_node = ugv_navigation.nodes.static_tf_node:main",
            "planner_node = ugv_navigation.nodes.planner_node:main",
            "dynamic_detector_node = ugv_navigation.nodes.dynamic_detector_node:main",
            "dynamic_tracker_node = ugv_navigation.nodes.dynamic_tracker_node:main",
            "dynamic_predictor_node = ugv_navigation.nodes.dynamic_predictor_node:main",
            "dynamic_occupancy_node = ugv_navigation.nodes.dynamic_occupancy_node:main",
            "dynamic_planner_node = ugv_navigation.nodes.dynamic_planner_node:main",
            "behavioral_state_node = ugv_navigation.nodes.behavioral_state_node:main",
            "global_replanner_node = ugv_navigation.nodes.global_replanner_node:main",
        ],
    },
)
