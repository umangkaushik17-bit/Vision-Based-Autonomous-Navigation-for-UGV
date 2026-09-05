"""Structural ROS2 node graph and communication wiring tests for Phase 13.12.

Validates the full node communication graph, publisher/subscriber contracts,
publisher ownership, QoS consistency, coordinate frame flow, and launch wiring
without requiring DDS or native ROS2 execution.
"""

import ast
import os
import sys
from pathlib import Path
import pytest
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
UGV_PKG_DIR = WORKSPACE_ROOT / "ros2_ws" / "src" / "ugv_navigation"
CONFIG_DIR = UGV_PKG_DIR / "config"
NODES_DIR = UGV_PKG_DIR / "ugv_navigation" / "nodes"
LAUNCH_DIR = UGV_PKG_DIR / "launch"

# Canonical topic set defined in M13.12.4
CANONICAL_TOPICS = {
    "/perception/dynamic_obstacles",
    "/tracking/dynamic_obstacles",
    "/prediction/obstacle_trajectories",
    "/occupancy/dynamic_slices",
    "/behavior/tactical_decision",
    "/planning/dynamic_replan_event",
    "/planning/global_path",
    "/planning/dynamic_trajectory",
    "/cmd_vel",
    "/localization/status",
    "/terrain/world_map",
}

DYNAMIC_NODES = {
    "dynamic_detector_node": "ugv_navigation.nodes.dynamic_detector_node:main",
    "dynamic_tracker_node": "ugv_navigation.nodes.dynamic_tracker_node:main",
    "dynamic_predictor_node": "ugv_navigation.nodes.dynamic_predictor_node:main",
    "dynamic_occupancy_node": "ugv_navigation.nodes.dynamic_occupancy_node:main",
    "dynamic_planner_node": "ugv_navigation.nodes.dynamic_planner_node:main",
    "behavioral_state_node": "ugv_navigation.nodes.behavioral_state_node:main",
    "global_replanner_node": "ugv_navigation.nodes.global_replanner_node:main",
}


def _extract_pub_sub(filepath: Path):
    """Extract topic strings for create_publisher and create_subscription calls."""
    content = filepath.read_text(encoding="utf-8")
    tree = ast.parse(content)
    pubs = set()
    subs = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            fn_name = node.func.attr
            if fn_name == "create_publisher":
                # create_publisher(msg_type, topic, qos)
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    pubs.add(node.args[1].value)
            elif fn_name == "create_subscription":
                # create_subscription(msg_type, topic, callback, qos)
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    subs.add(node.args[1].value)
    return pubs, subs


class TestRos2NodeGraph:
    """Validate full ROS2 communication graph and node wiring."""

    # 1. Node Names and Entry Points
    def test_all_dynamic_nodes_in_setup_entry_points(self):
        """All 7 Phase 13 dynamic nodes must be declared as console_scripts in setup.py."""
        setup_file = UGV_PKG_DIR / "setup.py"
        assert setup_file.exists()
        content = setup_file.read_text(encoding="utf-8")

        for exe_name, entry_point in DYNAMIC_NODES.items():
            expected = f'"{exe_name} = {entry_point}"'
            assert expected in content, f"Missing or mismatched entry point for {exe_name}: expected {expected}"

    def test_node_files_exist_and_importable(self):
        """All dynamic node Python modules must exist and be syntactically valid."""
        for exe_name in DYNAMIC_NODES:
            node_file = NODES_DIR / f"{exe_name}.py"
            assert node_file.exists(), f"Missing node file: {node_file}"
            # Ensure file parses as valid Python AST
            tree = ast.parse(node_file.read_text(encoding="utf-8"))
            assert tree is not None

    # 2. Logical Dynamic Perception Pipeline Flow
    def test_dynamic_perception_pipeline_flow(self):
        """Verify strict chain: Perception -> Tracking -> Prediction -> Occupancy."""
        det_pubs, det_subs = _extract_pub_sub(NODES_DIR / "dynamic_detector_node.py")
        trk_pubs, trk_subs = _extract_pub_sub(NODES_DIR / "dynamic_tracker_node.py")
        pred_pubs, pred_subs = _extract_pub_sub(NODES_DIR / "dynamic_predictor_node.py")
        occ_pubs, occ_subs = _extract_pub_sub(NODES_DIR / "dynamic_occupancy_node.py")

        # Step 1: Detector publishes /perception/dynamic_obstacles
        assert "/perception/dynamic_obstacles" in det_pubs

        # Step 2: Tracker subscribes /perception/dynamic_obstacles, publishes /tracking/dynamic_obstacles
        assert "/perception/dynamic_obstacles" in trk_subs
        assert "/tracking/dynamic_obstacles" in trk_pubs

        # Step 3: Predictor subscribes /tracking/dynamic_obstacles, publishes /prediction/obstacle_trajectories
        assert "/tracking/dynamic_obstacles" in pred_subs
        assert "/prediction/obstacle_trajectories" in pred_pubs

        # Step 4: Occupancy subscribes /prediction/obstacle_trajectories, publishes /occupancy/dynamic_slices
        assert "/prediction/obstacle_trajectories" in occ_subs
        assert "/occupancy/dynamic_slices" in occ_pubs

    # 3. Dynamic Information to Planning & Behavior Consumers
    def test_dynamic_information_reaches_planner_and_behavior(self):
        """Verify dynamic obstacle channels reach DynamicPlannerNode, BehavioralStateNode, and GlobalReplannerNode."""
        plan_pubs, plan_subs = _extract_pub_sub(NODES_DIR / "dynamic_planner_node.py")
        beh_pubs, beh_subs = _extract_pub_sub(NODES_DIR / "behavioral_state_node.py")
        rep_pubs, rep_subs = _extract_pub_sub(NODES_DIR / "global_replanner_node.py")

        # DynamicPlannerNode subscribes to tracks and predictions
        assert "/tracking/dynamic_obstacles" in plan_subs
        assert "/prediction/obstacle_trajectories" in plan_subs

        # BehavioralStateNode subscribes to tracks and predictions
        assert "/tracking/dynamic_obstacles" in beh_subs
        assert "/prediction/obstacle_trajectories" in beh_subs

        # GlobalReplannerNode subscribes to occupancy, tracks, predictions, and behavioral decision
        assert "/occupancy/dynamic_slices" in rep_subs
        assert "/tracking/dynamic_obstacles" in rep_subs
        assert "/prediction/obstacle_trajectories" in rep_subs
        assert "/behavior/tactical_decision" in rep_subs

    # 4. Context Inputs Connected to Consumers
    def test_context_inputs_connected(self):
        """Verify /localization/status, /terrain/world_map, and /planning/global_path reach planning nodes."""
        plan_pubs, plan_subs = _extract_pub_sub(NODES_DIR / "dynamic_planner_node.py")
        rep_pubs, rep_subs = _extract_pub_sub(NODES_DIR / "global_replanner_node.py")

        # DynamicPlannerNode consumes localization and world map
        assert "/localization/status" in plan_subs
        assert "/terrain/world_map" in plan_subs

        # GlobalReplannerNode consumes localization and world map, and produces global path
        assert "/localization/status" in rep_subs
        assert "/terrain/world_map" in rep_subs
        assert "/planning/global_path" in rep_pubs

    # 5. Final Control Path (/cmd_vel) Ownership
    def test_final_control_path_cmd_vel_ownership(self):
        """Verify /cmd_vel has exactly one authoritative publisher in the launched dynamic pipeline."""
        plan_pubs, _ = _extract_pub_sub(NODES_DIR / "dynamic_planner_node.py")
        det_pubs, _ = _extract_pub_sub(NODES_DIR / "dynamic_detector_node.py")
        trk_pubs, _ = _extract_pub_sub(NODES_DIR / "dynamic_tracker_node.py")
        pred_pubs, _ = _extract_pub_sub(NODES_DIR / "dynamic_predictor_node.py")
        occ_pubs, _ = _extract_pub_sub(NODES_DIR / "dynamic_occupancy_node.py")
        beh_pubs, _ = _extract_pub_sub(NODES_DIR / "behavioral_state_node.py")
        rep_pubs, _ = _extract_pub_sub(NODES_DIR / "global_replanner_node.py")

        # DynamicPlannerNode is the authoritative /cmd_vel publisher
        assert "/cmd_vel" in plan_pubs

        # Perception, tracking, prediction, occupancy, behavior, replanner must NOT publish /cmd_vel
        assert "/cmd_vel" not in det_pubs
        assert "/cmd_vel" not in trk_pubs
        assert "/cmd_vel" not in pred_pubs
        assert "/cmd_vel" not in occ_pubs
        assert "/cmd_vel" not in beh_pubs
        assert "/cmd_vel" not in rep_pubs

    # 6. Publisher Ownership Audit: /behavior/tactical_decision
    def test_publisher_ownership_tactical_decision(self):
        """Verify /behavior/tactical_decision publisher ownership between monolithic and distributed nodes."""
        beh_pubs, _ = _extract_pub_sub(NODES_DIR / "behavioral_state_node.py")
        plan_pubs, _ = _extract_pub_sub(NODES_DIR / "dynamic_planner_node.py")
        rep_pubs, rep_subs = _extract_pub_sub(NODES_DIR / "global_replanner_node.py")

        # BehavioralStateNode is the standalone authoritative source for tactical decisions
        assert "/behavior/tactical_decision" in beh_pubs
        # GlobalReplannerNode consumes it
        assert "/behavior/tactical_decision" in rep_subs
        # DynamicPlannerNode also publishes its local behavioral decision
        assert "/behavior/tactical_decision" in plan_pubs

        # Launch file isolation: Launch file must NOT launch both simultaneously as competing command publishers
        launch_file = LAUNCH_DIR / "dynamic_navigation.launch.py"
        launch_content = launch_file.read_text(encoding="utf-8")
        # In the active launch file, dynamic_planner_node is launched as the integrated planner
        assert 'executable="dynamic_planner_node"' in launch_content
        # behavioral_state_node is NOT launched in dynamic_navigation.launch.py, preventing duplicate publishers
        assert 'executable="behavioral_state_node"' not in launch_content

    # 7. Publisher Ownership Audit: /planning/dynamic_replan_event
    def test_publisher_ownership_replan_event(self):
        """Verify /planning/dynamic_replan_event publisher ownership."""
        rep_pubs, _ = _extract_pub_sub(NODES_DIR / "global_replanner_node.py")
        plan_pubs, _ = _extract_pub_sub(NODES_DIR / "dynamic_planner_node.py")

        assert "/planning/dynamic_replan_event" in rep_pubs
        assert "/planning/dynamic_replan_event" in plan_pubs

        # Both use the canonical topic name, matching QoS (reliable, volatile, depth 10)
        with open(CONFIG_DIR / "qos_profiles.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        replan_qos = cfg["qos_profiles"]["replan_event"]
        assert replan_qos["reliability"] == "reliable"
        assert replan_qos["durability"] == "volatile"

        # Launch isolation: Launch file launches dynamic_planner_node, not global_replanner_node
        launch_content = (LAUNCH_DIR / "dynamic_navigation.launch.py").read_text(encoding="utf-8")
        assert 'executable="global_replanner_node"' not in launch_content

    # 8. No Circular Dependencies in Dynamic Pipeline
    def test_no_circular_dependencies_in_dynamic_pipeline(self):
        """Verify the dynamic perception-planning graph is a DAG with no circular feedback loops."""
        # Define forward edges (producer -> topic -> consumer)
        # Perception -> Tracking -> Prediction -> Occupancy -> Planning
        # Verify no upstream node subscribes to a downstream topic
        _, det_subs = _extract_pub_sub(NODES_DIR / "dynamic_detector_node.py")
        _, trk_subs = _extract_pub_sub(NODES_DIR / "dynamic_tracker_node.py")
        _, pred_subs = _extract_pub_sub(NODES_DIR / "dynamic_predictor_node.py")
        _, occ_subs = _extract_pub_sub(NODES_DIR / "dynamic_occupancy_node.py")

        # Detector must not depend on tracking, prediction, occupancy, or planning
        for t in ["/tracking/dynamic_obstacles", "/prediction/obstacle_trajectories", "/occupancy/dynamic_slices", "/cmd_vel"]:
            assert t not in det_subs

        # Tracker must not depend on prediction, occupancy, or planning
        for t in ["/prediction/obstacle_trajectories", "/occupancy/dynamic_slices", "/cmd_vel"]:
            assert t not in trk_subs

        # Predictor must not depend on occupancy or planning
        for t in ["/occupancy/dynamic_slices", "/cmd_vel"]:
            assert t not in pred_subs

        # Occupancy must not depend on planning
        assert "/cmd_vel" not in occ_subs

    # 9. Canonical Topics Completeness
    def test_all_canonical_topics_covered(self):
        """All 11 canonical topics must be present in active node publishers/subscribers."""
        all_pubs = set()
        all_subs = set()
        for filename in NODES_DIR.glob("*.py"):
            pubs, subs = _extract_pub_sub(filename)
            all_pubs.update(pubs)
            all_subs.update(subs)

        for topic in CANONICAL_TOPICS:
            in_pubs = topic in all_pubs
            in_subs = topic in all_subs
            assert in_pubs or in_subs, f"Canonical topic {topic} neither published nor subscribed anywhere"

    # 10. QoS Consistency Across Connected Topics
    def test_qos_consistency_across_connected_topics(self):
        """Validate QoS profiles defined for each topic meet DDS compatibility rules."""
        with open(CONFIG_DIR / "qos_profiles.yaml", "r") as f:
            cfg = yaml.safe_load(f)["qos_profiles"]

        # High-rate data streams must use best_effort / volatile
        for stream in ["dynamic_detections", "tracked_obstacles", "predictions", "dynamic_occupancy"]:
            p = cfg[stream]
            assert p["reliability"] == "best_effort"
            assert p["durability"] == "volatile"

        # Critical decision & actuation streams must use reliable / volatile
        for stream in ["cmd_vel", "behavioral_decision", "replan_event"]:
            p = cfg[stream]
            assert p["reliability"] == "reliable"
            assert p["durability"] == "volatile"


class TestPackageBuildAudit:
    """Validate ROS2 package architecture, manifests, CMakeLists, and setup metadata."""

    def test_ugv_navigation_msgs_manifest(self):
        """ugv_navigation_msgs/package.xml must be valid format 3 ament_cmake interface package."""
        import xml.etree.ElementTree as ET
        manifest = WORKSPACE_ROOT / "ros2_ws" / "src" / "ugv_navigation_msgs" / "package.xml"
        assert manifest.exists()
        tree = ET.parse(manifest)
        root = tree.getroot()
        assert root.tag == "package"
        assert root.attrib.get("format") == "3"

        buildtool_depends = [el.text for el in root.findall("buildtool_depend")]
        assert "ament_cmake" in buildtool_depends
        assert "rosidl_default_generators" in buildtool_depends

        exec_depends = [el.text for el in root.findall("exec_depend")]
        assert "rosidl_default_runtime" in exec_depends

        member_groups = [el.text for el in root.findall("member_of_group")]
        assert "rosidl_interface_packages" in member_groups

        build_types = [el.text for el in root.findall(".//build_type")]
        assert "ament_cmake" in build_types

    def test_ugv_navigation_msgs_cmake(self):
        """ugv_navigation_msgs/CMakeLists.txt must generate all 13 canonical .msg interfaces."""
        cmake_file = WORKSPACE_ROOT / "ros2_ws" / "src" / "ugv_navigation_msgs" / "CMakeLists.txt"
        assert cmake_file.exists()
        content = cmake_file.read_text(encoding="utf-8")

        msg_dir = WORKSPACE_ROOT / "ros2_ws" / "src" / "ugv_navigation_msgs" / "msg"
        all_msg_files = [f.name for f in msg_dir.glob("*.msg")]
        assert len(all_msg_files) == 13

        for msg_name in all_msg_files:
            assert f'"msg/{msg_name}"' in content, f"{msg_name} not listed in CMakeLists.txt msg_files"

        assert "rosidl_generate_interfaces" in content
        assert "ament_export_dependencies(rosidl_default_runtime)" in content
        assert "ament_package()" in content

    def test_ugv_navigation_manifest(self):
        """ugv_navigation/package.xml must be valid ament_python with ugv_navigation_msgs dependency."""
        import xml.etree.ElementTree as ET
        manifest = WORKSPACE_ROOT / "ros2_ws" / "src" / "ugv_navigation" / "package.xml"
        assert manifest.exists()
        tree = ET.parse(manifest)
        root = tree.getroot()

        buildtool_depends = [el.text for el in root.findall("buildtool_depend")]
        assert "ament_python" in buildtool_depends

        depends = [el.text for el in root.findall("depend")]
        assert "ugv_navigation_msgs" in depends
        assert "rclpy" in depends
        assert "geometry_msgs" in depends
        assert "tf2_ros" in depends

        build_types = [el.text for el in root.findall(".//build_type")]
        assert "ament_python" in build_types

    def test_ugv_navigation_setup_and_resource(self):
        """ugv_navigation setup.py and resource/ index must be complete and valid."""
        setup_file = WORKSPACE_ROOT / "ros2_ws" / "src" / "ugv_navigation" / "setup.py"
        assert setup_file.exists()
        content = setup_file.read_text(encoding="utf-8")
        assert "share/ament_index/resource_index/packages" in content
        assert "launch/*.launch.py" in content
        assert "config/*.yaml" in content

        resource_file = WORKSPACE_ROOT / "ros2_ws" / "src" / "ugv_navigation" / "resource" / "ugv_navigation"
        assert resource_file.exists()

