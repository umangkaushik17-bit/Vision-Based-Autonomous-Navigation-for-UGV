"""Static contract validation tests for ROS2 topic, QoS, and parameter contracts.

Validates:
1. Canonical topic naming contracts across nodes and configuration.
2. Publisher / Subscriber consistency across the dynamic node graph.
3. QoS profile specifications in qos_profiles.yaml.
4. Parameter declaration consistency between ugv_params.yaml and nodes.
5. Launch file node executable and parameter correctness.
6. Strict absence of obsolete topic names in active M13.12 node files.
7. Dynamic trajectory (/planning/dynamic_trajectory) publishing contract.
"""

import ast
import os
import yaml
import pytest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
UGV_PKG_DIR = WORKSPACE_ROOT / "ros2_ws" / "src" / "ugv_navigation"
CONFIG_DIR = UGV_PKG_DIR / "config"
NODES_DIR = UGV_PKG_DIR / "ugv_navigation" / "nodes"
LAUNCH_DIR = UGV_PKG_DIR / "launch"

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

OBSOLETE_TOPIC_STRINGS = [
    "/tracking/dynamic_tracks",
    "/planning/predicted_trajectories",
    "/planning/dynamic_occupancy",
    "/behavior/decision",
    "/replan/event",
]

DYNAMIC_NODE_FILES = [
    "dynamic_detector_node.py",
    "dynamic_tracker_node.py",
    "dynamic_predictor_node.py",
    "dynamic_occupancy_node.py",
    "dynamic_planner_node.py",
    "behavioral_state_node.py",
    "global_replanner_node.py",
]


class TestTopicContract:
    """Validate canonical topic names and absence of obsolete topics."""

    def test_ugv_params_topics_canonical(self):
        """ugv_params.yaml topics must reference canonical topic names."""
        params_file = CONFIG_DIR / "ugv_params.yaml"
        assert params_file.exists(), f"Missing {params_file}"

        with open(params_file, "r") as f:
            cfg = yaml.safe_load(f)

        topics = cfg.get("ugv_navigation", {}).get("topics", {})
        assert topics, "No topics section in ugv_params.yaml"

        assert topics["dynamic_detections"] == "/perception/dynamic_obstacles"
        assert topics["dynamic_tracks"] == "/tracking/dynamic_obstacles"
        assert topics["predicted_trajectories"] == "/prediction/obstacle_trajectories"
        assert topics["dynamic_occupancy"] == "/occupancy/dynamic_slices"
        assert topics["behavioral_decision"] == "/behavior/tactical_decision"
        assert topics["replan_event"] == "/planning/dynamic_replan_event"
        assert topics["cmd_vel"] == "/cmd_vel"
        assert topics["global_path"] == "/planning/global_path"
        assert topics["dynamic_trajectory"] == "/planning/dynamic_trajectory"

    def test_no_obsolete_topic_names_in_node_files(self):
        """Active dynamic node files must not contain obsolete topic strings."""
        for filename in DYNAMIC_NODE_FILES:
            node_path = NODES_DIR / filename
            assert node_path.exists(), f"Missing node file: {node_path}"
            content = node_path.read_text(encoding="utf-8")

            for obsolete in OBSOLETE_TOPIC_STRINGS:
                assert obsolete not in content, (
                    f"Found obsolete topic string '{obsolete}' in {filename}"
                )

    def test_standalone_global_path_not_in_active_nodes(self):
        """Nodes must not use legacy '/global_path' (must use '/planning/global_path')."""
        for filename in DYNAMIC_NODE_FILES:
            node_path = NODES_DIR / filename
            content = node_path.read_text(encoding="utf-8")
            # Ensure any occurrence of /global_path is prefixed by /planning
            lines = content.splitlines()
            for idx, line in enumerate(lines, 1):
                # Ignore comments
                if line.strip().startswith("#"):
                    continue
                if '"/global_path"' in line or "'/global_path'" in line:
                    pytest.fail(f"Found bare '/global_path' on line {idx} in {filename}")

    def test_node_publisher_subscriber_consistency(self):
        """Verify dynamic perception-planning topic pipeline matches 1:1."""
        # 1. Detector publishes -> Tracker subscribes
        det_content = (NODES_DIR / "dynamic_detector_node.py").read_text(encoding="utf-8")
        trk_content = (NODES_DIR / "dynamic_tracker_node.py").read_text(encoding="utf-8")
        assert "/perception/dynamic_obstacles" in det_content
        assert "/perception/dynamic_obstacles" in trk_content

        # 2. Tracker publishes -> Predictor & Planner subscribe
        pred_content = (NODES_DIR / "dynamic_predictor_node.py").read_text(encoding="utf-8")
        plan_content = (NODES_DIR / "dynamic_planner_node.py").read_text(encoding="utf-8")
        assert "/tracking/dynamic_obstacles" in trk_content
        assert "/tracking/dynamic_obstacles" in pred_content
        assert "/tracking/dynamic_obstacles" in plan_content

        # 3. Predictor publishes -> Occupancy & Planner subscribe
        occ_content = (NODES_DIR / "dynamic_occupancy_node.py").read_text(encoding="utf-8")
        assert "/prediction/obstacle_trajectories" in pred_content
        assert "/prediction/obstacle_trajectories" in occ_content
        assert "/prediction/obstacle_trajectories" in plan_content

        # 4. Occupancy publishes -> Replanner subscribes
        replan_content = (NODES_DIR / "global_replanner_node.py").read_text(encoding="utf-8")
        assert "/occupancy/dynamic_slices" in occ_content
        assert "/occupancy/dynamic_slices" in replan_content

        # 5. Behavior publishes -> Replanner subscribes
        beh_content = (NODES_DIR / "behavioral_state_node.py").read_text(encoding="utf-8")
        assert "/behavior/tactical_decision" in beh_content
        assert "/behavior/tactical_decision" in replan_content

        # 6. Global replanner & dynamic planner publish replan event
        assert "/planning/dynamic_replan_event" in replan_content
        assert "/planning/dynamic_replan_event" in plan_content

        # 7. Global replanner & dynamic planner publish global path
        assert "/planning/global_path" in replan_content
        assert "/planning/global_path" in plan_content

    def test_dynamic_trajectory_published_by_planner(self):
        """DynamicPlannerNode must declare publisher and publish to /planning/dynamic_trajectory."""
        content = (NODES_DIR / "dynamic_planner_node.py").read_text(encoding="utf-8")
        assert '"/planning/dynamic_trajectory"' in content
        assert "self._trajectory_pub = self.create_publisher(" in content
        assert "self._trajectory_pub.publish(" in content


class TestQoSContract:
    """Validate canonical QoS profiles in qos_profiles.yaml and dynamic node usage."""

    def test_canonical_qos_profiles_exist(self):
        """qos_profiles.yaml must define canonical base and topic profiles."""
        qos_file = CONFIG_DIR / "qos_profiles.yaml"
        assert qos_file.exists(), f"Missing {qos_file}"

        with open(qos_file, "r") as f:
            cfg = yaml.safe_load(f)

        profiles = cfg.get("qos_profiles", {})
        assert profiles, "No qos_profiles in qos_profiles.yaml"

        # Canonical base profiles
        assert "sensor_data" in profiles
        assert "reliable_state" in profiles
        assert "reliable_cmd" in profiles
        assert "transient_local" in profiles

        # Verify semantics of base profiles
        sd = profiles["sensor_data"]
        assert sd["reliability"] == "best_effort"
        assert sd["durability"] == "volatile"
        assert sd["history"] == "keep_last"
        assert sd["depth"] > 0

        rs = profiles["reliable_state"]
        assert rs["reliability"] == "reliable"
        assert rs["durability"] == "volatile"
        assert rs["history"] == "keep_last"
        assert rs["depth"] >= 1

        rc = profiles["reliable_cmd"]
        assert rc["reliability"] == "reliable"
        assert rc["durability"] == "volatile"
        assert rc["history"] == "keep_last"
        assert rc["depth"] == 1

        tl = profiles["transient_local"]
        assert tl["reliability"] == "reliable"
        assert tl["durability"] == "transient_local"
        assert tl["history"] == "keep_last"
        assert tl["depth"] == 1

    def test_topic_specific_qos_profiles(self):
        """Topic-specific QoS profiles must conform to bounded dynamic perception standards."""
        with open(CONFIG_DIR / "qos_profiles.yaml", "r") as f:
            profiles = yaml.safe_load(f).get("qos_profiles", {})

        # High-rate perception must be best-effort volatile
        for stream in ["dynamic_detections", "tracked_obstacles", "predictions", "dynamic_occupancy"]:
            assert stream in profiles, f"Missing QoS profile for {stream}"
            p = profiles[stream]
            assert p["reliability"] == "best_effort", f"{stream} must be best_effort"
            assert p["durability"] == "volatile", f"{stream} must be volatile"
            assert 1 <= p["depth"] <= 10, f"{stream} depth must be bounded (1-10)"

        # Safety command and behavioral state must be reliable
        for stream in ["cmd_vel", "behavioral_decision", "replan_event"]:
            assert stream in profiles, f"Missing QoS profile for {stream}"
            p = profiles[stream]
            assert p["reliability"] == "reliable", f"{stream} must be reliable"
            assert p["durability"] == "volatile", f"{stream} must be volatile"

        # Trajectory profile
        assert "dynamic_trajectory" in profiles
        assert profiles["dynamic_trajectory"]["reliability"] == "best_effort"
        assert profiles["dynamic_trajectory"]["depth"] == 1


class TestParameterContract:
    """Validate parameter declarations between ugv_params.yaml and dynamic nodes."""

    def test_parameter_sections_exist(self):
        """All expected parameter namespaces must exist in ugv_params.yaml."""
        with open(CONFIG_DIR / "ugv_params.yaml", "r") as f:
            cfg = yaml.safe_load(f).get("ugv_navigation", {})

        required_sections = [
            "topics",
            "detection",
            "tf_frames",
            "tracking",
            "prediction",
            "dynamic_occupancy",
            "behavior",
            "replanning",
            "safety",
        ]
        for sec in required_sections:
            assert sec in cfg, f"Missing section '{sec}' in ugv_params.yaml"

    def test_node_parameter_declarations_match_yaml(self):
        """Verify dynamic nodes declare parameters corresponding to YAML configuration."""
        with open(CONFIG_DIR / "ugv_params.yaml", "r") as f:
            cfg = yaml.safe_load(f).get("ugv_navigation", {})

        # Detector
        det_ast = ast.parse((NODES_DIR / "dynamic_detector_node.py").read_text(encoding="utf-8"))
        det_params = [
            arg.args[0].value
            for node in ast.walk(det_ast)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "declare_parameter"
            for arg in [node]
            if isinstance(arg.args[0], ast.Constant)
        ]
        assert "frame_id" in det_params
        assert "min_confidence" in det_params
        assert cfg["detection"]["frame_id"] == "base_link"
        assert cfg["detection"]["min_confidence"] == 0.30

        # Tracker
        trk_ast = ast.parse((NODES_DIR / "dynamic_tracker_node.py").read_text(encoding="utf-8"))
        trk_params = [
            arg.args[0].value
            for node in ast.walk(trk_ast)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "declare_parameter"
            for arg in [node]
            if isinstance(arg.args[0], ast.Constant)
        ]
        assert "hits_to_confirm" in trk_params
        assert "misses_to_expire" in trk_params
        assert "max_association_distance_m" in trk_params
        assert cfg["tracking"]["hits_to_confirm"] == 3
        assert cfg["tracking"]["misses_to_expire"] == 5

        # Predictor
        pred_ast = ast.parse((NODES_DIR / "dynamic_predictor_node.py").read_text(encoding="utf-8"))
        pred_params = [
            arg.args[0].value
            for node in ast.walk(pred_ast)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "declare_parameter"
            for arg in [node]
            if isinstance(arg.args[0], ast.Constant)
        ]
        assert "default_horizon_s" in pred_params
        assert "default_dt_s" in pred_params
        assert "sigma_scale" in pred_params
        assert cfg["prediction"]["default_horizon_s"] == 2.0
        assert cfg["prediction"]["default_dt_s"] == 0.25

        # Dynamic Occupancy
        occ_ast = ast.parse((NODES_DIR / "dynamic_occupancy_node.py").read_text(encoding="utf-8"))
        occ_params = [
            arg.args[0].value
            for node in ast.walk(occ_ast)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "declare_parameter"
            for arg in [node]
            if isinstance(arg.args[0], ast.Constant)
        ]
        assert "resolution_m" in occ_params
        assert "collision_cost_threshold" in occ_params
        assert "frame_id" in occ_params
        assert cfg["dynamic_occupancy"]["resolution_m"] == 0.10
        assert cfg["dynamic_occupancy"]["frame_id"] == "odom"

        # Dynamic Planner
        plan_ast = ast.parse((NODES_DIR / "dynamic_planner_node.py").read_text(encoding="utf-8"))
        plan_params = [
            arg.args[0].value
            for node in ast.walk(plan_ast)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "declare_parameter"
            for arg in [node]
            if isinstance(arg.args[0], ast.Constant)
        ]
        assert "prediction_horizon_s" in plan_params
        assert "replan_cooldown_s" in plan_params
        assert "deadlock_time_threshold_s" in plan_params
        assert "stop_distance_m" in plan_params
        assert "wait_distance_m" in plan_params
        assert cfg["behavior"]["stop_distance_m"] == 0.80
        assert cfg["behavior"]["wait_distance_m"] == 1.80

        # Behavioral State Node
        beh_ast = ast.parse((NODES_DIR / "behavioral_state_node.py").read_text(encoding="utf-8"))
        beh_params = [
            arg.args[0].value
            for node in ast.walk(beh_ast)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "declare_parameter"
            for arg in [node]
            if isinstance(arg.args[0], ast.Constant)
        ]
        assert "stop_distance_m" in beh_params
        assert "wait_distance_m" in beh_params
        assert "deadlock_time_threshold_s" in beh_params

        # Global Replanner Node
        rep_ast = ast.parse((NODES_DIR / "global_replanner_node.py").read_text(encoding="utf-8"))
        rep_params = [
            arg.args[0].value
            for node in ast.walk(rep_ast)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "declare_parameter"
            for arg in [node]
            if isinstance(arg.args[0], ast.Constant)
        ]
        assert "replan_cooldown_s" in rep_params
        assert "deadlock_threshold_s" in rep_params
        assert cfg["replanning"]["replan_cooldown_s"] == 2.0
        assert cfg["replanning"]["deadlock_threshold_s"] == 3.0


class TestFrameContract:
    """Validate canonical coordinate frame conventions across YAML and nodes."""

    def test_canonical_frame_conventions(self):
        """Verify frame conventions: base_link for local detection/cmd_vel, odom for world states."""
        with open(CONFIG_DIR / "ugv_params.yaml", "r") as f:
            cfg = yaml.safe_load(f).get("ugv_navigation", {})

        tf_frames = cfg.get("tf_frames", {})
        assert tf_frames["odom"] == "odom"
        assert tf_frames["base_link"] == "base_link"

        # Detection boundary is base_link
        assert cfg["detection"]["frame_id"] == "base_link"

        # Tracks are transformed to odom
        assert cfg["tracking"]["target_frame"] == "odom"

        # Dynamic occupancy grid is in odom
        assert cfg["dynamic_occupancy"]["frame_id"] == "odom"


class TestLaunchContract:
    """Validate dynamic_navigation.launch.py node references and setup.py entry points."""

    def test_launch_executables_match_setup_entry_points(self):
        """All node executables referenced in dynamic_navigation.launch.py must exist in setup.py."""
        setup_file = UGV_PKG_DIR / "setup.py"
        setup_content = setup_file.read_text(encoding="utf-8")

        launch_file = LAUNCH_DIR / "dynamic_navigation.launch.py"
        launch_content = launch_file.read_text(encoding="utf-8")

        # Expected dynamic pipeline executables
        dynamic_executables = [
            "static_tf_node",
            "localization_node",
            "spatial_fusion_node",
            "dynamic_detector_node",
            "dynamic_tracker_node",
            "dynamic_predictor_node",
            "dynamic_occupancy_node",
            "dynamic_planner_node",
        ]

        for exe in dynamic_executables:
            assert f'executable="{exe}"' in launch_content, f"Executable {exe} missing in launch file"
            assert f'"{exe} = ' in setup_content, f"Executable {exe} missing in setup.py entry_points"

    def test_launch_nodes_are_unique(self):
        """Launch file must not define duplicate node names."""
        launch_file = LAUNCH_DIR / "dynamic_navigation.launch.py"
        launch_ast = ast.parse(launch_file.read_text(encoding="utf-8"))

        node_names = []
        for node in ast.walk(launch_ast):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Node":
                for kw in node.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        node_names.append(kw.value.value)

        assert len(node_names) == len(set(node_names)), f"Duplicate nodes found in launch: {node_names}"
        assert len(node_names) == 8, f"Expected 8 launched nodes, found {len(node_names)}"

    def test_no_obsolete_topics_in_launch(self):
        """dynamic_navigation.launch.py must not contain obsolete topic strings."""
        launch_content = (LAUNCH_DIR / "dynamic_navigation.launch.py").read_text(encoding="utf-8")
        for obsolete in OBSOLETE_TOPIC_STRINGS:
            assert obsolete not in launch_content, (
                f"Found obsolete topic string '{obsolete}' in dynamic_navigation.launch.py"
            )
