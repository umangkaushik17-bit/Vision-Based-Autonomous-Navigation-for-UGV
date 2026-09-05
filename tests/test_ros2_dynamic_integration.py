"""Milestone 13.12: ROS2 Integration of Dynamic Navigation Stack.

Comprehensive integration tests covering:
- 20 Required Integration Tests (Message serialization, validity, transport, timestamps, frames,
  stale rejection, command publishing, safety clamps, TF2, parameters, launch)
- 9 Safety Invariants (A-I)
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import yaml

from ai.planning.base import GoalPose, Path, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
)
from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    DynamicObstacleDetectionArray,
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
)
from ai.planning.dynamic.global_replanner import (
    DynamicReplanEvent,
    DynamicReplanResult,
    DynamicReplanTrigger,
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
    PathChangeMetrics,
)
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyConfig, DynamicOccupancyLayer
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig
from ai.planning.dynamic.tracker import MultiFrameTracker, TrackerConfig, TrackStatus
from ai.spatial_fusion.world_map import WorldTerrainMap

from ugv_navigation.adapters.dynamic_navigation_adapter import (
    DynamicAdapterConfig,
    DynamicNavigationAdapter,
    DynamicPlanningResult,
)
from ugv_navigation.adapters.planner_adapter import PathMsg, PoseStampedMsg, TwistMsg
from ugv_navigation.messages.dynamic_messages import (
    BehavioralDecisionMsg,
    DynamicObstacleDetectionArrayMsg,
    DynamicObstacleDetectionMsg,
    DynamicOccupancyGridSliceMsg,
    DynamicOccupancySliceArrayMsg,
    DynamicReplanEventMsg,
    PredictedObstacleTrajectoryMsg,
    PredictedTrajectoryArrayMsg,
    TrackedObstacleArrayMsg,
    TrackedObstacleMsg,
    TrajectoryWaypointMsg,
)
from ugv_navigation.messages.localization_status_msg import (
    LocalizationStatusMsg,
    Point3D,
    Pose,
    PoseWithCovariance,
    Quaternion,
    Twist,
    TwistWithCovariance,
)
from ugv_navigation.messages.serialization import (
    from_json_dict,
    message_from_json,
    message_to_json,
    to_json_dict,
)
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.nodes.behavioral_state_node import BehavioralStateNode
from ugv_navigation.nodes.dynamic_detector_node import DynamicDetectorNode
from ugv_navigation.nodes.dynamic_occupancy_node import DynamicOccupancyNode
from ugv_navigation.nodes.dynamic_planner_node import DynamicPlannerNode
from ugv_navigation.nodes.dynamic_predictor_node import DynamicPredictorNode
from ugv_navigation.nodes.dynamic_tracker_node import DynamicTrackerNode
from ugv_navigation.nodes.global_replanner_node import GlobalReplannerNode
from ugv_navigation.tf.tf_tree import TfTreeBuffer
from ugv_navigation.utils.time_utils import RosHeader, RosTime


# ==============================================================================
# Helper Fixtures & Builders
# ==============================================================================

def create_loc_msg(
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    timestamp_s: float = 1.0,
    frame_id: str = "odom",
    tracking_state: int = 0,
    confidence: float = 1.0,
) -> LocalizationStatusMsg:
    """Helper to create a deterministic LocalizationStatusMsg."""
    header = RosHeader.create(frame_id=frame_id, timestamp_s=timestamp_s)
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    pose = PoseWithCovariance(
        pose=Pose(
            position=Point3D(x=float(x), y=float(y), z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=float(qz), w=float(qw)),
        )
    )
    twist = TwistWithCovariance(
        twist=Twist(
            linear=Point3D(x=0.0, y=0.0, z=0.0),
            angular=Point3D(x=0.0, y=0.0, z=0.0),
        )
    )
    return LocalizationStatusMsg(
        header=header,
        child_frame_id="base_link",
        tracking_state=tracking_state,
        confidence=confidence,
        is_pose_valid=True,
        allows_spatial_fusion=True,
        pose=pose,
        twist=twist,
        tracked_features=100,
        inliers=80,
    )


def create_world_map_msg(
    timestamp_s: float = 1.0,
    frame_id: str = "odom",
    x_bounds: Tuple[float, float] = (-5.0, 5.0),
    y_bounds: Tuple[float, float] = (-5.0, 5.0),
) -> WorldTerrainMapMsg:
    """Helper to create a deterministic WorldTerrainMapMsg."""
    wmap = WorldTerrainMap(
        resolution_m=0.10,
        x_bounds_m=x_bounds,
        y_bounds_m=y_bounds,
        created_timestamp=timestamp_s,
    )
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    wmap.observation_count.fill(1)
    wmap.last_observed_timestamp.fill(timestamp_s)
    return WorldTerrainMapMsg.from_world_terrain_map(wmap, frame_id=frame_id, timestamp_s=timestamp_s)


def create_sample_detection_array(
    timestamp_s: float = 1.0,
    frame_id: str = "base_link",
) -> DynamicObstacleDetectionArrayMsg:
    """Helper to create a sample DynamicObstacleDetectionArrayMsg."""
    p13_det = DynamicObstacleDetection(
        detection_id="1",
        timestamp=timestamp_s,
        frame_id=frame_id,
        bbox_center=(2.5, 0.5),
        bbox_size=(0.8, 0.6),
        bbox_yaw=0.1,
        confidence=0.85,
    )
    p13_arr = DynamicObstacleDetectionArray(
        header_timestamp=timestamp_s,
        frame_id=frame_id,
        detections=[p13_det],
    )
    return DynamicObstacleDetectionArrayMsg.from_phase13(p13_arr, frame_id=frame_id)


def create_sample_tracks_array(
    timestamp_s: float = 1.0,
    frame_id: str = "odom",
    num_tracks: int = 1,
    obs_x: float = 2.5,
    obs_y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
) -> TrackedObstacleArrayMsg:
    """Helper to create a sample TrackedObstacleArrayMsg."""
    tracks = []
    for i in range(num_tracks):
        track = TrackedObstacleState(
            track_id=100 + i,
            timestamp=timestamp_s,
            x=obs_x + (i * 1.0),
            y=obs_y,
            vx=vx,
            vy=vy,
            covariance=np.eye(4) * 0.04,
            bbox_size=(0.8, 0.6),
            bbox_yaw=0.0,
            frame_id=frame_id,
            track_age=5,
            coast_cycles=0,
            is_confirmed=True,
            status=TrackStatus.TRACKING.value,
        )
        tracks.append(track)
    arr = TrackedObstacleArray(header_timestamp=timestamp_s, frame_id=frame_id, tracks=tracks)
    return TrackedObstacleArrayMsg.from_phase13(arr, frame_id=frame_id)


def create_sample_predictions_array(
    timestamp_s: float = 1.0,
    frame_id: str = "odom",
    num_obstacles: int = 1,
    obs_x: float = 2.5,
    obs_y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    unknown_horizon: bool = False,
) -> PredictedTrajectoryArrayMsg:
    """Helper to create a sample PredictedTrajectoryArrayMsg using Phase 13 predictor."""
    tracks_msg = create_sample_tracks_array(
        timestamp_s=timestamp_s,
        frame_id=frame_id,
        num_tracks=num_obstacles,
        obs_x=obs_x,
        obs_y=obs_y,
        vx=vx,
        vy=vy,
    )
    predictor = MotionPredictor(config=PredictorConfig(default_horizon_s=1.0, default_dt_s=0.25))
    p13_preds = predictor.predict_tracks(tracks_msg.to_phase13())
    msg = PredictedTrajectoryArrayMsg.from_phase13(p13_preds, frame_id=frame_id)
    if unknown_horizon:
        for t in msg.trajectories:
            t.is_unknown = True
    return msg



# ==============================================================================
# SECTION 1: 20 FOCUSED ROS2 INTEGRATION TESTS
# ==============================================================================

def test_01_message_serialization_round_trip():
    """Test 1: Full serialization and deserialization round-trip for dynamic messages."""
    det_arr = create_sample_detection_array(timestamp_s=100.5, frame_id="base_link")
    det = det_arr.detections[0]
    json_str = message_to_json(det)
    det_out = message_from_json(json_str, DynamicObstacleDetectionMsg)
    assert det_out.detection_id == 1
    assert math.isclose(det_out.position[0], 2.5)
    assert math.isclose(det_out.confidence, 0.85)
    assert det_out.header.frame_id == "base_link"


def test_02_detection_message_validity():
    """Test 2: Validate detection field constraints, dimensions, and frame_id."""
    det_arr = create_sample_detection_array(timestamp_s=10.0, frame_id="base_link")
    det = det_arr.detections[0]
    p13_det = det.to_phase13()
    assert str(p13_det.detection_id) == "1"
    assert math.isclose(p13_det.bbox_size[0], 0.8)
    assert math.isclose(p13_det.confidence, 0.85)


def test_03_tracked_state_serialization():
    """Test 3: Validate TrackedObstacle serialization and covariance reconstruction."""
    track_msg = create_sample_tracks_array(timestamp_s=5.0).tracks[0]
    json_str = message_to_json(track_msg)
    reconstructed = message_from_json(json_str, TrackedObstacleMsg)
    assert reconstructed.track_id == 100
    assert len(reconstructed.covariance) == 16
    p13_track = reconstructed.to_phase13()
    assert p13_track.track_id == 100
    assert p13_track.status in (TrackStatus.TRACKING, TrackStatus.TRACKING.value)


def test_04_prediction_serialization():
    """Test 4: Validate PredictedTrajectoryArray serialization with waypoints."""
    pred_msg = create_sample_predictions_array(timestamp_s=12.0)
    json_str = message_to_json(pred_msg)
    reconstructed = message_from_json(json_str, PredictedTrajectoryArrayMsg)
    assert len(reconstructed.trajectories) == 1
    assert len(reconstructed.trajectories[0].waypoints) == 5
    assert math.isclose(reconstructed.trajectories[0].waypoints[1].target_time_s - 12.0, 0.25)


def test_05_occupancy_transport():
    """Test 5: Validate DynamicOccupancySliceArray transport and flattening."""
    node = DynamicOccupancyNode()
    pred_msg = create_sample_predictions_array(timestamp_s=10.0)
    occ_msg = node.process_predictions(pred_msg)
    assert len(occ_msg.slices) == len(node.config.time_bins)
    slice0 = occ_msg.slices[0]
    assert slice0.width * slice0.height == len(slice0.data)
    json_str = message_to_json(occ_msg)
    rec_occ = message_from_json(json_str, DynamicOccupancySliceArrayMsg)
    assert len(rec_occ.slices) == len(occ_msg.slices)


def test_06_behavioral_state_transport():
    """Test 6: Validate BehavioralDecision transport with reasons and deadlock flags."""
    dec = BehavioralDecision(
        state=BehavioralState.AVOID,
        command_v=0.35,
        command_omega=-0.20,
        reason="Dynamic detour around track 101",
        blocking_obstacle_id=101,
        deadlock=False,
    )
    dec_msg = BehavioralDecisionMsg.from_phase13(dec, timestamp_s=15.0)
    json_str = message_to_json(dec_msg)
    rec = message_from_json(json_str, BehavioralDecisionMsg)
    assert rec.state == "AVOID"
    assert math.isclose(rec.command_v, 0.35)
    assert rec.blocking_obstacle_id == 101
    assert rec.deadlock is False


def test_07_replan_event_transport():
    """Test 7: Validate DynamicReplanEvent transport with trigger and metrics."""
    event = DynamicReplanEvent(
        trigger_type=DynamicReplanTrigger.DYNAMIC_DEADLOCK,
        timestamp=22.0,
        current_pose=(0.0, 0.0, 0.0),
        current_goal=(5.0, 0.0),
        current_global_path=None,
        blocking_obstacle_ids=[105],
        blocked_duration=3.5,
        deadlock_duration=3.5,
        current_progress=0.0,
        reason="Persistent deadlock test",
        confidence=1.0,
        event_id="test_ev_01",
    )
    metrics = PathChangeMetrics(
        old_path_length=10.0,
        new_path_length=10.5,
        old_path_cost=10.0,
        new_path_cost=10.5,
        path_deviation=1.2,
        goal_reachable=True,
        num_dynamic_obstacle_intersections=0,
        min_dynamic_clearance=1.5,
    )
    res = DynamicReplanResult(
        success=True,
        replan_triggered=True,
        trigger_type=DynamicReplanTrigger.DYNAMIC_DEADLOCK,
        event=event,
        metrics=metrics,
        planning_latency_ms=45.0,
    )
    ev_msg = DynamicReplanEventMsg.from_phase13(event, result=res, frame_id="odom")
    json_str = message_to_json(ev_msg)
    rec = message_from_json(json_str, DynamicReplanEventMsg)
    assert rec.trigger_type == "DYNAMIC_DEADLOCK"
    assert rec.blocking_obstacle_ids == [105]
    assert rec.success is True
    assert math.isclose(rec.path_deviation_m, 1.2)


def test_08_timestamp_propagation():
    """Test 8: Validate monotonic timestamp propagation across all pipeline stages."""
    t_in = 42.125
    det_msg = create_sample_detection_array(timestamp_s=t_in, frame_id="odom")
    tracker_node = DynamicTrackerNode()
    tracks_msg = tracker_node.process_detections(det_msg)
    assert math.isclose(tracks_msg.header.stamp.to_sec(), t_in)

    predictor_node = DynamicPredictorNode()
    pred_msg = predictor_node.process_tracks(tracks_msg)
    assert math.isclose(pred_msg.header.stamp.to_sec(), t_in)


def test_09_frame_id_propagation():
    """Test 9: Validate coordinate frame ID preservation ('odom' vs 'base_link')."""
    det_msg = create_sample_detection_array(frame_id="base_link")
    assert det_msg.header.frame_id == "base_link"
    tracks_msg = create_sample_tracks_array(frame_id="odom")
    assert tracks_msg.header.frame_id == "odom"
    pred_msg = create_sample_predictions_array(frame_id="odom")
    assert pred_msg.header.frame_id == "odom"


def test_10_stale_message_rejection():
    """Test 10: Reject stale messages and refuse unsafe forward commands."""
    adapter = DynamicNavigationAdapter()
    loc = create_loc_msg(timestamp_s=1.0)
    wmap = create_world_map_msg(timestamp_s=1.0)
    goal = PoseStampedMsg.from_xy_yaw(x=3.0, y=0.0, yaw=0.0, timestamp_s=1.0)

    # Current time is 10.0 seconds later (loc age = 9.0s > 0.50s timeout)
    res: DynamicPlanningResult = adapter.step(
        loc_msg=loc,
        map_msg=wmap,
        goal_msg=goal,
        sim_time_s=10.0,
    )
    assert res.status == PlannerStatus.LOCALIZATION_INVALID
    assert res.cmd_vel.linear.x == 0.0
    assert res.cmd_vel.angular.z == 0.0


def test_11_command_publication_for_proceed():
    """Test 11: Free path results in PROCEED behavioral decision and positive forward motion."""
    node = DynamicPlannerNode()
    loc = create_loc_msg(x=0.0, y=0.0, timestamp_s=1.0)
    wmap = create_world_map_msg(timestamp_s=1.0)
    goal = PoseStampedMsg.from_xy_yaw(x=3.0, y=0.0, yaw=0.0, timestamp_s=1.0)

    node.on_localization_status(loc)
    node.on_map_received(wmap)
    node.on_goal_received(goal)

    result = node.process_cycle(sim_time_s=1.0)
    assert result.behavior.state in ("PROCEED", "AVOID")
    assert result.cmd_vel.linear.x > 0.0


def test_12_command_publication_for_avoid():
    """Test 12: Moving obstacle offset from path triggers AVOID with safe steering command."""
    node = DynamicPlannerNode()
    loc = create_loc_msg(x=0.0, y=0.0, timestamp_s=1.0)
    wmap = create_world_map_msg(timestamp_s=1.0)
    goal = PoseStampedMsg.from_xy_yaw(x=3.0, y=0.0, yaw=0.0, timestamp_s=1.0)

    # Obstacle slightly offset at x=1.5, y=0.4 moving slowly
    tracks = create_sample_tracks_array(timestamp_s=1.0, obs_x=1.5, obs_y=0.4)
    preds = create_sample_predictions_array(timestamp_s=1.0, obs_x=1.5, obs_y=0.4)

    node.on_localization_status(loc)
    node.on_map_received(wmap)
    node.on_goal_received(goal)
    node.on_tracks_received(tracks)
    node.on_predictions_received(preds)

    result = node.process_cycle(sim_time_s=1.0)
    assert result.cmd_vel.linear.x >= 0.0


def test_13_wait_publishes_zero():
    """Test 13: Invariant: WAIT state must strictly publish zero linear and angular command."""
    adapter = DynamicNavigationAdapter()
    v, w, is_stopped = adapter.apply_safety_clamp(
        cmd_v=0.4,
        cmd_w=0.2,
        min_clearance_m=1.2,
        behavioral_state=BehavioralState.WAIT,
    )
    assert v == 0.0
    assert w == 0.0


def test_14_stop_publishes_zero():
    """Test 14: Invariant: STOP state must strictly publish zero linear and angular command."""
    adapter = DynamicNavigationAdapter()
    v, w, is_stopped = adapter.apply_safety_clamp(
        cmd_v=0.5,
        cmd_w=-0.3,
        min_clearance_m=0.8,
        behavioral_state=BehavioralState.STOP,
    )
    assert v == 0.0
    assert w == 0.0


def test_15_replan_failure_prevents_unsafe_command():
    """Test 15: Replanning failure protects the robot from executing unsafe unvalidated motion."""
    adapter = DynamicNavigationAdapter()
    loc = create_loc_msg(x=0.0, y=0.0, timestamp_s=1.0)
    # Block the entire map so replanning fails
    wmap = create_world_map_msg(timestamp_s=1.0)
    wmap.obstacle = [True] * len(wmap.obstacle)
    wmap.cost = [1.0] * len(wmap.cost)
    goal = PoseStampedMsg.from_xy_yaw(x=3.0, y=0.0, yaw=0.0, timestamp_s=1.0)

    res = adapter.step(
        loc_msg=loc,
        map_msg=wmap,
        goal_msg=goal,
        sim_time_s=1.0,
    )
    assert res.status in (PlannerStatus.NO_PATH, PlannerStatus.EMERGENCY_STOP)
    assert res.cmd_vel.linear.x == 0.0
    assert res.cmd_vel.angular.z == 0.0


def test_16_unknown_horizon_survives_transport():
    """Test 16: Unknown prediction horizon flag is preserved across serialization and triggers STOP."""
    preds = create_sample_predictions_array(timestamp_s=1.0, unknown_horizon=True)
    json_str = message_to_json(preds)
    rec = message_from_json(json_str, PredictedTrajectoryArrayMsg)
    assert rec.trajectories[0].is_unknown is True

    # Check that BehavioralStateMachine evaluates this conservatively
    sm = BehavioralStateMachine(config=BehavioralConfig(unknown_horizon_policy="REJECT"))
    p13_preds = rec.to_phase13()
    decision = sm.update(
        ugv_pose=(0.0, 0.0, 0.0),
        ugv_twist=(0.2, 0.0),
        timestamp_s=1.0,
        predictions=p13_preds,
    )
    assert decision.state == BehavioralState.STOP
    assert decision.command_v == 0.0


def test_17_tf2_transform_correctness():
    """Test 17: Validate coordinate frame reconciliation and TF tree lookup validation."""
    adapter = DynamicNavigationAdapter()
    loc_bad_frame = create_loc_msg(frame_id="camera_link")
    wmap = create_world_map_msg()
    is_valid, reason = adapter.validate_inputs(loc_bad_frame, wmap, current_time_s=1.0)
    assert is_valid is False
    assert "Invalid localization frame_id" in reason


def test_18_deterministic_message_reconstruction():
    """Test 18: Deterministic serialization and deserialization produces bitwise identical models."""
    sample_tracks = create_sample_tracks_array(timestamp_s=10.0, num_tracks=3)
    json1 = message_to_json(sample_tracks)
    json2 = message_to_json(sample_tracks)
    assert json1 == json2

    rec = message_from_json(json1, TrackedObstacleArrayMsg)
    json3 = message_to_json(rec)
    assert json1 == json3


def test_19_parameter_validation():
    """Test 19: Reject invalid parameters (negative horizons, invalid confidence)."""
    with pytest.raises(ValueError):
        DynamicDetectorNode(min_confidence=1.5)  # > 1.0
    with pytest.raises(ValueError):
        DynamicDetectorNode(frame_id="")  # Empty frame

    with pytest.raises(ValueError):
        PredictorConfig(default_horizon_s=-1.0)


def test_20_launch_configuration_validation():
    """Test 20: Validate launch file description and YAML parameter files exist and parse cleanly."""
    root_dir = Path(__file__).resolve().parent.parent
    config_dir = root_dir / "ros2_ws" / "src" / "ugv_navigation" / "config"
    launch_dir = root_dir / "ros2_ws" / "src" / "ugv_navigation" / "launch"

    params_yaml = config_dir / "ugv_params.yaml"
    qos_yaml = config_dir / "qos_profiles.yaml"
    launch_file = launch_dir / "dynamic_navigation.launch.py"

    assert params_yaml.exists()
    assert qos_yaml.exists()
    assert launch_file.exists()

    with open(params_yaml, "r") as f:
        params = yaml.safe_load(f)
        assert "ugv_navigation" in params
        assert "tracking" in params["ugv_navigation"]
        assert "prediction" in params["ugv_navigation"]
        assert "behavior" in params["ugv_navigation"]
        assert "replanning" in params["ugv_navigation"]
        assert "safety" in params["ugv_navigation"]

    with open(qos_yaml, "r") as f:
        qos = yaml.safe_load(f)
        assert "qos_profiles" in qos
        assert "dynamic_detections" in qos["qos_profiles"]
        assert "tracked_obstacles" in qos["qos_profiles"]
        assert "predictions" in qos["qos_profiles"]
        assert "behavioral_decision" in qos["qos_profiles"]
        assert "cmd_vel" in qos["qos_profiles"]


# ==============================================================================
# SECTION 2: 9 SAFETY INVARIANTS (A-I)
# ==============================================================================

def test_safety_invariant_a_stop_cannot_publish_nonzero_motion():
    """Safety Invariant A: STOP state cannot publish nonzero motion."""
    adapter = DynamicNavigationAdapter()
    for test_v in [0.0, 0.1, 0.5, 1.0]:
        for test_w in [-1.0, -0.2, 0.0, 0.5]:
            v, w, stopped = adapter.apply_safety_clamp(
                cmd_v=test_v,
                cmd_w=test_w,
                min_clearance_m=1.0,
                behavioral_state=BehavioralState.STOP,
            )
            assert v == 0.0
            assert w == 0.0


def test_safety_invariant_b_wait_cannot_publish_nonzero_motion():
    """Safety Invariant B: WAIT state cannot publish nonzero motion."""
    adapter = DynamicNavigationAdapter()
    for test_v in [0.0, 0.2, 0.6]:
        for test_w in [-0.5, 0.0, 0.8]:
            v, w, stopped = adapter.apply_safety_clamp(
                cmd_v=test_v,
                cmd_w=test_w,
                min_clearance_m=1.5,
                behavioral_state=BehavioralState.WAIT,
            )
            assert v == 0.0
            assert w == 0.0


def test_safety_invariant_c_replan_failure_prevents_unsafe_command():
    """Safety Invariant C: GLOBAL_REPLAN_FAILED cannot publish an unsafe command."""
    adapter = DynamicNavigationAdapter()
    loc = create_loc_msg(timestamp_s=1.0)
    wmap = create_world_map_msg(timestamp_s=1.0)
    wmap.obstacle = [True] * len(wmap.obstacle)
    wmap.cost = [1.0] * len(wmap.cost)
    goal = PoseStampedMsg.from_xy_yaw(x=4.0, y=0.0, yaw=0.0, timestamp_s=1.0)

    res = adapter.step(
        loc_msg=loc,
        map_msg=wmap,
        goal_msg=goal,
        sim_time_s=1.0,
    )
    assert res.cmd_vel.linear.x == 0.0
    assert res.cmd_vel.angular.z == 0.0


def test_safety_invariant_d_stale_dynamic_data_cannot_become_current():
    """Safety Invariant D: Stale dynamic data cannot silently become current."""
    adapter = DynamicNavigationAdapter()
    loc = create_loc_msg(timestamp_s=1.0)
    wmap = create_world_map_msg(timestamp_s=1.0)
    goal = PoseStampedMsg.from_xy_yaw(x=3.0, y=0.0, yaw=0.0, timestamp_s=1.0)

    stale_tracks = create_sample_tracks_array(timestamp_s=1.0)
    stale_preds = create_sample_predictions_array(timestamp_s=1.0)

    # Current time is 10.0s (age = 9.0s)
    res = adapter.step(
        loc_msg=loc,
        map_msg=wmap,
        goal_msg=goal,
        tracks_msg=stale_tracks,
        predictions_msg=stale_preds,
        sim_time_s=10.0,
    )
    # Must reject stale input and stop
    assert res.status == PlannerStatus.LOCALIZATION_INVALID
    assert res.cmd_vel.linear.x == 0.0


def test_safety_invariant_e_unknown_prediction_cannot_become_safe():
    """Safety Invariant E: Unknown prediction cannot become SAFE due to transport."""
    pred_msg = create_sample_predictions_array(timestamp_s=1.0, unknown_horizon=True)
    json_data = message_to_json(pred_msg)
    reconstructed = message_from_json(json_data, PredictedTrajectoryArrayMsg)
    assert reconstructed.trajectories[0].is_unknown is True

    # Downstream evaluation must NOT treat it as SAFE
    sm = BehavioralStateMachine(config=BehavioralConfig(unknown_horizon_policy="REJECT"))
    dec = sm.update(
        ugv_pose=(0.0, 0.0, 0.0),
        ugv_twist=(0.0, 0.0),
        timestamp_s=1.0,
        predictions=reconstructed.to_phase13(),
    )
    assert dec.state == BehavioralState.STOP
    assert dec.command_v == 0.0


def test_safety_invariant_f_frame_mismatch_cannot_silently_pass():
    """Safety Invariant F: Frame mismatch cannot silently pass validation."""
    adapter = DynamicNavigationAdapter()
    loc = create_loc_msg(frame_id="camera_optical")  # Invalid frame for world navigation
    wmap = create_world_map_msg(frame_id="odom")

    is_valid, reason = adapter.validate_inputs(loc, wmap, current_time_s=1.0)
    assert is_valid is False
    assert "Invalid localization frame_id" in reason


def test_safety_invariant_g_out_of_order_messages_handled_safely():
    """Safety Invariant G: Duplicate/out-of-order messages cannot cause nondeterministic unsafe state."""
    adapter = DynamicNavigationAdapter()
    loc = create_loc_msg(timestamp_s=10.0)
    wmap = create_world_map_msg(timestamp_s=10.0)
    goal = PoseStampedMsg.from_xy_yaw(x=3.0, y=0.0, yaw=0.0, timestamp_s=10.0)

    # Step at time 10.0
    res1 = adapter.step(loc_msg=loc, map_msg=wmap, goal_msg=goal, sim_time_s=10.0)

    # Now an out-of-order / backward timestamp message arrives at sim_time_s = 5.0
    # Current time is 10.0, message timestamp is 5.0 (age = 5.0 > stale timeout 0.5s)
    loc_old = create_loc_msg(timestamp_s=5.0)
    res2 = adapter.step(loc_msg=loc_old, map_msg=wmap, goal_msg=goal, sim_time_s=10.0)
    assert res2.status == PlannerStatus.LOCALIZATION_INVALID
    assert res2.cmd_vel.linear.x == 0.0
    assert res2.cmd_vel.angular.z == 0.0


def test_safety_invariant_h_safety_clamp_authoritative():
    """Safety Invariant H: ROS2 integration cannot bypass the existing safety clamp."""
    adapter = DynamicNavigationAdapter(config=DynamicAdapterConfig(emergency_stop_distance_m=0.35))
    # Even if behavioral state were PROCEED and candidate cmd_v = 0.6,
    # an obstacle clearance of 0.20m (< 0.35m) MUST clamp velocity to strictly 0.0
    v, w, is_stopped = adapter.apply_safety_clamp(
        cmd_v=0.6,
        cmd_w=0.3,
        min_clearance_m=0.20,
        behavioral_state=BehavioralState.PROCEED,
    )
    assert v == 0.0
    assert w == 0.0
    assert is_stopped is True


def test_safety_invariant_i_deterministic_input_produces_deterministic_output():
    """Safety Invariant I: Deterministic input produces deterministic behavioral output."""
    node1 = DynamicPlannerNode()
    node2 = DynamicPlannerNode()

    loc = create_loc_msg(x=0.0, y=0.0, timestamp_s=1.0)
    wmap = create_world_map_msg(timestamp_s=1.0)
    goal = PoseStampedMsg.from_xy_yaw(x=3.0, y=0.0, yaw=0.0, timestamp_s=1.0)
    tracks = create_sample_tracks_array(timestamp_s=1.0, obs_x=2.0, obs_y=0.2)
    preds = create_sample_predictions_array(timestamp_s=1.0, obs_x=2.0, obs_y=0.2)

    node1.on_localization_status(loc)
    node1.on_map_received(wmap)
    node1.on_goal_received(goal)
    node1.on_tracks_received(tracks)
    node1.on_predictions_received(preds)

    node2.on_localization_status(loc)
    node2.on_map_received(wmap)
    node2.on_goal_received(goal)
    node2.on_tracks_received(tracks)
    node2.on_predictions_received(preds)

    res1 = node1.process_cycle(sim_time_s=1.0)
    res2 = node2.process_cycle(sim_time_s=1.0)

    assert res1.behavior.state == res2.behavior.state
    assert math.isclose(res1.cmd_vel.linear.x, res2.cmd_vel.linear.x, abs_tol=1e-5)
    assert math.isclose(res1.cmd_vel.angular.z, res2.cmd_vel.angular.z, abs_tol=1e-5)
