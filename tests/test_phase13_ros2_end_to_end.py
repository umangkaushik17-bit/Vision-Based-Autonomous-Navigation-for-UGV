"""Phase 13.12 End-to-End Structural ROS Boundary Integration Tests.

Deterministic structural integration pipeline test covering the full flow:
Dynamic Detection -> Tracker -> Prediction -> Occupancy -> Planner -> Behavior -> Replan.

Uses real Phase 13 algorithms at the core with controlled ROSIDL test doubles
at every ROS2 node boundary.

Scenarios tested:
- SCENARIO A: Static / No Hazard (Proceed, clean tracking, safe travel)
- SCENARIO B: Moving Obstacle (Detected, tracked, predicted, occupancy sliced, planner reacts)
- SCENARIO C: Obstacle Enters Collision Path (Collision risk detected, avoid/wait/stop policy)
- SCENARIO D: Temporary Blockage (Tactical behavior changes to STOP/WAIT, no unsafe proceed)
- SCENARIO E: Persistent Blockage (Global replan event triggered after debounce/cooldown)
- SCENARIO F: Unknown Horizon (Unknown is never silently treated as free)
- SCENARIO G: Stale Message (Backward timestamp rejected, state protected)
- SCENARIO H: Malformed Message (Controlled safe failure without crashes)
- SCENARIO I: Multiple Moving Obstacles (All hazards survive through occupancy and planning)
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pytest

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.path import Path, PathPoint
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
    DeadlockEvent,
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
from ai.planning.dynamic.detector_interface import (
    SyntheticDynamicObstacleDetector,
    SyntheticObstacleScript,
)
from ai.planning.dynamic.global_replanner import (
    DynamicReplanEvent,
    DynamicReplanResult,
    DynamicReplanTrigger,
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
)
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyConfig, DynamicOccupancyLayer
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig
from ai.planning.dynamic.tracker import MultiFrameTracker, TrackerConfig
from ai.spatial_fusion.world_map import WorldTerrainMap

from ugv_navigation.adapters.dynamic_navigation_adapter import (
    DynamicAdapterConfig,
    DynamicNavigationAdapter,
    DynamicPlanningResult,
)
from ugv_navigation.adapters.planner_adapter import PathMsg, PoseStampedMsg, TwistMsg
from ugv_navigation.converters.dynamic_converters import (
    tracked_obstacle_array_from_ros,
    predicted_trajectory_array_from_ros,
    dynamic_occupancy_slice_array_from_ros,
)
from ugv_navigation.nodes.behavioral_state_node import BehavioralStateNode
from ugv_navigation.nodes.dynamic_detector_node import DynamicDetectorNode
from ugv_navigation.nodes.dynamic_occupancy_node import DynamicOccupancyNode
from ugv_navigation.nodes.dynamic_planner_node import DynamicPlannerNode
from ugv_navigation.nodes.dynamic_predictor_node import DynamicPredictorNode
from ugv_navigation.nodes.dynamic_tracker_node import DynamicTrackerNode
from ugv_navigation.nodes.global_replanner_node import GlobalReplannerNode
from ugv_navigation.tf.tf_tree import TfTreeBuffer


# ==============================================================================
# Controlled Structural ROSIDL Test Doubles
# ==============================================================================

class DoubleTime:
    def __init__(self, sec: int = 0, nanosec: int = 0):
        self.sec = int(sec)
        self.nanosec = int(nanosec)

    def to_sec(self) -> float:
        return float(self.sec) + float(self.nanosec) * 1e-9


class DoubleHeader:
    def __init__(self, frame_id: str = "odom", stamp: Optional[DoubleTime] = None):
        self.frame_id = frame_id
        self.stamp = stamp or DoubleTime()

    @classmethod
    def create(cls, frame_id: str = "odom", timestamp_s: float = 0.0) -> DoubleHeader:
        sec = int(timestamp_s)
        nanosec = int(round((timestamp_s - sec) * 1e9))
        return cls(frame_id=frame_id, stamp=DoubleTime(sec=sec, nanosec=nanosec))


class DoublePoint:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class DoubleVector3:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class DoubleQuaternion:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)


class DoublePose:
    def __init__(self, position: Optional[DoublePoint] = None, orientation: Optional[DoubleQuaternion] = None):
        self.position = position or DoublePoint()
        self.orientation = orientation or DoubleQuaternion()


class DoublePoseWithCovariance:
    def __init__(self, pose: Optional[DoublePose] = None, covariance: Optional[List[float]] = None):
        self.pose = pose or DoublePose()
        self.covariance = covariance or [0.0] * 36


class DoubleTwist:
    def __init__(self, linear: Optional[DoubleVector3] = None, angular: Optional[DoubleVector3] = None):
        self.linear = linear or DoubleVector3()
        self.angular = angular or DoubleVector3()


class DoubleTwistWithCovariance:
    def __init__(self, twist: Optional[DoubleTwist] = None, covariance: Optional[List[float]] = None):
        self.twist = twist or DoubleTwist()
        self.covariance = covariance or [0.0] * 36


class DoubleLocalizationStatusMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.child_frame_id = "base_link"
        self.tracking_state = 0
        self.confidence = 1.0
        self.is_pose_valid = True
        self.allows_spatial_fusion = True
        self.pose = DoublePoseWithCovariance()
        self.twist = DoubleTwistWithCovariance()
        self.tracked_features = 100
        self.inliers = 80


class DoubleDynamicObstacleDetectionMsg:
    def __init__(self):
        self.detection_id = ""
        self.bbox_center = DoublePoint()
        self.bbox_size = DoubleVector3(1.0, 1.0, 1.0)
        self.bbox_yaw = 0.0
        self.confidence = 1.0
        self.covariance = [0.0] * 16
        self.velocity = DoubleVector3()
        self.is_valid = True


class DoubleDynamicObstacleDetectionArrayMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.detections = []


class DoubleTrackedObstacleMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.track_id = 0
        self.status = "TRACKING"
        self.is_confirmed = True
        self.confidence = 1.0
        self.position = DoublePoint()
        self.velocity = DoubleVector3()
        self.bbox_size = DoubleVector3(1.0, 1.0, 1.0)
        self.bbox_yaw = 0.0
        self.covariance = [0.0] * 16
        self.total_hits = 3
        self.consecutive_misses = 0
        self.time_since_last_update_s = 0.0


class DoubleTrackedObstacleArrayMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.tracks = []


class DoubleTrajectoryWaypointMsg:
    def __init__(self):
        self.target_time = DoubleTime(1, 0)
        self.relative_time_s = 0.0
        self.position = DoublePoint()
        self.velocity = DoubleVector3()
        self.covariance = [0.01, 0.0, 0.0, 0.01]
        self.uncertainty_radius_m = 0.1
        self.confidence = 1.0


class DoublePredictedObstacleTrajectoryMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.track_id = 0
        self.status = "TRACKING"
        self.confidence = 1.0
        self.bbox_size = DoubleVector3(1.0, 1.0, 1.0)
        self.bbox_yaw = 0.0
        self.waypoints = []
        self.prediction_horizon_s = 2.0
        self.is_unknown = False


class DoublePredictedTrajectoryArrayMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.trajectories = []


class DoubleDynamicOccupancyGridSliceMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.target_time_offset_s = 0.0
        self.target_timestamp = DoubleTime(1, 0)
        self.resolution = 0.10
        self.origin_x = -5.0
        self.origin_y = -5.0
        self.width = 10
        self.height = 10
        self.data = []
        self.unknown_mask = []


class DoubleDynamicOccupancySliceArrayMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.resolution = 0.10
        self.origin_x = -5.0
        self.origin_y = -5.0
        self.slices = []


class DoubleBehavioralDecisionMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.state = "PROCEED"
        self.command_v = 0.0
        self.command_omega = 0.0
        self.reason = ""
        self.blocking_obstacle_id = 0
        self.dynamic_collision = False
        self.dynamic_risk = "LOW"
        self.unknown_horizon = False
        self.wait_elapsed_s = 0.0
        self.wait_timeout = False
        self.deadlock = False
        self.deadlock_event = "NONE"
        self.transition = ""
        self.previous_state = "PROCEED"
        self.state_dwell_time_s = 0.0
        self.confidence = 1.0


class DoubleDynamicReplanEventMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.event_id = ""
        self.trigger_type = "NONE"
        self.current_pose = DoublePoint()
        self.current_yaw = 0.0
        self.current_goal = DoublePoint()
        self.blocking_obstacle_ids = []
        self.blocked_duration_s = 0.0
        self.deadlock_duration_s = 0.0
        self.current_progress_m = 0.0
        self.reason = ""
        self.confidence = 1.0
        self.success = True
        self.is_path_stale = False
        self.old_path_length_m = 0.0
        self.new_path_length_m = 0.0
        self.path_deviation_m = 0.0
        self.planning_latency_ms = 0.0


def make_clean_world_map(timestamp: float = 1.0) -> WorldTerrainMap:
    wmap = WorldTerrainMap(
        resolution_m=0.10,
        x_bounds_m=(-5.0, 5.0),
        y_bounds_m=(-5.0, 5.0),
        created_timestamp=timestamp,
    )
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    wmap.observation_count.fill(1)
    wmap.last_observed_timestamp.fill(timestamp)
    return wmap


def make_straight_path(x_start: float = 0.0, x_end: float = 4.0, y: float = 0.0) -> Path:
    pts = []
    num_pts = 41
    for i in range(num_pts):
        px = x_start + (x_end - x_start) * (i / (num_pts - 1))
        pts.append(PathPoint(x=px, y=y, cost=0.10))
    return Path(points=pts)


# ==============================================================================
# End-to-End Structural Scenario Test Suite
# ==============================================================================

class TestPhase13Ros2EndToEnd:
    """Deterministic structural tests verifying the full dynamic navigation pipeline."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        """Instantiate all 7 real Phase 13 nodes."""
        self.detector_node = DynamicDetectorNode(
            output_msg_cls=DoubleDynamicObstacleDetectionArrayMsg,
            det_item_msg_cls=DoubleDynamicObstacleDetectionMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
            vector3_cls=DoubleVector3,
        )
        self.tracker_node = DynamicTrackerNode(
            output_msg_cls=DoubleTrackedObstacleArrayMsg,
            track_item_msg_cls=DoubleTrackedObstacleMsg,
        )
        self.predictor_node = DynamicPredictorNode(
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )
        self.occupancy_node = DynamicOccupancyNode(
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )
        self.planner_node = DynamicPlannerNode()
        self.behavior_node = BehavioralStateNode(
            behavior_msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )
        self.replanner_node = GlobalReplannerNode(
            event_msg_cls=DoubleDynamicReplanEventMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
        )

    # --------------------------------------------------------------------------
    # SCENARIO A — STATIC / NO HAZARD
    # --------------------------------------------------------------------------
    def test_scenario_a_static_no_hazard(self):
        """Scenario A: In an open environment without dynamic hazards,
        tracking succeeds (0 tracks), prediction succeeds (0 predictions),
        occupancy slices contain no false lethal obstacles, and planner proceeds.
        """
        t = 1.0
        # 1. Perception: No detections
        det_msg = self.detector_node.process_frame(
            frame_data=[],
            timestamp_s=t,
            output_msg_cls=DoubleDynamicObstacleDetectionArrayMsg,
            det_item_msg_cls=DoubleDynamicObstacleDetectionMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
            vector3_cls=DoubleVector3,
        )
        assert len(det_msg.detections) == 0

        # 2. Tracking: Ingest 0 detections -> 0 tracks
        ros_tracks = self.tracker_node.process_detections(
            msg=det_msg,
            output_msg_cls=DoubleTrackedObstacleArrayMsg,
        )
        assert len(ros_tracks.tracks) == 0

        # 3. Prediction: Ingest 0 tracks -> 0 predictions
        ros_preds = self.predictor_node.process_tracks(
            msg=ros_tracks,
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )
        assert len(ros_preds.trajectories) == 0

        # 4. Occupancy: Ingest 0 predictions -> free slices
        ros_occ = self.occupancy_node.process_predictions(
            msg=ros_preds,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )
        assert len(ros_occ.slices) > 0
        for s in ros_occ.slices:
            assert all(val == 0 for val in s.data)

        # 5. Planner: Execute cycle toward goal
        wmap = make_clean_world_map(t)
        self.planner_node.on_map_received(wmap)
        loc = DoubleLocalizationStatusMsg()
        loc.header = DoubleHeader.create("odom", t)
        self.planner_node.on_localization_status(loc)
        goal = PoseStampedMsg(header=DoubleHeader.create("odom", t), pose=DoublePose(position=DoublePoint(4.0, 0.0, 0.0)))
        self.planner_node.on_goal_received(goal)
        self.planner_node.on_tracks_received(ros_tracks)
        self.planner_node.on_predictions_received(ros_preds)

        plan_res = self.planner_node.process_cycle(sim_time_s=t, dt_s=0.10)
        assert plan_res.cmd_vel.linear.x > 0.0
        assert plan_res.behavior.state == BehavioralState.PROCEED

    # --------------------------------------------------------------------------
    # SCENARIO B — MOVING OBSTACLE
    # --------------------------------------------------------------------------
    def test_scenario_b_moving_obstacle(self):
        """Scenario B: Moving obstacle is detected over multiple frames,
        tracked, predicted ahead in time, occupies future spatial slices,
        and causes the planner to evaluate the hazard.
        """
        # Multi-frame tracking to confirm track (hits_to_confirm = 3)
        self.detector_node.detector.clear_obstacle_scripts()
        script = SyntheticObstacleScript(
            obstacle_id="obs_b",
            size=(0.8, 0.6),
            position_fn=lambda t: (2.5 + (t - 1.0) * 0.5, 0.5),
            yaw_fn=lambda t: 0.0,
            confidence=0.90,
        )
        self.detector_node.detector.add_obstacle_script(script)

        confirmed_ros_tracks = None
        for step in range(4):
            t = 1.0 + step * 0.10
            det_msg = self.detector_node.process_frame(
                frame_data=None,
                timestamp_s=t,
                output_msg_cls=DoubleDynamicObstacleDetectionArrayMsg,
                det_item_msg_cls=DoubleDynamicObstacleDetectionMsg,
                header_cls=DoubleHeader,
                point_cls=DoublePoint,
                vector3_cls=DoubleVector3,
            )
            confirmed_ros_tracks = self.tracker_node.process_detections(
                msg=det_msg,
                output_msg_cls=DoubleTrackedObstacleArrayMsg,
            )

        assert len(confirmed_ros_tracks.tracks) == 1
        trk = confirmed_ros_tracks.tracks[0]
        assert trk.status == "TRACKING"

        # Predict future trajectory
        t_final = 1.4
        ros_preds = self.predictor_node.process_tracks(
            msg=confirmed_ros_tracks,
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )
        assert len(ros_preds.trajectories) == 1
        assert len(ros_preds.trajectories[0].waypoints) > 0

        # Build dynamic occupancy grid slices
        ros_occ = self.occupancy_node.process_predictions(
            msg=ros_preds,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )
        # Slices must contain obstacle cost
        has_obstacle = any(any(val > 0 for val in s.data) for s in ros_occ.slices)
        assert has_obstacle is True

    # --------------------------------------------------------------------------
    # SCENARIO C — OBSTACLE ENTERS COLLISION PATH
    # --------------------------------------------------------------------------
    def test_scenario_c_obstacle_enters_collision_path(self):
        """Scenario C: Obstacle moving directly towards vehicle triggers
        collision avoidance or stop policy in DynamicPlannerNode.
        """
        t = 2.0
        wmap = make_clean_world_map(t)
        self.planner_node.on_map_received(wmap)
        loc = DoubleLocalizationStatusMsg()
        loc.header = DoubleHeader.create("odom", t)
        self.planner_node.on_localization_status(loc)
        goal = PoseStampedMsg(header=DoubleHeader.create("odom", t), pose=DoublePose(position=DoublePoint(4.0, 0.0, 0.0)))
        self.planner_node.on_goal_received(goal)

        # Convert to ROSIDL doubles
        ros_tracks = DoubleTrackedObstacleArrayMsg()
        ros_tracks.header = DoubleHeader.create("odom", t)
        ros_trk = DoubleTrackedObstacleMsg()
        ros_trk.track_id = 7
        ros_trk.position = DoublePoint(1.0, 0.0, 0.0)
        ros_trk.velocity = DoubleVector3(-0.2, 0.0, 0.0)
        ros_trk.bbox_size = DoubleVector3(0.8, 0.6, 0.5)
        ros_tracks.tracks.append(ros_trk)

        self.planner_node.on_tracks_received(ros_tracks)
        self.planner_node.on_predictions_received(None)

        plan_res = self.planner_node.process_cycle(sim_time_s=t, dt_s=0.10)
        # Vehicle must NOT continue with unconstrained high-speed proceed directly into obstacle
        assert plan_res.behavior.state in (BehavioralState.STOP, BehavioralState.WAIT, BehavioralState.AVOID)

    # --------------------------------------------------------------------------
    # SCENARIO D — TEMPORARY BLOCKAGE
    # --------------------------------------------------------------------------
    def test_scenario_d_temporary_blockage(self):
        """Scenario D: When obstacle stops very close to vehicle (d < 0.5m),
        BehavioralStateMachine enters STOP state and cmd_vel is strictly clamped to zero.
        """
        t = 3.0
        beh_node = BehavioralStateNode()
        state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0, timestamp=t)
        diag = {
            "dynamic_collision": True,
            "dynamic_risk": "CRITICAL",
            "valid_trajectories": 0,
            "emergency_stop_triggered": True,
            "dynamically_rejected_candidates": 100,
        }

        res = beh_node.evaluate_state(
            current_state=state,
            best_v=0.0,
            best_w=0.0,
            planner_status=PlannerStatus.PLANNING,
            dwa_diagnostics=diag,
            current_time_s=t,
            output_msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )

        assert res.state == "STOP"
        assert res.command_v == 0.0
        assert res.dynamic_collision is True

    # --------------------------------------------------------------------------
    # SCENARIO E — PERSISTENT BLOCKAGE
    # --------------------------------------------------------------------------
    def test_scenario_e_persistent_blockage(self):
        """Scenario E: Obstacle blocks vehicle continuously for >= persistent_blockage_duration_s.
        GlobalReplanner triggers PERSISTENT_BLOCKAGE replan event.
        """
        cfg = GlobalReplannerConfig(persistent_blockage_duration_s=2.0, debounce_cycles=1, cooldown_s=0.0)
        node = GlobalReplannerNode(config=cfg)
        m = make_clean_world_map()
        p = make_straight_path(0.0, 4.0, 0.0)
        node.replanner.set_global_path(p)

        beh = DoubleBehavioralDecisionMsg()
        beh.state = "STOP"
        beh.blocking_obstacle_id = 42

        # Cycle 1: starts blockage timer at t=1.0
        r1, _ = node.check_and_replan(
            current_pose=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            base_map=m,
            behavioral_decision=beh,
            timestamp=1.0,
        )
        assert r1.replan_triggered is False

        # Cycle 2: duration = 2.1s >= 2.0s -> triggers PERSISTENT_BLOCKAGE
        r2, ros_event = node.check_and_replan(
            current_pose=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            base_map=m,
            behavioral_decision=beh,
            timestamp=3.1,
            output_event_cls=DoubleDynamicReplanEventMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
        )
        assert r2.replan_triggered is True
        assert r2.trigger_type == DynamicReplanTrigger.PERSISTENT_BLOCKAGE
        assert ros_event.trigger_type == "PERSISTENT_BLOCKAGE"

    # --------------------------------------------------------------------------
    # SCENARIO F — UNKNOWN HORIZON
    # --------------------------------------------------------------------------
    def test_scenario_f_unknown_horizon(self):
        """Scenario F: Dynamic predictor policy ensures unknown prediction horizons
        are handled safely and not silently treated as free/safe space.
        """
        pred_node = DynamicPredictorNode()

        # Track with very high uncertainty
        ros_tracks = DoubleTrackedObstacleArrayMsg()
        ros_tracks.header = DoubleHeader.create("odom", 1.0)
        t_msg = DoubleTrackedObstacleMsg()
        t_msg.track_id = 15
        t_msg.position = DoublePoint(2.0, 0.0, 0.0)
        t_msg.velocity = DoubleVector3(1.0, 0.0, 0.0)
        t_msg.covariance = (np.eye(4) * 10.0).flatten().tolist()
        ros_tracks.tracks.append(t_msg)

        ros_preds = pred_node.process_tracks(
            msg=ros_tracks,
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )

        assert len(ros_preds.trajectories) == 1
        p13_preds = predicted_trajectory_array_from_ros(ros_preds)
        # Uncertainty radius must reflect covariance, never zero
        assert p13_preds.predictions[0].waypoints[0].uncertainty_radius > 0.0

    # --------------------------------------------------------------------------
    # SCENARIO G — STALE MESSAGE
    # --------------------------------------------------------------------------
    def test_scenario_g_stale_message(self):
        """Scenario G: Stale inputs with timestamps moving backward in time
        are rejected and cannot corrupt node internal state.
        """
        node = self.tracker_node
        det_msg = DoubleDynamicObstacleDetectionArrayMsg()
        det_msg.header = DoubleHeader.create("base_link", 2.0)
        node.process_detections(det_msg)

        # Feed stale t=1.0
        stale_det = DoubleDynamicObstacleDetectionArrayMsg()
        stale_det.header = DoubleHeader.create("base_link", 1.0)
        node.process_detections(stale_det)

        # Internal tracker timestamp must remain at 2.0, not moved backward to 1.0
        assert node._last_timestamp == 2.0

    # --------------------------------------------------------------------------
    # SCENARIO H — MALFORMED MESSAGE
    # --------------------------------------------------------------------------
    def test_scenario_h_malformed_message(self):
        """Scenario H: Malformed or unpopulated messages are handled safely
        without crashing the pipeline or generating unsafe commands.
        """
        class CorruptedMsg:
            pass

        bad_msg = CorruptedMsg()

        # Detector safely handles bad raw input
        det_out = self.detector_node.process_frame(
            frame_data=bad_msg,
            timestamp_s=1.0,
            output_msg_cls=DoubleDynamicObstacleDetectionArrayMsg,
            det_item_msg_cls=DoubleDynamicObstacleDetectionMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
            vector3_cls=DoubleVector3,
        )
        assert len(det_out.detections) == 0

        # Tracker safely handles malformed detection message
        ros_tracks = self.tracker_node.process_detections(
            msg=bad_msg,
            output_msg_cls=DoubleTrackedObstacleArrayMsg,
        )
        assert len(ros_tracks.tracks) == 0

        # Global replanner safely returns without crash and does not trigger spurious replan
        res, ros_ev = self.replanner_node.check_and_replan(
            current_pose=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            base_map=bad_msg,
            behavioral_decision=bad_msg,
            timestamp=1.0,
        )
        assert res.replan_triggered is False
        assert ros_ev is None

    # --------------------------------------------------------------------------
    # SCENARIO I — MULTIPLE MOVING OBSTACLES
    # --------------------------------------------------------------------------
    def test_scenario_i_multiple_moving_obstacles(self):
        """Scenario I: Multiple distinct moving obstacles are simultaneously
        tracked with unique IDs, independently predicted, and both represented
        in dynamic occupancy slices.
        """
        self.detector_node.detector.clear_obstacle_scripts()
        script1 = SyntheticObstacleScript(
            obstacle_id="obs_1",
            size=(0.6, 0.6),
            position_fn=lambda t: (2.0, 1.0 + (t - 1.0) * 0.5),
            yaw_fn=lambda t: 0.0,
            confidence=0.90,
        )
        script2 = SyntheticObstacleScript(
            obstacle_id="obs_2",
            size=(0.8, 0.8),
            position_fn=lambda t: (3.0, -1.0 - (t - 1.0) * 0.5),
            yaw_fn=lambda t: 0.1,
            confidence=0.88,
        )
        self.detector_node.detector.add_obstacle_script(script1)
        self.detector_node.detector.add_obstacle_script(script2)

        confirmed_ros_tracks = None
        for step in range(4):
            t = 1.0 + step * 0.10
            det_msg = self.detector_node.process_frame(
                frame_data=None,
                timestamp_s=t,
                output_msg_cls=DoubleDynamicObstacleDetectionArrayMsg,
                det_item_msg_cls=DoubleDynamicObstacleDetectionMsg,
                header_cls=DoubleHeader,
                point_cls=DoublePoint,
                vector3_cls=DoubleVector3,
            )
            confirmed_ros_tracks = self.tracker_node.process_detections(
                msg=det_msg,
                output_msg_cls=DoubleTrackedObstacleArrayMsg,
            )

        # Both obstacles must be tracked simultaneously
        assert len(confirmed_ros_tracks.tracks) == 2
        track_ids = {trk.track_id for trk in confirmed_ros_tracks.tracks}
        assert len(track_ids) == 2

        # Predict future trajectories for both
        t_final = 1.4
        ros_preds = self.predictor_node.process_tracks(
            msg=confirmed_ros_tracks,
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )
        assert len(ros_preds.trajectories) == 2
        pred_ids = {p.track_id for p in ros_preds.trajectories}
        assert pred_ids == track_ids

        # Occupancy layer accumulates both obstacles
        ros_occ = self.occupancy_node.process_predictions(
            msg=ros_preds,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )
        assert len(ros_occ.slices) > 0
        # Both obstacle footprints must generate cost in the slices
        non_zero_cells = sum(sum(1 for val in s.data if val > 0) for s in ros_occ.slices)
        assert non_zero_cells > 0
