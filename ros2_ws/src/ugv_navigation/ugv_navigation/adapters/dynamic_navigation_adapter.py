"""ROS2 Dynamic Navigation Adapter bridging Phase 13 dynamic planning stack with ROS2 middleware.

Encapsulates:
- Message conversions (TrackedObstacleArrayMsg, PredictedTrajectoryArrayMsg, BehavioralDecisionMsg, DynamicReplanEventMsg)
- Timestamp validation and staleness rejection (localization, map, tracks, predictions)
- Coordinated invocation of Predictive DWA, Behavioral State Machine, Event-Driven Replanner, and Safety Supervisor
- Strict command zeroing under STOP, WAIT, localization failure, or replan failure
- Guaranteed preservation of Phase 12 safety clamps
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
    DeadlockEvent,
)
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionConfig,
)
from ai.planning.dynamic.data_model import (
    DynamicObstacleDetectionArray,
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
    TrackedObstacleState,
)
from ai.planning.dynamic.detector_interface import (
    BaseDynamicObstacleDetector,
    SyntheticDynamicObstacleDetector,
)
from ai.planning.dynamic.dwa_filter import DynamicPlanningConfig
from ai.planning.dynamic.global_replanner import (
    DynamicPlanningMapOverlay,
    DynamicReplanEvent,
    DynamicReplanResult,
    DynamicReplanTrigger,
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
)
from ai.planning.dynamic.motion_estimator import MotionEstimator
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyConfig, DynamicOccupancyLayer
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig
from ai.planning.dynamic.tracker import MultiFrameTracker, TrackerConfig
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.navigator import UGVNavigationPlanner
from ai.planning.path import Path, PathPoint
from ai.spatial_fusion.world_map import WorldTerrainMap
from ugv_navigation.adapters.planner_adapter import PathMsg, PoseStampedMsg, TwistMsg
from ugv_navigation.messages.dynamic_messages import (
    BehavioralDecisionMsg,
    DynamicObstacleDetectionArrayMsg,
    DynamicOccupancyGridSliceMsg,
    DynamicOccupancySliceArrayMsg,
    DynamicReplanEventMsg,
    PredictedTrajectoryArrayMsg,
    TrackedObstacleArrayMsg,
)
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.utils.time_utils import RosHeader, RosTime


@dataclass
class DynamicAdapterConfig:
    """Tunable thresholds for ROS2 dynamic navigation adapter."""

    stale_loc_timeout_s: float = 0.50
    stale_map_timeout_s: float = 2.00
    stale_track_timeout_s: float = 0.50
    stale_pred_timeout_s: float = 0.50
    cmd_linear_max: float = 0.60
    cmd_angular_max: float = 0.80
    emergency_stop_distance_m: float = 0.35
    safety_clamp_distance_m: float = 0.60
    safety_clamped_linear_v: float = 0.20
    prediction_horizon_s: float = 2.0
    replan_cooldown_s: float = 2.0
    deadlock_time_threshold_s: float = 3.0
    stop_distance_m: float = 0.80
    wait_distance_m: float = 1.80


@dataclass
class DynamicPlanningResult:
    """Consolidated result of a dynamic planning cycle."""

    cmd_vel: TwistMsg
    global_path: PathMsg
    behavior: BehavioralDecisionMsg
    replan_event: Optional[DynamicReplanEventMsg]
    status: PlannerStatus
    diagnostics: Dict[str, Any]
    cycle_latency_ms: float = 0.0



class DynamicNavigationAdapter:
    """Full-pipeline deterministic adapter bridging ROS2 topics to Phase 13 dynamic navigation."""

    def __init__(
        self,
        config: Optional[DynamicAdapterConfig] = None,
        dwa_planner: Optional[DWAPlanner] = None,
        replanner: Optional[EventDrivenGlobalReplanner] = None,
        occupancy_layer: Optional[DynamicOccupancyLayer] = None,
    ) -> None:
        self.config = config or DynamicAdapterConfig()
        self.dwa_planner = dwa_planner or DWAPlanner()
        self.replanner = replanner or EventDrivenGlobalReplanner()
        self.occupancy_layer = occupancy_layer or DynamicOccupancyLayer()

        self._active_global_path: Optional[Path] = None
        self._last_cmd_v: float = 0.0
        self._last_cmd_w: float = 0.0
        self._last_behavioral_decision: Optional[BehavioralDecision] = None
        self._last_replan_event: Optional[DynamicReplanEvent] = None

    def loc_msg_to_robot_state(self, loc_msg: LocalizationStatusMsg) -> RobotState:
        """Extract RobotState from LocalizationStatusMsg."""
        p = loc_msg.pose.pose.position
        o = loc_msg.pose.pose.orientation
        t = loc_msg.twist.twist.linear
        w = loc_msg.twist.twist.angular

        siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
        cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
        yaw = float(math.atan2(siny_cosp, cosy_cosp))

        t_sec = float(loc_msg.header.stamp.to_sec())
        v_linear = float(t.x)
        w_angular = float(w.z)

        if abs(v_linear) < 1e-4 and abs(self._last_cmd_v) > 1e-4:
            v_linear = self._last_cmd_v
        if abs(w_angular) < 1e-4 and abs(self._last_cmd_w) > 1e-4:
            w_angular = self._last_cmd_w

        return RobotState(
            x=float(p.x),
            y=float(p.y),
            yaw=yaw,
            v=v_linear,
            omega=w_angular,
            timestamp=t_sec,
        )

    def goal_msg_to_goal_pose(self, goal_msg: PoseStampedMsg) -> GoalPose:
        """Extract GoalPose from PoseStampedMsg."""
        p = goal_msg.pose.position
        o = goal_msg.pose.orientation

        if getattr(goal_msg, "has_orientation", True):
            siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
            cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
            yaw = float(math.atan2(siny_cosp, cosy_cosp))
        else:
            yaw = None

        return GoalPose(
            x=float(p.x),
            y=float(p.y),
            yaw=yaw,
            frame_id=goal_msg.header.frame_id,
            timestamp=float(goal_msg.header.stamp.to_sec()),
            xy_tolerance_m=0.20,
        )

    def terrain_msg_to_world_map(self, map_msg: Union[WorldTerrainMapMsg, WorldTerrainMap]) -> WorldTerrainMap:
        """Convert WorldTerrainMapMsg or return WorldTerrainMap directly."""
        if isinstance(map_msg, WorldTerrainMap):
            return map_msg
        return map_msg.to_world_terrain_map()

    def tracks_msg_to_tracks(self, tracks_msg: Optional[TrackedObstacleArrayMsg]) -> List[TrackedObstacleState]:
        """Convert TrackedObstacleArrayMsg to list of TrackedObstacleState."""
        if tracks_msg is None:
            return []
        return [t.to_phase13() for t in tracks_msg.tracks]

    def preds_msg_to_trajectories(self, preds_msg: Optional[PredictedTrajectoryArrayMsg]) -> List[PredictedObstacleTrajectory]:
        """Convert PredictedTrajectoryArrayMsg to list of PredictedObstacleTrajectory."""
        if preds_msg is None:
            return []
        res = []
        for t in preds_msg.trajectories:
            traj = t.to_phase13()
            traj.bbox_yaw = float(t.bbox_yaw)
            traj.bbox_size = (float(t.bbox_size[0]), float(t.bbox_size[1]))
            res.append(traj)
        return res

    def validate_inputs(
        self,
        loc_msg: Optional[LocalizationStatusMsg],
        map_msg: Optional[Union[WorldTerrainMapMsg, WorldTerrainMap]],
        current_time_s: float = 0.0,
        tracks_msg: Optional[TrackedObstacleArrayMsg] = None,
        preds_msg: Optional[PredictedTrajectoryArrayMsg] = None,
    ) -> Tuple[bool, str]:
        """Convenience method checking freshness and frame conventions."""
        return self.validate_freshness(
            current_time_s=current_time_s,
            loc_msg=loc_msg,
            map_msg=map_msg,
            tracks_msg=tracks_msg,
            preds_msg=preds_msg,
        )

    def validate_freshness(
        self,
        current_time_s: float,
        loc_msg: Optional[LocalizationStatusMsg],
        map_msg: Optional[Union[WorldTerrainMapMsg, WorldTerrainMap]],
        tracks_msg: Optional[TrackedObstacleArrayMsg] = None,
        preds_msg: Optional[PredictedTrajectoryArrayMsg] = None,
    ) -> Tuple[bool, str]:
        """Verify that incoming message streams are fresh and have matching frame_id conventions."""
        if loc_msg is None:
            return False, "Localization message is None"
        if map_msg is None:
            return False, "WorldTerrainMap message is None"

        t_loc = float(loc_msg.header.stamp.to_sec())
        if (current_time_s - t_loc) > self.config.stale_loc_timeout_s:
            return False, f"Localization is stale: age {current_time_s - t_loc:.3f}s > {self.config.stale_loc_timeout_s}s"

        t_map = float(map_msg.header.stamp.to_sec()) if hasattr(map_msg, "header") else float(getattr(map_msg, "created_timestamp", current_time_s))
        if (current_time_s - t_map) > self.config.stale_map_timeout_s:
            return False, f"WorldTerrainMap is stale: age {current_time_s - t_map:.3f}s > {self.config.stale_map_timeout_s}s"

        # Frame consistency check
        if loc_msg.header.frame_id != "odom":
            return False, f"Invalid localization frame_id '{loc_msg.header.frame_id}', expected 'odom'"

        if hasattr(map_msg, "header") and map_msg.header.frame_id != "odom":
            return False, f"Invalid terrain map frame_id '{map_msg.header.frame_id}', expected 'odom'"

        return True, "Fresh"

    def apply_safety_clamp(
        self,
        cmd_v: float,
        cmd_w: float,
        min_clearance_m: float,
        behavioral_state: BehavioralState,
    ) -> Tuple[float, float, bool]:
        """Enforce Phase 12 supervisory safety clamps over candidate velocity commands."""
        # 1. Genuine zero-motion states
        if behavioral_state in (BehavioralState.STOP, BehavioralState.WAIT):
            return 0.0, 0.0, False

        # 2. Hard emergency stop proximity threshold (d < 0.35m)
        if min_clearance_m < self.config.emergency_stop_distance_m:
            return 0.0, 0.0, True

        # 3. Speed clamp margin (0.35m <= d < 0.60m)
        clamped_v = cmd_v
        if min_clearance_m < self.config.safety_clamp_distance_m:
            clamped_v = min(clamped_v, self.config.safety_clamped_linear_v)

        # 4. Physical kinematic saturation limits
        clamped_v = max(-self.config.cmd_linear_max, min(self.config.cmd_linear_max, clamped_v))
        clamped_w = max(-self.config.cmd_angular_max, min(self.config.cmd_angular_max, cmd_w))

        return float(clamped_v), float(clamped_w), False

    def plan_step(
        self,
        loc_msg: LocalizationStatusMsg,
        map_msg: Union[WorldTerrainMapMsg, WorldTerrainMap],
        goal_msg: Optional[PoseStampedMsg] = None,
        tracks_msg: Optional[TrackedObstacleArrayMsg] = None,
        preds_msg: Optional[PredictedTrajectoryArrayMsg] = None,
        sim_time_s: Optional[float] = None,
        dt_s: float = 0.10,
        tf_available: bool = True,
    ) -> Tuple[TwistMsg, PathMsg, BehavioralDecisionMsg, Optional[DynamicReplanEventMsg], PlannerStatus, Dict[str, Any]]:
        """Run one complete integrated dynamic navigation step from ROS2 message inputs."""
        t_now = sim_time_s if sim_time_s is not None else float(loc_msg.header.stamp.to_sec())

        # 1. Freshness & frame validation
        is_fresh, fresh_msg = self.validate_freshness(t_now, loc_msg, map_msg, tracks_msg, preds_msg)
        if not is_fresh or not tf_available or loc_msg.tracking_state == LocalizationStatusMsg.LOST:
            # Immediate fail-safe emergency stop
            zero_cmd = TwistMsg.create(0.0, 0.0)
            zero_dec = BehavioralDecision(
                state=BehavioralState.STOP,
                command_v=0.0,
                command_omega=0.0,
                reason=f"Safety intervention: {fresh_msg}",
            )
            dec_msg = BehavioralDecisionMsg.from_phase13(zero_dec, timestamp_s=t_now)
            path_msg = PathMsg.from_planner_path(self._active_global_path, timestamp_s=t_now)
            status = PlannerStatus.LOCALIZATION_INVALID if not is_fresh else PlannerStatus.EMERGENCY_STOP
            return zero_cmd, path_msg, dec_msg, None, status, {"error": fresh_msg, "emergency_stop": True}

        # 2. Extract types
        robot_state = self.loc_msg_to_robot_state(loc_msg)
        world_map = self.terrain_msg_to_world_map(map_msg)
        goal_pose = self.goal_msg_to_goal_pose(goal_msg) if goal_msg is not None else None

        # Extract dynamic tracks & predictions with staleness validation
        tracks = []
        if tracks_msg is not None:
            t_tr = float(tracks_msg.header.stamp.to_sec())
            if (t_now - t_tr) <= self.config.stale_track_timeout_s:
                tracks = self.tracks_msg_to_tracks(tracks_msg)

        predictions = []
        if preds_msg is not None:
            t_pr = float(preds_msg.header.stamp.to_sec())
            if (t_now - t_pr) <= self.config.stale_pred_timeout_s:
                predictions = self.preds_msg_to_trajectories(preds_msg)

        # 3. Initial Global Path Generation if not present
        if self._active_global_path is None or len(self._active_global_path.points) == 0:
            if goal_pose is not None:
                initial_replan = self.replanner.execute_replan(
                    current_pose=robot_state.pose,
                    goal=(goal_pose.x, goal_pose.y),
                    base_map=world_map,
                    dynamic_obstacles=tracks if tracks else predictions,
                )
                if initial_replan.success and initial_replan.path is not None:
                    self._active_global_path = initial_replan.path
                    self.replanner.set_global_path(self._active_global_path)

        # 4. Check Event-Driven Global Replanning Triggers
        replan_event_msg: Optional[DynamicReplanEventMsg] = None
        if goal_pose is not None and self._active_global_path is not None:
            replan_res = self.replanner.check_and_replan(
                current_pose=robot_state.pose,
                goal=(goal_pose.x, goal_pose.y),
                base_map=world_map,
                behavioral_decision=self._last_behavioral_decision,
                dynamic_obstacles=tracks if tracks else predictions,
                timestamp=t_now,
            )
            if replan_res.replan_triggered:
                if replan_res.event:
                    replan_event_msg = DynamicReplanEventMsg.from_phase13(replan_res.event, replan_res)
                if replan_res.success and replan_res.path is not None:
                    self._active_global_path = replan_res.path
                elif not replan_res.success:
                    # GLOBAL_REPLAN_FAILED: Hold STOP/WAIT, do not invent path
                    self._active_global_path = None

        # 5. Check if goal is already reached
        if goal_pose is not None:
            dist_to_goal = math.hypot(robot_state.x - goal_pose.x, robot_state.y - goal_pose.y)
            if dist_to_goal <= 0.20:
                zero_cmd = TwistMsg.create(0.0, 0.0)
                goal_dec = BehavioralDecision(
                    state=BehavioralState.STOP,
                    command_v=0.0,
                    command_omega=0.0,
                    reason="Goal successfully reached",
                )
                dec_msg = BehavioralDecisionMsg.from_phase13(goal_dec, timestamp_s=t_now)
                path_msg = PathMsg.from_planner_path(self._active_global_path, timestamp_s=t_now)
                return zero_cmd, path_msg, dec_msg, replan_event_msg, PlannerStatus.GOAL_REACHED, {"goal_reached": True}

        # 6. If active path is None or stale -> zero command
        if self._active_global_path is None or self.replanner.is_current_path_stale:
            zero_cmd = TwistMsg.create(0.0, 0.0)
            stale_dec = BehavioralDecision(
                state=BehavioralState.STOP,
                command_v=0.0,
                command_omega=0.0,
                reason="Global route unavailable or stale",
            )
            dec_msg = BehavioralDecisionMsg.from_phase13(stale_dec, timestamp_s=t_now)
            path_msg = PathMsg.from_planner_path(None, timestamp_s=t_now)
            return zero_cmd, path_msg, dec_msg, replan_event_msg, PlannerStatus.NO_PATH, {"no_path": True}

        # 7. Predictive DWA Local Rollout & Tactical Behavioral Arbitration
        traj, raw_v, raw_w, dwa_status, dwa_diag, decision = self.dwa_planner.plan_with_behavior(
            current_state=robot_state,
            global_path=self._active_global_path,
            costmap=world_map,
            dynamic_layer=self.occupancy_layer,
            predicted_tracks=predictions,
        )

        self._last_behavioral_decision = decision

        # 8. Apply Phase 12 Safety Clamping
        min_clear = float(dwa_diag.get("minimum_dynamic_clearance", 1.0))
        clamped_v, clamped_w, emergency_stop = self.apply_safety_clamp(
            cmd_v=decision.command_v,
            cmd_w=decision.command_omega,
            min_clearance_m=min_clear,
            behavioral_state=decision.state,
        )

        self._last_cmd_v = clamped_v
        self._last_cmd_w = clamped_w

        twist_cmd = TwistMsg.create(clamped_v, clamped_w)
        path_msg = PathMsg.from_planner_path(self._active_global_path, frame_id="odom", timestamp_s=t_now)
        dec_msg = BehavioralDecisionMsg.from_phase13(decision, timestamp_s=t_now)

        diagnostics = {
            "behavioral_decision": decision.to_dict(),
            "dwa_diagnostics": dwa_diag,
            "emergency_stop": emergency_stop,
            "min_clearance_m": min_clear,
            "command_v": clamped_v,
            "command_omega": clamped_w,
        }

        return twist_cmd, path_msg, dec_msg, replan_event_msg, dwa_status, diagnostics

    def step(
        self,
        loc_msg: Optional[LocalizationStatusMsg],
        map_msg: Optional[Union[WorldTerrainMapMsg, WorldTerrainMap]],
        goal_msg: Optional[PoseStampedMsg] = None,
        tracks_msg: Optional[TrackedObstacleArrayMsg] = None,
        predictions_msg: Optional[PredictedTrajectoryArrayMsg] = None,
        sim_time_s: Optional[float] = None,
        dt_s: float = 0.10,
        tf_available: bool = True,
    ) -> DynamicPlanningResult:
        """Convenience single-call interface returning a consolidated DynamicPlanningResult."""
        import time
        t0 = time.perf_counter()

        if loc_msg is None:
            zero_cmd = TwistMsg.create(0.0, 0.0)
            empty_path = PathMsg.from_planner_path(None)
            dec = BehavioralDecisionMsg.from_phase13(
                BehavioralDecision(
                    state=BehavioralState.STOP,
                    command_v=0.0,
                    command_omega=0.0,
                    reason="No localization received",
                )
            )
            return DynamicPlanningResult(
                cmd_vel=zero_cmd,
                global_path=empty_path,
                behavior=dec,
                replan_event=None,
                status=PlannerStatus.LOCALIZATION_INVALID,
                diagnostics={"error": "No localization received"},
                cycle_latency_ms=0.0,
            )

        if map_msg is None:
            zero_cmd = TwistMsg.create(0.0, 0.0)
            empty_path = PathMsg.from_planner_path(None)
            dec = BehavioralDecisionMsg.from_phase13(
                BehavioralDecision(
                    state=BehavioralState.STOP,
                    command_v=0.0,
                    command_omega=0.0,
                    reason="No WorldTerrainMap received",
                )
            )
            return DynamicPlanningResult(
                cmd_vel=zero_cmd,
                global_path=empty_path,
                behavior=dec,
                replan_event=None,
                status=PlannerStatus.MAP_UNAVAILABLE,
                diagnostics={"error": "No WorldTerrainMap received"},
                cycle_latency_ms=0.0,
            )

        cmd, path, dec, replan_ev, status, diag = self.plan_step(
            loc_msg=loc_msg,
            map_msg=map_msg,
            goal_msg=goal_msg,
            tracks_msg=tracks_msg,
            preds_msg=predictions_msg,
            sim_time_s=sim_time_s,
            dt_s=dt_s,
            tf_available=tf_available,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DynamicPlanningResult(
            cmd_vel=cmd,
            global_path=path,
            behavior=dec,
            replan_event=replan_ev,
            status=status,
            diagnostics=diag,
            cycle_latency_ms=latency_ms,
        )

