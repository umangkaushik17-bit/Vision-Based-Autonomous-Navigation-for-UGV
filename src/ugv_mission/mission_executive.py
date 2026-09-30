"""High-level autonomous mission executive and sequential waypoint orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ai.control.command_buffer import SafeCommandBuffer
from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.dynamic.data_model import TrackedObstacleState
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.path import Path
from ai.planning.recovery_controller import RecoveryState
from ai.planning.safety_governor import LatchedHazard, SafetyGovernor

from ugv_mission.mission_state import (
    MissionAbortReason,
    MissionState,
    MissionStatus,
    is_valid_mission_transition,
)
from ugv_mission.recovery_orchestrator import (
    MissionRecoveryOrchestrator,
    RecoveryOrchestratorConfig,
)
from ugv_mission.waypoint import Waypoint
from ugv_mission.waypoint_manager import WaypointManager

logger = logging.getLogger("ugv_mission.executive")


@dataclass
class MissionExecutiveConfig:
    """Configurable parameters for mission execution and safety watchdogs."""

    mission_name: str = "default_mission"
    default_xy_tolerance_m: float = 0.35
    max_recovery_attempts: int = 3
    progress_window_s: float = 2.50
    min_progress_displacement_m: float = 0.15
    auto_replan_on_waypoint_advance: bool = True


class MissionExecutive:
    """Orchestrates multi-waypoint autonomous missions on top of Phase 13 navigation.
    
    Strict Invariants:
    1. Waypoints are executed strictly in order (W1 -> W2 -> ... -> WN).
    2. Waypoints are NEVER skipped or marked reached prematurely.
    3. Arrival is strictly checked via position/heading tolerances.
    4. Recovery is delegated to the verified Phase 13 RecoveryController.
    5. Recovery success replans toward the SAME active waypoint.
    6. Recovery retry exhaustion enters MISSION_ABORTED and enforces a safe stop.
    7. All motion commands pass through the authoritative Safety Governor and SafeCommandBuffer.
    """

    def __init__(
        self,
        config: Optional[MissionExecutiveConfig] = None,
        global_planner: Optional[Any] = None,
        safety_governor: Optional[SafetyGovernor] = None,
        command_buffer: Optional[SafeCommandBuffer] = None,
    ) -> None:
        self.config = config or MissionExecutiveConfig()
        self.waypoint_manager = WaypointManager(mission_name=self.config.mission_name)
        self.recovery_orchestrator = MissionRecoveryOrchestrator(
            config=RecoveryOrchestratorConfig(max_recovery_attempts=self.config.max_recovery_attempts)
        )
        self.global_planner = global_planner or AStarPlanner()
        self.safety_governor = safety_governor or SafetyGovernor()
        self.command_buffer = command_buffer

        # Internal state machine
        self._state: MissionState = MissionState.IDLE
        self._abort_reason: MissionAbortReason = MissionAbortReason.NONE
        self._active_global_path: Optional[Path] = None
        self._last_step_time_s: float = 0.0

        # Spatial progress tracking for deadlock detection
        self._pose_history: List[Tuple[float, float, float]] = []  # (t, x, y)

    @property
    def state(self) -> MissionState:
        """Current mission state."""
        return self._state

    @property
    def abort_reason(self) -> MissionAbortReason:
        """Reason for mission abort, if aborted."""
        return self._abort_reason

    @property
    def active_global_path(self) -> Optional[Path]:
        """Current active global path to the active waypoint."""
        return self._active_global_path

    @property
    def active_waypoint(self) -> Optional[Waypoint]:
        """Currently active target waypoint."""
        return self.waypoint_manager.active_waypoint

    @property
    def active_index(self) -> int:
        """0-based index of currently active waypoint."""
        return self.waypoint_manager.active_index

    @property
    def total_waypoints(self) -> int:
        """Total waypoints in active mission."""
        return self.waypoint_manager.total_count

    def _transition_to(self, new_state: MissionState, reason: str = "") -> None:
        """Enforce valid state transition rules and update state."""
        if not is_valid_mission_transition(self._state, new_state):
            raise RuntimeError(
                f"[MISSION] Invalid state transition rejected: {self._state.value} -> {new_state.value}. "
                f"Reason: {reason}"
            )

        old_state = self._state
        self._state = new_state
        logger.info(f"[MISSION] Transition: {old_state.value} -> {new_state.value} | {reason}")

    def load_mission(self, waypoints: Sequence[Union[Waypoint, Dict[str, Any]]]) -> Tuple[bool, str]:
        """Load ordered waypoint list into the executive."""
        self._transition_to(MissionState.INITIALIZING, "Loading new mission")
        success, reason = self.waypoint_manager.load_waypoints(waypoints)
        if not success:
            self._abort_reason = (
                MissionAbortReason.EMPTY_WAYPOINTS if "empty" in reason.lower()
                else MissionAbortReason.INVALID_MISSION
            )
            self._transition_to(MissionState.MISSION_ABORTED, f"Load failed: {reason}")
            return False, reason

        self._abort_reason = MissionAbortReason.NONE
        self._active_global_path = None
        self._pose_history.clear()
        self.recovery_orchestrator.reset()
        logger.info(
            f"[MISSION] Mission '{self.config.mission_name}' loaded successfully "
            f"with {self.waypoint_manager.total_count} waypoints."
        )
        return True, "ok"

    def load_from_yaml(self, file_path: str) -> Tuple[bool, str]:
        """Load mission from a YAML file."""
        self._transition_to(MissionState.INITIALIZING, f"Loading YAML: {file_path}")
        success, reason = self.waypoint_manager.load_from_yaml(file_path)
        if not success:
            self._abort_reason = MissionAbortReason.INVALID_MISSION
            self._transition_to(MissionState.MISSION_ABORTED, f"YAML load failed: {reason}")
            return False, reason

        self._abort_reason = MissionAbortReason.NONE
        self._active_global_path = None
        self._pose_history.clear()
        self.recovery_orchestrator.reset()
        return True, "ok"

    def start_mission(
        self,
        current_pose: Tuple[float, float, float],
        costmap: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """Start or resume mission execution toward the first waypoint."""
        if self._state not in (MissionState.INITIALIZING, MissionState.IDLE):
            if self._state == MissionState.NAVIGATING_TO_WAYPOINT:
                return True, "Already navigating"
            return False, f"Cannot start mission from state {self._state.value}"

        if self.waypoint_manager.total_count == 0:
            self._abort_reason = MissionAbortReason.EMPTY_WAYPOINTS
            self._transition_to(MissionState.MISSION_ABORTED, "Cannot start mission with 0 waypoints")
            return False, "Empty waypoints"

        wp = self.waypoint_manager.active_waypoint
        if wp is None:
            self._abort_reason = MissionAbortReason.INVALID_MISSION
            self._transition_to(MissionState.MISSION_ABORTED, "No active waypoint")
            return False, "No active waypoint"

        self.recovery_orchestrator.on_waypoint_changed(wp.id)

        # Compute initial global path if costmap provided
        if costmap is not None:
            plan_ok = self._replan_to_active_waypoint(current_pose, costmap)
            if not plan_ok:
                logger.warning(f"[MISSION] Initial path planning to waypoint {wp.id} failed; holding in navigation state.")

        self._transition_to(
            MissionState.NAVIGATING_TO_WAYPOINT,
            f"Starting mission toward waypoint {wp.id} ({self.active_index + 1}/{self.total_waypoints})"
        )
        return True, "ok"

    def _replan_to_active_waypoint(
        self,
        current_pose: Tuple[float, float, float],
        costmap: Any,
        dynamic_obstacles: Optional[Sequence[Any]] = None,
    ) -> bool:
        """Compute/replan global path to the current active waypoint."""
        wp = self.waypoint_manager.active_waypoint
        if wp is None:
            return False

        # Support both WorldTerrainMap and BEVTerrainCostmap seamlessly
        if costmap is not None and hasattr(costmap, "world_to_grid") and not hasattr(costmap, "meter_to_grid"):
            try:
                setattr(costmap, "meter_to_grid", getattr(costmap, "world_to_grid"))
                setattr(costmap, "grid_to_meter", getattr(costmap, "grid_to_world"))
            except Exception:
                pass

        rx, ry, ryaw = current_pose
        start_state = RobotState(x=rx, y=ry, yaw=ryaw)
        goal_state = wp.to_goal_pose()

        # Handle EventDrivenGlobalReplanner or AStarPlanner
        if hasattr(self.global_planner, "execute_replan"):
            res = self.global_planner.execute_replan(
                current_pose=current_pose,
                goal=(wp.x, wp.y),
                base_map=costmap,
                dynamic_obstacles=dynamic_obstacles or [],
            )
            if res.success and res.path is not None:
                self._active_global_path = res.path
                if hasattr(self.global_planner, "set_global_path"):
                    self.global_planner.set_global_path(self._active_global_path)
                return True
            return False

        elif hasattr(self.global_planner, "plan"):
            path, status, _ = self.global_planner.plan(
                costmap=costmap,
                start=start_state,
                goal=goal_state,
            )
            if status == PlannerStatus.SUCCESS and path is not None:
                self._active_global_path = path
                return True
            return False

        return False

    def _check_spatial_progress(
        self,
        current_time_s: float,
        current_pose: Tuple[float, float, float],
        dist_to_wp: float,
    ) -> bool:
        """Detect lack of spatial progress over the configured time window."""
        t_now = current_time_s
        rx, ry, _ = current_pose
        self._pose_history.append((t_now, rx, ry))

        # Trim history older than window
        cutoff = t_now - self.config.progress_window_s
        self._pose_history = [p for p in self._pose_history if p[0] >= cutoff]

        if len(self._pose_history) >= 6 and (t_now - self._pose_history[0][0]) >= (self.config.progress_window_s * 0.75):
            start_x, start_y = self._pose_history[0][1], self._pose_history[0][2]
            disp = math.hypot(rx - start_x, ry - start_y)
            if disp < self.config.min_progress_displacement_m and dist_to_wp > 0.40:
                return True  # Lack of progress
        return False

    def step(
        self,
        current_time_s: float,
        robot_pose: Tuple[float, float, float],
        planner_status: PlannerStatus = PlannerStatus.FOLLOWING,
        hazards: Optional[Sequence[Union[TrackedObstacleState, LatchedHazard]]] = None,
        costmap: Any = None,
        dynamic_obstacles: Optional[Sequence[Any]] = None,
        dwa_decision_str: str = "PROCEED",
        deadlock_flag: bool = False,
    ) -> Tuple[MissionState, Optional[float], Optional[float], str]:
        """Main periodic mission executive tick.
        
        Evaluates arrival, deadlock, recovery, replanning, and goal progression.
        
        Returns:
            Tuple of:
            - state: Active MissionState
            - cmd_v: Commanded linear velocity (None if delegated to nominal DWA)
            - cmd_w: Commanded angular velocity (None if delegated to nominal DWA)
            - reason: Diagnostic text
        """
        self._last_step_time_s = current_time_s
        rx, ry, ryaw = robot_pose
        active_wp = self.waypoint_manager.active_waypoint

        # Terminal state handling
        if self._state == MissionState.MISSION_COMPLETE:
            self._halt_ugv("Mission already completed")
            return MissionState.MISSION_COMPLETE, 0.0, 0.0, "MISSION_COMPLETE"

        if self._state == MissionState.MISSION_ABORTED:
            self._halt_ugv(f"Mission aborted: {self._abort_reason.value}")
            return MissionState.MISSION_ABORTED, 0.0, 0.0, f"MISSION_ABORTED: {self._abort_reason.value}"

        if self._state in (MissionState.IDLE, MissionState.INITIALIZING):
            return self._state, 0.0, 0.0, f"Holding in {self._state.value}"

        if active_wp is None:
            self._abort_reason = MissionAbortReason.INVALID_MISSION
            self._transition_to(MissionState.MISSION_ABORTED, "Active waypoint disappeared")
            self._halt_ugv("No active waypoint")
            return MissionState.MISSION_ABORTED, 0.0, 0.0, "No active waypoint"

        dist_to_wp = active_wp.distance_to(robot_pose)

        # ---------------------------------------------------------------------
        # 1. Check Waypoint Arrival Condition (NON-NEGOTIABLE ARRIVAL GUARD)
        # ---------------------------------------------------------------------
        if active_wp.is_reached(robot_pose):
            self.waypoint_manager.mark_active_reached(robot_pose)
            self._transition_to(
                MissionState.WAYPOINT_REACHED,
                f"Waypoint {active_wp.id} reached (dist={dist_to_wp:.3f}m <= {active_wp.xy_tolerance_m:.3f}m)"
            )

            # Check if this was the FINAL waypoint
            if self.waypoint_manager.is_final_waypoint:
                self._transition_to(
                    MissionState.MISSION_COMPLETE,
                    f"Final waypoint {active_wp.id} reached! Mission completed successfully."
                )
                self._halt_ugv("FINAL_WAYPOINT_REACHED")
                return MissionState.MISSION_COMPLETE, 0.0, 0.0, "MISSION_COMPLETE"

            # Advance to next waypoint
            next_wp = self.waypoint_manager.advance()
            if next_wp is not None:
                self._pose_history.clear()
                self.recovery_orchestrator.on_waypoint_changed(next_wp.id)
                if costmap is not None and self.config.auto_replan_on_waypoint_advance:
                    self._replan_to_active_waypoint(robot_pose, costmap, dynamic_obstacles)

                self._transition_to(
                    MissionState.NAVIGATING_TO_WAYPOINT,
                    f"Advancing to waypoint {next_wp.id} ({self.active_index + 1}/{self.total_waypoints})"
                )
                return self._state, None, None, f"Advanced to {next_wp.id}"
            else:
                self._transition_to(MissionState.MISSION_COMPLETE, "All waypoints completed.")
                self._halt_ugv("ALL_WAYPOINTS_COMPLETE")
                return MissionState.MISSION_COMPLETE, 0.0, 0.0, "MISSION_COMPLETE"

        # ---------------------------------------------------------------------
        # 2. In RECOVERY State
        # ---------------------------------------------------------------------
        if self._state == MissionState.RECOVERY:
            # Step the recovery orchestrator
            cmd_v, cmd_w, rec_state, retry_exceeded, rec_reason = self.recovery_orchestrator.step(
                current_time_s=current_time_s,
                robot_pose=robot_pose,
                is_deadlocked=True,
                hazards=hazards or [],
                costmap=costmap,
            )

            if retry_exceeded:
                self._abort_reason = (
                    MissionAbortReason.RECOVERY_PERMANENTLY_BLOCKED
                    if rec_state == RecoveryState.ABORTED_REAR_BLOCKED
                    else MissionAbortReason.RECOVERY_LIMIT_EXCEEDED
                )
                self._transition_to(
                    MissionState.MISSION_ABORTED,
                    f"Recovery retry limit reached at {active_wp.id}: {rec_reason}"
                )
                self._halt_ugv("RECOVERY_RETRY_LIMIT_EXCEEDED")
                return MissionState.MISSION_ABORTED, 0.0, 0.0, rec_reason

            if rec_state in (RecoveryState.ABORTED_REAR_BLOCKED, RecoveryState.FAILED_TIMEOUT):
                # Recovery attempt failed; hold safe stop while in RECOVERY awaiting retry or timeout
                return self._state, 0.0, 0.0, f"RECOVERY_ATTEMPT_FAILED: {rec_reason}"

            if rec_state == RecoveryState.REPLAN_READY:
                # Disengagement complete: trigger replan toward SAME waypoint
                self._transition_to(MissionState.REPLANNING, f"Disengaged from obstacle; replanning to {active_wp.id}")
                replan_ok = self._replan_to_active_waypoint(robot_pose, costmap, dynamic_obstacles)
                if replan_ok:
                    self.recovery_orchestrator.mark_replan_completed()
                    self._pose_history.clear()
                    self._transition_to(
                        MissionState.NAVIGATING_TO_WAYPOINT,
                        f"Replan succeeded to {active_wp.id}; resuming navigation"
                    )
                    return self._state, None, None, "Replan succeeded; resuming navigation"
                else:
                    logger.warning(f"[MISSION] Replan to {active_wp.id} failed after recovery disengagement.")
                    # Return zero command while holding
                    return self._state, 0.0, 0.0, "Replan failed; holding safe stop"

            # Continue executing recovery motion (e.g. controlled reverse)
            # Route through SafetyGovernor to ensure rear safety
            if cmd_v is not None and cmd_w is not None:
                gov_v, gov_w, gov_dec, _ = self.safety_governor.evaluate(
                    requested_v=cmd_v,
                    requested_w=cmd_w,
                    robot_pose=robot_pose,
                    hazards=hazards or [],
                    current_time_s=current_time_s,
                )
                if self.command_buffer is not None:
                    self.command_buffer.set_command(
                        linear_v=gov_v,
                        angular_w=gov_w,
                        timestamp_s=current_time_s,
                        is_safe=("STOP" not in gov_dec),
                        source="RECOVERY",
                        reason=f"RECOVERY | {rec_reason}",
                    )
                return self._state, gov_v, gov_w, f"RECOVERY: {rec_reason}"

            return self._state, 0.0, 0.0, f"RECOVERY: {rec_reason}"

        # ---------------------------------------------------------------------
        # 3. In REPLANNING State
        # ---------------------------------------------------------------------
        if self._state == MissionState.REPLANNING:
            replan_ok = self._replan_to_active_waypoint(robot_pose, costmap, dynamic_obstacles)
            if replan_ok:
                self.recovery_orchestrator.mark_replan_completed()
                self._pose_history.clear()
                self._transition_to(
                    MissionState.NAVIGATING_TO_WAYPOINT,
                    f"Replan succeeded to {active_wp.id}; resuming navigation"
                )
                return self._state, None, None, "Replan succeeded"
            else:
                self._transition_to(MissionState.RECOVERY, "Replan failed; returning to recovery")
                return self._state, 0.0, 0.0, "Replan failed"

        # ---------------------------------------------------------------------
        # 4. In NAVIGATING_TO_WAYPOINT State: Deadlock & Failure Monitoring
        # ---------------------------------------------------------------------
        lack_of_progress = self._check_spatial_progress(current_time_s, robot_pose, dist_to_wp)
        is_deadlock = (
            deadlock_flag
            or (dwa_decision_str in ("STOP", "EMERGENCY_STOP") and dist_to_wp > 0.40)
            or (planner_status in (PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, PlannerStatus.NO_SAFE_TRAJECTORY) and dist_to_wp > 0.40)
            or (planner_status == PlannerStatus.RECOVERY_REQUIRED)
            or lack_of_progress
        )

        if is_deadlock:
            deadlock_reason = "lack_of_progress" if lack_of_progress else (
                f"planner_status={planner_status.value}, decision={dwa_decision_str}"
            )
            logger.warning(f"[MISSION] Deadlock recognized at waypoint {active_wp.id}: {deadlock_reason}")
            self._transition_to(
                MissionState.RECOVERY,
                f"Entering recovery at {active_wp.id} due to {deadlock_reason}"
            )
            # Step recovery immediately
            return self.step(
                current_time_s=current_time_s,
                robot_pose=robot_pose,
                planner_status=planner_status,
                hazards=hazards,
                costmap=costmap,
                dynamic_obstacles=dynamic_obstacles,
                dwa_decision_str=dwa_decision_str,
                deadlock_flag=True,
            )

        # Nominal navigation: Return None velocities so caller uses DWA output
        return MissionState.NAVIGATING_TO_WAYPOINT, None, None, f"Navigating to {active_wp.id} (dist={dist_to_wp:.2f}m)"

    def _halt_ugv(self, reason: str) -> None:
        """Command safe emergency stop through the command buffer."""
        if self.command_buffer is not None:
            self.command_buffer.emergency_stop(
                timestamp_s=self._last_step_time_s if self._last_step_time_s > 0 else time.time(),
                reason=reason,
            )

    def cancel_mission(self, reason: str = "USER_CANCELLED") -> None:
        """Cancel active mission safely."""
        self._abort_reason = MissionAbortReason.USER_CANCELLED
        self._transition_to(MissionState.MISSION_ABORTED, f"Cancelled: {reason}")
        self._halt_ugv(f"MISSION_CANCELLED: {reason}")

    def reset(self) -> None:
        """Reset mission executive back to IDLE state."""
        self._state = MissionState.IDLE
        self._abort_reason = MissionAbortReason.NONE
        self._active_global_path = None
        self._pose_history.clear()
        self.waypoint_manager.reset()
        self.recovery_orchestrator.reset()
        self._halt_ugv("MISSION_RESET")
        logger.info("[MISSION] Mission executive reset to IDLE.")

    def get_status(self, current_pose: Optional[Tuple[float, float, float]] = None) -> MissionStatus:
        """Generate current mission telemetry status snapshot."""
        wp = self.waypoint_manager.active_waypoint
        dist = 0.0
        if wp is not None and current_pose is not None:
            dist = wp.distance_to(current_pose)

        return MissionStatus(
            mission_name=self.config.mission_name,
            state=self._state,
            active_waypoint_id=wp.id if wp else None,
            active_waypoint_index=self.active_index,
            total_waypoints=self.total_waypoints,
            distance_to_active_waypoint=dist,
            recovery_attempt_count=self.recovery_orchestrator.attempt_count,
            last_recovery_result=self.recovery_orchestrator.last_result,
            is_complete=(self._state == MissionState.MISSION_COMPLETE),
            is_aborted=(self._state == MissionState.MISSION_ABORTED),
            abort_reason=self._abort_reason,
            timestamp=self._last_step_time_s if self._last_step_time_s > 0 else time.time(),
            reached_waypoint_ids=self.waypoint_manager.reached_ids,
            diagnostics={
                "total_recovery_count": self.recovery_orchestrator.total_recovery_count,
                "has_global_path": self._active_global_path is not None,
            },
        )
