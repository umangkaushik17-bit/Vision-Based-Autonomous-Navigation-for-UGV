"""Deterministic Behavioral State Machine for Dynamic Local Planning (Milestone 13.10).

Governs UGV tactical decision layer around predictive DWA candidate evaluations:
- PROCEED: Nominal travel along preferred path with acceptable risk.
- AVOID: Executing a collision-free local detour when preferred path is obstructed.
- WAIT: Genuine zero-motion holding state (v=0, w=0) when an obstacle is predicted to clear.
- STOP: Genuine zero-motion safety intervention (v=0, w=0) when no safe motion is possible.

Enforces state persistence, hysteresis, dwell times, wait timeouts, and deadlock detection
(emitting DYNAMIC_DEADLOCK event for Milestone 13.11 global replanning).
Preserves Phase 12 emergency stop and static safety gating as strictly authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ai.planning.base import PlannerStatus, RobotState
from ai.planning.dynamic.collision_checker import CollisionStatus, PredictiveCollisionResult
from ai.planning.dynamic.data_model import PredictedObstacleTrajectory, PredictedTrajectoryArray
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory


class BehavioralState(str, Enum):
    """Core operational states for tactical local navigation."""

    PROCEED = "PROCEED"
    AVOID = "AVOID"
    WAIT = "WAIT"
    STOP = "STOP"


class DeadlockEvent(str, Enum):
    """Deadlock classification emitted to trigger global replanning."""

    NONE = "NONE"
    DYNAMIC_DEADLOCK = "DYNAMIC_DEADLOCK"


@dataclass
class BehavioralConfig:
    """Tunable parameters for behavioral state arbitration and hysteresis."""

    # Wait behavior
    wait_timeout_s: float = 3.0
    wait_clearance_predicted_horizon_s: float = 1.50
    wait_corridor_half_width_m: float = 0.55
    prefer_wait_over_risky_avoid: bool = True

    # Hysteresis and dwell
    min_dwell_time_s: float = 0.20
    recovery_confirmation_cycles: int = 2
    avoid_heading_threshold_rad: float = 0.08
    avoid_omega_threshold_radps: float = 0.06

    # Deadlock detection
    deadlock_consecutive_stop_cycles: int = 5
    deadlock_max_wait_timeouts: int = 2
    deadlock_progress_window_s: float = 5.0
    deadlock_progress_threshold_m: float = 0.10

    # Risk thresholds
    max_acceptable_proceed_risk: str = "LOW"  # FREE or LOW allowed for PROCEED
    unknown_horizon_policy: str = "PENALIZE"  # PENALIZE, REJECT, ALLOW


@dataclass
class BehavioralDecision:
    """Structured, serializable decision output from the behavioral state machine."""

    state: BehavioralState
    command_v: float
    command_omega: float
    reason: str
    blocking_obstacle_id: Optional[int] = None
    dynamic_collision: bool = False
    dynamic_risk: str = "FREE"
    unknown_horizon: bool = False
    wait_elapsed_s: float = 0.0
    wait_timeout: bool = False
    deadlock: bool = False
    deadlock_event: str = DeadlockEvent.NONE.value
    transition: str = "NONE"
    previous_state: BehavioralState = BehavioralState.PROCEED
    state_dwell_time_s: float = 0.0
    confidence: float = 1.0
    selected_trajectory: Optional[Trajectory] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to JSON-serializable dictionary."""
        return {
            "state": self.state.value,
            "command_v": round(float(self.command_v), 4),
            "command_omega": round(float(self.command_omega), 4),
            "reason": self.reason,
            "blocking_obstacle_id": self.blocking_obstacle_id,
            "dynamic_collision": bool(self.dynamic_collision),
            "dynamic_risk": str(self.dynamic_risk),
            "unknown_horizon": bool(self.unknown_horizon),
            "wait_elapsed_s": round(float(self.wait_elapsed_s), 3),
            "wait_timeout": bool(self.wait_timeout),
            "deadlock": bool(self.deadlock),
            "deadlock_event": str(self.deadlock_event),
            "transition": self.transition,
            "previous_state": self.previous_state.value,
            "state_dwell_time_s": round(float(self.state_dwell_time_s), 3),
            "confidence": round(float(self.confidence), 3),
            "diagnostics": self.diagnostics,
        }


class BehavioralStateMachine:
    """Deterministic, hysteresis-protected tactical state machine for dynamic DWA.

    Consumes DWA candidate trajectory scores, safety guards, and dynamic predictions
    to select among PROCEED, AVOID, WAIT, and STOP behaviors while preventing
    high-frequency churn and detecting persistent deadlocks.
    """

    def __init__(self, config: Optional[BehavioralConfig] = None) -> None:
        self.config = config or BehavioralConfig()

        # State tracking
        self.current_state: BehavioralState = BehavioralState.PROCEED
        self.previous_state: BehavioralState = BehavioralState.PROCEED
        self.state_entry_time: float = 0.0
        self.wait_start_time: Optional[float] = None
        self.last_update_time: float = 0.0

        # Hysteresis & confirmation counters
        self.consecutive_recovery_count: int = 0
        self.consecutive_stop_count: int = 0
        self.wait_timeout_count: int = 0
        self.last_transition: str = "INITIALIZED"
        self.last_transition_reason: str = "System startup"

        # Deadlock monitoring: rolling window of (timestamp, x, y)
        self.pose_history: List[Tuple[float, float, float]] = []

    def reset(self, initial_state: BehavioralState = BehavioralState.PROCEED, timestamp: float = 0.0) -> None:
        """Reset internal state, counters, and history."""
        self.current_state = initial_state
        self.previous_state = initial_state
        self.state_entry_time = float(timestamp)
        self.wait_start_time = None
        self.last_update_time = float(timestamp)
        self.consecutive_recovery_count = 0
        self.consecutive_stop_count = 0
        self.wait_timeout_count = 0
        self.last_transition = "RESET"
        self.last_transition_reason = "Manual state reset"
        self.pose_history.clear()

    def _is_obstacle_clearing_corridor(
        self,
        obstacle: PredictedObstacleTrajectory,
        current_state: RobotState,
        max_horizon_s: float = 1.50,
        corridor_half_width: float = 0.55,
    ) -> Tuple[bool, Optional[float]]:
        """Determine whether an obstacle currently blocking the path will clear the corridor.

        Evaluates obstacle waypoints along the robot's forward heading.
        Returns:
            (will_clear, earliest_clearance_time)
        """
        wps = obstacle.waypoints
        if not wps:
            return False, None

        # Robot forward direction unit vector
        rx, ry, ryaw = current_state.x, current_state.y, current_state.yaw
        fx = math.cos(ryaw)
        fy = math.sin(ryaw)
        # Perpendicular normal
        nx = -fy
        ny = fx

        was_in_corridor = False
        exited_time = None

        for wp in wps:
            if wp.time_offset > max_horizon_s:
                break
            # Relative vector from robot to obstacle waypoint
            dx = wp.x - rx
            dy = wp.y - ry
            forward_dist = dx * fx + dy * fy
            lateral_dist = abs(dx * nx + dy * ny)

            # Check if within forward bounding corridor (0.2m to 3.0m ahead, |lateral| < half_width)
            in_corridor = (0.20 <= forward_dist <= 3.0) and (lateral_dist <= corridor_half_width + wp.uncertainty_radius)

            if in_corridor:
                was_in_corridor = True
                exited_time = None  # Reset if it re-enters
            elif was_in_corridor and not in_corridor:
                if exited_time is None:
                    exited_time = wp.time_offset

        if was_in_corridor and (exited_time is not None):
            return True, exited_time

        return False, None

    def _evaluate_deadlock(self, current_time: float, current_x: float, current_y: float) -> Tuple[bool, str]:
        """Check for persistent inability to progress."""
        # 1. Check consecutive stop cycles
        if self.consecutive_stop_count >= self.config.deadlock_consecutive_stop_cycles:
            return True, f"Blocked for {self.consecutive_stop_count} consecutive cycles"

        # 2. Check repeated wait timeouts
        if self.wait_timeout_count >= self.config.deadlock_max_wait_timeouts:
            return True, f"Exceeded {self.wait_timeout_count} wait timeouts"

        # 3. Check physical progress over rolling time window
        self.pose_history.append((current_time, current_x, current_y))
        cutoff = current_time - self.config.deadlock_progress_window_s
        self.pose_history = [p for p in self.pose_history if p[0] >= cutoff]

        if len(self.pose_history) >= 5 and (current_time - self.pose_history[0][0]) >= (self.config.deadlock_progress_window_s * 0.8):
            start_x, start_y = self.pose_history[0][1], self.pose_history[0][2]
            net_disp = math.hypot(current_x - start_x, current_y - start_y)
            if net_disp < self.config.deadlock_progress_threshold_m and self.current_state in (BehavioralState.STOP, BehavioralState.WAIT):
                return True, f"Insufficient progress ({net_disp:.3f}m < {self.config.deadlock_progress_threshold_m}m) over {self.config.deadlock_progress_window_s}s"

        return False, "NONE"

    def evaluate(
        self,
        current_state: RobotState,
        best_trajectory: Optional[Trajectory],
        best_v: float,
        best_w: float,
        planner_status: PlannerStatus,
        dwa_diagnostics: Dict[str, Any],
        predicted_tracks: Optional[Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]]] = None,
        global_path: Optional[Path] = None,
        current_time_s: Optional[float] = None,
    ) -> BehavioralDecision:
        """Evaluate behavioral state transitions and generate arbitrated action.

        Args:
            current_state: Current estimated robot state.
            best_trajectory: Best scored candidate trajectory from DWA (or None if no safe trajectory).
            best_v: Commanded linear velocity from DWA.
            best_w: Commanded angular velocity from DWA.
            planner_status: Status code from DWA planner.
            dwa_diagnostics: Detailed diagnostics dictionary from DWA.
            predicted_tracks: Active predicted obstacle trajectories.
            global_path: Global path reference.
            current_time_s: Timestamp in seconds (uses current_state.timestamp if None).

        Returns:
            BehavioralDecision containing state, commands, transition, and diagnostics.
        """
        now = float(current_time_s if current_time_s is not None else current_state.timestamp)
        dt_state = max(0.0, now - self.state_entry_time)

        # Extract diagnostic flags
        dyn_collision = bool(dwa_diagnostics.get("dynamic_collision", False))
        dyn_risk = str(dwa_diagnostics.get("dynamic_risk", "FREE"))
        unknown_horizon = bool(dwa_diagnostics.get("unknown_horizon", False))
        blocking_obs_id = dwa_diagnostics.get("blocking_obstacle_id", None)
        emergency_stop = bool(dwa_diagnostics.get("emergency_stop_triggered", False))
        dyn_rejected = int(dwa_diagnostics.get("dynamically_rejected_candidates", 0))
        valid_trajectories = int(dwa_diagnostics.get("valid_trajectories", 1 if best_trajectory and best_trajectory.is_valid else 0))

        # Flatten tracks sequence
        tracks: Sequence[PredictedObstacleTrajectory] = []
        if predicted_tracks is not None:
            if isinstance(predicted_tracks, PredictedTrajectoryArray):
                tracks = predicted_tracks.predictions
            else:
                tracks = list(predicted_tracks)

        # ---------------------------------------------------------------------
        # Step 1: Check Obstacle Clearance Prospects (for WAIT viability)
        # ---------------------------------------------------------------------
        clearing_obstacle = False
        earliest_clear_t: Optional[float] = None
        clearing_obs_id: Optional[int] = None

        for trk in tracks:
            if trk.current_state.status.upper() == "EXPIRED":
                continue
            is_clearing, t_exit = self._is_obstacle_clearing_corridor(
                trk, current_state,
                max_horizon_s=self.config.wait_clearance_predicted_horizon_s,
                corridor_half_width=self.config.wait_corridor_half_width_m,
            )
            if is_clearing:
                clearing_obstacle = True
                earliest_clear_t = t_exit
                clearing_obs_id = trk.track_id
                break

        # ---------------------------------------------------------------------
        # Step 2: Determine Raw Candidate Tactical Intention (Proposed State)
        # ---------------------------------------------------------------------
        # Invariant A & E: Physical collision or no safe trajectory MUST proposal STOP
        if planner_status in (PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, PlannerStatus.EMERGENCY_STOP) or emergency_stop or valid_trajectories == 0:
            if clearing_obstacle and self.current_state == BehavioralState.WAIT:
                # We are already waiting and obstacle is still predicted to clear
                proposed_state = BehavioralState.WAIT
                proposed_reason = f"Corridor blocked, waiting for obstacle {clearing_obs_id} to clear (est {earliest_clear_t:.2f}s)"
            elif clearing_obstacle and self.current_state in (BehavioralState.PROCEED, BehavioralState.AVOID):
                # Temporary moving blockage: waiting is safer than immediate emergency stop
                proposed_state = BehavioralState.WAIT
                proposed_reason = f"Obstacle {clearing_obs_id} temporarily blocking corridor, predicted to clear at t={earliest_clear_t:.2f}s"
            else:
                proposed_state = BehavioralState.STOP
                proposed_reason = "No safe collision-free trajectory exists"

        elif dyn_collision:
            # Imminent physical collision detected on selected candidate
            proposed_state = BehavioralState.STOP
            proposed_reason = f"Physical collision predicted with obstacle {blocking_obs_id}"

        elif unknown_horizon and self.config.unknown_horizon_policy == "REJECT":
            # Safety policy prohibits driving into unknown horizon
            if clearing_obstacle:
                proposed_state = BehavioralState.WAIT
                proposed_reason = f"Trajectory extends into unknown horizon; holding position for obstacle {clearing_obs_id}"
            else:
                proposed_state = BehavioralState.STOP
                proposed_reason = "Unknown prediction horizon encountered under REJECT policy"

        else:
            # A valid collision-free trajectory exists.
            # Differentiate PROCEED vs AVOID vs WAIT:
            # Check if DWA had to deviate from nominal forward motion due to dynamic obstacles
            is_deviating = (
                abs(best_w) > self.config.avoid_omega_threshold_radps
                or dyn_rejected > 0
                or dyn_risk in ("HIGH", "MEDIUM")
            )

            if not is_deviating and dyn_risk in ("FREE", "LOW") and not unknown_horizon:
                # Preferred straight trajectory is safe and nominal
                proposed_state = BehavioralState.PROCEED
                proposed_reason = "Preferred path is clear with nominal clearance and acceptable risk"
            else:
                # Safe alternative trajectory chosen, or near-miss avoidance maneuver
                if clearing_obstacle and dyn_risk == "HIGH" and self.config.prefer_wait_over_risky_avoid:
                    # Near-miss avoidance is high risk, while obstacle is moving away: safer to wait
                    proposed_state = BehavioralState.WAIT
                    proposed_reason = f"High-risk avoidance detour suppressed; waiting for obstacle {clearing_obs_id} to clear"
                else:
                    proposed_state = BehavioralState.AVOID
                    proposed_reason = f"Executing safe local avoidance maneuver (dyn_rejected={dyn_rejected}, risk={dyn_risk})"

        # ---------------------------------------------------------------------
        # Step 3: Apply State Persistence, Dwell Time, Hysteresis, and Timeouts
        # ---------------------------------------------------------------------
        prev_state = self.current_state
        next_state = prev_state
        transition_label = "HOLD"
        final_reason = proposed_reason

        # Wait timeout tracking
        wait_elapsed = 0.0
        wait_timed_out = False
        if prev_state == BehavioralState.WAIT:
            if self.wait_start_time is None:
                self.wait_start_time = now
            wait_elapsed = max(0.0, now - self.wait_start_time)
            if wait_elapsed >= self.config.wait_timeout_s:
                wait_timed_out = True

        # Rule 1: Emergency STOP transitions immediately without dwell delay (Safety Invariant)
        if proposed_state == BehavioralState.STOP:
            if prev_state != BehavioralState.STOP:
                next_state = BehavioralState.STOP
                transition_label = f"{prev_state.value} -> STOP"
                final_reason = proposed_reason
            else:
                next_state = BehavioralState.STOP
                transition_label = "HOLD"

        # Rule 2: WAIT timeout triggers STOP transition
        elif prev_state == BehavioralState.WAIT and wait_timed_out:
            next_state = BehavioralState.STOP
            transition_label = "WAIT -> STOP"
            final_reason = f"Wait timeout reached ({wait_elapsed:.2f}s >= {self.config.wait_timeout_s:.2f}s)"
            self.wait_timeout_count += 1

        # Rule 3: Dwell time protection: cannot transition between non-emergency states before min_dwell_time_s
        elif dt_state < self.config.min_dwell_time_s and proposed_state != prev_state:
            # Hold current state to prevent sub-cycle thrashing
            next_state = prev_state
            transition_label = "HOLD_DWELL"
            final_reason = f"Holding {prev_state.value} for dwell time ({dt_state:.3f}s < {self.config.min_dwell_time_s:.3f}s)"

        # Rule 4: Recovery from STOP or WAIT to PROCEED requires consecutive confirmation cycles
        elif prev_state in (BehavioralState.STOP, BehavioralState.WAIT) and proposed_state in (BehavioralState.PROCEED, BehavioralState.AVOID):
            self.consecutive_recovery_count += 1
            if self.consecutive_recovery_count >= self.config.recovery_confirmation_cycles:
                next_state = proposed_state
                transition_label = f"{prev_state.value} -> {proposed_state.value}"
                final_reason = f"Recovery confirmed over {self.consecutive_recovery_count} consecutive cycles"
                self.consecutive_recovery_count = 0
            else:
                next_state = prev_state
                transition_label = "HOLD_RECOVERY"
                final_reason = f"Confirming recovery ({self.consecutive_recovery_count}/{self.config.recovery_confirmation_cycles} cycles)"

        # Rule 5: Normal state transition
        elif proposed_state != prev_state:
            next_state = proposed_state
            transition_label = f"{prev_state.value} -> {proposed_state.value}"
            final_reason = proposed_reason
            self.consecutive_recovery_count = 0

        else:
            # Stable state hold
            next_state = prev_state
            transition_label = "HOLD"

        # ---------------------------------------------------------------------
        # Step 4: State Entry / Exit Bookkeeping
        # ---------------------------------------------------------------------
        if next_state != prev_state:
            self.previous_state = prev_state
            self.current_state = next_state
            self.state_entry_time = now
            self.last_transition = transition_label
            self.last_transition_reason = final_reason
            dt_state = 0.0

            if next_state == BehavioralState.WAIT:
                self.wait_start_time = now
                wait_elapsed = 0.0
            else:
                self.wait_start_time = None
        else:
            self.current_state = next_state

        if next_state == BehavioralState.STOP:
            self.consecutive_stop_count += 1
        else:
            self.consecutive_stop_count = 0

        self.last_update_time = now

        # ---------------------------------------------------------------------
        # Step 5: Deadlock Detection
        # ---------------------------------------------------------------------
        is_deadlock, deadlock_reason = self._evaluate_deadlock(now, current_state.x, current_state.y)
        deadlock_event_str = DeadlockEvent.DYNAMIC_DEADLOCK.value if is_deadlock else DeadlockEvent.NONE.value

        # ---------------------------------------------------------------------
        # Step 6: Command Arbitration (Safety Invariants B & C)
        # ---------------------------------------------------------------------
        # STOP and WAIT strictly command v=0, w=0
        if next_state in (BehavioralState.STOP, BehavioralState.WAIT):
            cmd_v = 0.0
            cmd_omega = 0.0
        else:
            # PROCEED and AVOID command the chosen DWA velocity
            cmd_v = float(best_v)
            cmd_omega = float(best_w)

        # Confidence metric
        confidence = 1.0
        if dyn_risk == "HIGH":
            confidence = 0.50
        elif dyn_risk == "MEDIUM":
            confidence = 0.75
        elif unknown_horizon:
            confidence = 0.40

        # Resolve obstacle ID: direct blocker, clearing obstacle, or primary track being avoided
        resolved_obs_id = blocking_obs_id or clearing_obs_id
        if resolved_obs_id is None and tracks and next_state in (BehavioralState.AVOID, BehavioralState.WAIT, BehavioralState.STOP):
            resolved_obs_id = tracks[0].track_id

        return BehavioralDecision(
            state=next_state,
            command_v=cmd_v,
            command_omega=cmd_omega,
            reason=final_reason,
            blocking_obstacle_id=resolved_obs_id,
            dynamic_collision=dyn_collision,
            dynamic_risk=dyn_risk,
            unknown_horizon=unknown_horizon,
            wait_elapsed_s=wait_elapsed,
            wait_timeout=wait_timed_out,
            deadlock=is_deadlock,
            deadlock_event=deadlock_event_str,
            transition=transition_label,
            previous_state=prev_state,
            state_dwell_time_s=dt_state,
            confidence=confidence,
            selected_trajectory=best_trajectory if next_state in (BehavioralState.PROCEED, BehavioralState.AVOID) else None,
            diagnostics={
                "proposed_state": proposed_state.value,
                "proposed_reason": proposed_reason,
                "deadlock_reason": deadlock_reason,
                "clearing_obstacle": clearing_obstacle,
                "earliest_clear_t": earliest_clear_t,
                "consecutive_recovery_count": self.consecutive_recovery_count,
                "consecutive_stop_count": self.consecutive_stop_count,
                "wait_timeout_count": self.wait_timeout_count,
                "raw_dwa_v": best_v,
                "raw_dwa_omega": best_w,
            },
        )
