"""Recovery orchestrator wrapping Phase 13 RecoveryController with bounded mission retry policy."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ai.planning.dynamic.data_model import TrackedObstacleState
from ai.planning.recovery_controller import (
    RecoveryConfig,
    RecoveryController,
    RecoveryState,
)
from ai.planning.safety_governor import LatchedHazard

logger = logging.getLogger("ugv_mission.recovery")


@dataclass
class RecoveryOrchestratorConfig:
    """Configuration for mission-level recovery retries and thresholds."""

    max_recovery_attempts: int = 3
    reset_attempts_on_waypoint_advance: bool = True
    recovery_config: RecoveryConfig = field(default_factory=RecoveryConfig)


class MissionRecoveryOrchestrator:
    """Orchestrates lower-level RecoveryController execution with mission retry counting.
    
    Strict Invariants:
    1. Reuses existing Phase 13.24 RecoveryController without modifying recovery mechanics.
    2. Rear clearance validation is non-negotiable.
    3. Recovery never advances the active waypoint or marks it reached.
    4. Successful recovery triggers replanning toward the SAME active waypoint.
    5. Retry exhaustion strictly triggers mission abort.
    """

    def __init__(self, config: Optional[RecoveryOrchestratorConfig] = None) -> None:
        self.config = config or RecoveryOrchestratorConfig()
        self.controller = RecoveryController(config=self.config.recovery_config)

        # Retry tracking state
        self._current_waypoint_id: Optional[str] = None
        self._attempt_count: int = 0
        self._total_recovery_count: int = 0
        self._last_result_summary: str = "INACTIVE"
        self._in_recovery_cycle: bool = False

    @property
    def attempt_count(self) -> int:
        """Number of recovery attempts made on the current active waypoint."""
        return self._attempt_count

    @property
    def total_recovery_count(self) -> int:
        """Lifetime recovery attempts across entire mission."""
        return self._total_recovery_count

    @property
    def last_result(self) -> str:
        """Human-readable string of the last recovery step result."""
        return self._last_result_summary

    @property
    def recovery_state(self) -> RecoveryState:
        """Current internal state of the RecoveryController."""
        return self.controller.state

    def on_waypoint_changed(self, new_waypoint_id: Optional[str]) -> None:
        """Notify orchestrator that active waypoint has changed. Resets per-waypoint retry counter."""
        if new_waypoint_id != self._current_waypoint_id:
            logger.info(
                f"[RECOVERY] Active waypoint changed from {self._current_waypoint_id} to {new_waypoint_id}. "
                f"Resetting per-waypoint recovery attempt counter."
            )
            self._current_waypoint_id = new_waypoint_id
            if self.config.reset_attempts_on_waypoint_advance:
                self._attempt_count = 0
            self.controller.reset()
            self._in_recovery_cycle = False

    def step(
        self,
        current_time_s: float,
        robot_pose: Tuple[float, float, float],
        is_deadlocked: bool,
        hazards: Sequence[Union[TrackedObstacleState, LatchedHazard]],
        costmap: Any = None,
    ) -> Tuple[Optional[float], Optional[float], RecoveryState, bool, str]:
        """Execute one step of the recovery orchestrator.
        
        Returns:
            Tuple of:
            - cmd_v: Optional linear velocity command
            - cmd_w: Optional angular velocity command
            - state: Current RecoveryState
            - retry_limit_exceeded: True if max recovery retries have been exhausted
            - reason: Diagnostic text
        """
        # Detect start of a new recovery engagement
        if is_deadlocked and not self._in_recovery_cycle:
            self._attempt_count += 1
            self._total_recovery_count += 1
            self._in_recovery_cycle = True
            logger.warning(
                f"[RECOVERY] Initiating recovery attempt {self._attempt_count}/{self.config.max_recovery_attempts} "
                f"at waypoint {self._current_waypoint_id}."
            )

            # Check if this new attempt already exceeds the bounded retry limit
            if self._attempt_count > self.config.max_recovery_attempts:
                reason = (
                    f"Recovery retry limit exceeded ({self._attempt_count - 1} "
                    f">= {self.config.max_recovery_attempts}) at waypoint {self._current_waypoint_id}"
                )
                self._last_result_summary = "RETRY_LIMIT_EXCEEDED"
                logger.error(f"[RECOVERY] {reason}")
                return 0.0, 0.0, RecoveryState.ABORTED_REAR_BLOCKED, True, reason

        # Delegate execution directly to the frozen Phase 13.24 RecoveryController
        cmd_v, cmd_w, rec_state, reason = self.controller.step(
            current_time_s=current_time_s,
            robot_pose=robot_pose,
            is_deadlocked=is_deadlocked,
            hazards=hazards,
            costmap=costmap,
        )

        self._last_result_summary = f"{rec_state.value}: {reason}"

        # Handle termination states of the recovery procedure
        retry_exceeded = False
        if rec_state in (RecoveryState.ABORTED_REAR_BLOCKED, RecoveryState.FAILED_TIMEOUT):
            self._in_recovery_cycle = False
            if self._attempt_count >= self.config.max_recovery_attempts:
                retry_exceeded = True
                reason = f"Max recovery attempts reached ({self._attempt_count}/{self.config.max_recovery_attempts}): {reason}"
                logger.error(f"[RECOVERY] {reason}")

        elif rec_state == RecoveryState.REPLAN_READY:
            # Replan is ready. Recovery maneuver succeeded.
            # Once replanning is completed, mark_replan_completed() will be called.
            self._in_recovery_cycle = False

        elif rec_state == RecoveryState.INACTIVE and not is_deadlocked:
            self._in_recovery_cycle = False

        return cmd_v, cmd_w, rec_state, retry_exceeded, reason

    def mark_replan_completed(self) -> None:
        """Notify controller that replanning succeeded and normal navigation is resuming."""
        self.controller.mark_replan_completed()
        self._in_recovery_cycle = False
        self._last_result_summary = "REPLAN_COMPLETED"
        logger.info(
            f"[RECOVERY] Replan completed for waypoint {self._current_waypoint_id}. Resuming navigation."
        )

    def reset(self) -> None:
        """Full reset of recovery orchestrator state."""
        self.controller.reset()
        self._attempt_count = 0
        self._total_recovery_count = 0
        self._in_recovery_cycle = False
        self._last_result_summary = "INACTIVE"
