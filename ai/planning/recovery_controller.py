"""Deterministic Safe Reverse Recovery Controller & Deadlock Replanning (Phase 13.24).

When an unyielding obstacle blocks the UGV's forward path, prevents intentional
pushing or permanent paralysis by:
1. Identifying genuine deadlock (multi-signal: blocked corridor, no progress, repeated stop).
2. Verifying rear safety clearance across static costmap and dynamic hazards.
3. Executing a bounded, speed-governed reverse maneuver (-0.15 m/s, ~0.45 m distance).
4. Continuous safety monitoring: if rear hazard appears during reverse, stops immediately.
5. Settling and triggering global replanning from the cleared pose.
6. Ensuring invalidated old paths are marked stale and never blindly resumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from ai.planning.collision import FootprintCollisionChecker
from ai.planning.dynamic.data_model import TrackedObstacleState
from ai.planning.safety_governor import LatchedHazard, SafetyGovernor


class RecoveryState(str, Enum):
    """Lifecycle states of the reverse recovery procedure."""
    INACTIVE = "INACTIVE"
    ASSESSING_REAR = "ASSESSING_REAR"
    REVERSING = "REVERSING"
    SETTLING = "SETTLING"
    REPLAN_READY = "REPLAN_READY"
    ABORTED_REAR_BLOCKED = "ABORTED_REAR_BLOCKED"
    FAILED_TIMEOUT = "FAILED_TIMEOUT"


@dataclass
class RecoveryConfig:
    """Tunable thresholds for safe reverse recovery and replan disengagement."""
    reverse_velocity_mps: float = -0.15        # Low-speed reverse command
    target_reverse_distance_m: float = 0.45    # Distance to disengage bumper from obstacle
    max_reverse_duration_s: float = 3.50       # Bounded duration timeout
    settle_duration_s: float = 0.25            # Stationary pause before replanning
    min_rear_clearance_m: float = 0.50         # Required rear clearance to initiate reverse
    rear_corridor_half_width_m: float = 0.35   # Half width for rear clearance check
    vehicle_length_m: float = 0.65             # Vehicle length
    vehicle_width_m: float = 0.45              # Vehicle width
    lethal_cost_threshold: float = 0.85        # Static costmap lethal threshold
    deadlock_stop_cycles: int = 4              # Consecutive stop ticks before recovery triggered


class RecoveryController:
    """Orchestrates controlled reverse recovery when deadlocked by an obstacle."""

    def __init__(self, config: Optional[RecoveryConfig] = None) -> None:
        self.config = config or RecoveryConfig()
        self.state = RecoveryState.INACTIVE

        # Tracking variables during recovery
        self._reverse_start_time_s: float = 0.0
        self._reverse_start_x: float = 0.0
        self._reverse_start_y: float = 0.0
        self._settle_start_time_s: float = 0.0
        self._consecutive_stop_ticks: int = 0
        self._last_state_transition_time_s: float = 0.0

        # Safety Governor for continuous reverse monitoring
        self.safety_governor = SafetyGovernor()

    def check_rear_clearance(
        self,
        robot_pose: Tuple[float, float, float],
        hazards: Sequence[Union[TrackedObstacleState, LatchedHazard]],
        costmap: Any = None,
    ) -> Tuple[bool, float, str]:
        """Verify rear clearance across both dynamic hazards and static costmap.

        Returns:
            Tuple of (is_clear, min_clearance_m, reason)
        """
        rx, ry, ryaw = robot_pose
        cos_yaw = math.cos(ryaw)
        sin_yaw = math.sin(ryaw)

        rear_req = self.config.min_rear_clearance_m
        min_dist_found = 999.0
        blocking_source = "None"

        # 1. Check dynamic / latched hazards behind robot
        for h in hazards:
            hx = float(h.x)
            hy = float(h.y)
            dx = hx - rx
            dy = hy - ry

            # Transform into robot frame (x_rel < 0 is rear)
            x_rel = dx * cos_yaw + dy * sin_yaw
            y_rel = -dx * sin_yaw + dy * cos_yaw

            obs_radius = 0.40
            if hasattr(h, "bbox_size") and len(h.bbox_size) >= 2:
                obs_radius = max(float(h.bbox_size[0]), float(h.bbox_size[1])) / 2.0

            # If behind the rear bumper
            if x_rel < -0.15 and (abs(y_rel) - obs_radius) <= self.config.rear_corridor_half_width_m:
                eff_dist = max(0.0, abs(x_rel) - (self.config.vehicle_length_m / 2.0) - obs_radius)
                if eff_dist < min_dist_found:
                    min_dist_found = eff_dist
                    blocking_source = f"Hazard {getattr(h, 'hazard_id', getattr(h, 'track_id', 'unknown'))}"

        # 2. Check static costmap along rearward sweep
        if costmap is not None:
            # Sample discrete test points backwards from rear bumper
            half_len = self.config.vehicle_length_m / 2.0
            for d in np.linspace(0.10, rear_req + 0.10, 6):
                test_x = rx - (half_len + d) * cos_yaw
                test_y = ry - (half_len + d) * sin_yaw

                # Check if in bounds and free
                if hasattr(costmap, "is_in_bounds") and not costmap.is_in_bounds(test_x, test_y):
                    min_dist_found = min(min_dist_found, d)
                    blocking_source = "Map Boundary"
                    break

                if hasattr(costmap, "world_to_grid"):
                    r, c = costmap.world_to_grid(test_x, test_y)
                    if 0 <= r < costmap.shape[0] and 0 <= c < costmap.shape[1]:
                        if hasattr(costmap, "obstacle") and costmap.obstacle[r, c]:
                            min_dist_found = min(min_dist_found, d)
                            blocking_source = "Static Obstacle"
                            break
                        if hasattr(costmap, "cost") and costmap.cost[r, c] >= self.config.lethal_cost_threshold:
                            min_dist_found = min(min_dist_found, d)
                            blocking_source = "Lethal Cost"
                            break

        if min_dist_found < rear_req:
            return False, min_dist_found, f"Rear blocked ({blocking_source}, dist={min_dist_found:.2f}m < {rear_req:.2f}m)"

        return True, min_dist_found, "Rear corridor verified clear"

    def step(
        self,
        current_time_s: float,
        robot_pose: Tuple[float, float, float],
        is_deadlocked: bool,
        hazards: Sequence[Union[TrackedObstacleState, LatchedHazard]],
        costmap: Any = None,
    ) -> Tuple[Optional[float], Optional[float], RecoveryState, str]:
        """Advance the recovery state machine.

        Returns:
            Tuple of (cmd_v, cmd_w, recovery_state, reason)
            If recovery is INACTIVE, returns (None, None, INACTIVE, reason).
        """
        now = float(current_time_s)
        rx, ry, ryaw = robot_pose

        # ---------------------------------------------------------------------
        # State: INACTIVE
        # ---------------------------------------------------------------------
        if self.state in (RecoveryState.INACTIVE, RecoveryState.ABORTED_REAR_BLOCKED, RecoveryState.FAILED_TIMEOUT):
            if is_deadlocked:
                self._consecutive_stop_ticks += 1
                if self._consecutive_stop_ticks >= self.config.deadlock_stop_cycles:
                    # Initiate clearance assessment
                    self.state = RecoveryState.ASSESSING_REAR
                    self._last_state_transition_time_s = now
            else:
                self._consecutive_stop_ticks = 0
                self.state = RecoveryState.INACTIVE
                return None, None, RecoveryState.INACTIVE, "Nominal operation"

        # ---------------------------------------------------------------------
        # State: ASSESSING_REAR
        # ---------------------------------------------------------------------
        if self.state == RecoveryState.ASSESSING_REAR:
            is_clear, clr_m, reason = self.check_rear_clearance(robot_pose, hazards, costmap)
            if not is_clear:
                self.state = RecoveryState.ABORTED_REAR_BLOCKED
                return 0.0, 0.0, self.state, f"Reverse aborted: {reason}"

            # Clear to reverse: transition to REVERSING
            self.state = RecoveryState.REVERSING
            self._reverse_start_time_s = now
            self._reverse_start_x = rx
            self._reverse_start_y = ry
            self._last_state_transition_time_s = now
            return self.config.reverse_velocity_mps, 0.0, self.state, "Starting controlled reverse recovery"

        # ---------------------------------------------------------------------
        # State: REVERSING
        # ---------------------------------------------------------------------
        if self.state == RecoveryState.REVERSING:
            # 1. Continuous Safety Governor Inspection: check if rear became blocked
            gov_v, gov_w, gov_dec, _ = self.safety_governor.evaluate(
                requested_v=self.config.reverse_velocity_mps,
                requested_w=0.0,
                robot_pose=robot_pose,
                hazards=hazards,
                current_time_s=now,
            )
            if "STOP" in gov_dec or gov_v == 0.0:
                self.state = RecoveryState.ABORTED_REAR_BLOCKED
                return 0.0, 0.0, self.state, f"Reverse halted by safety governor: {gov_dec}"

            # 2. Check displacement
            disp = math.hypot(rx - self._reverse_start_x, ry - self._reverse_start_y)
            dur = now - self._reverse_start_time_s

            if disp >= self.config.target_reverse_distance_m:
                # Target reverse distance achieved
                self.state = RecoveryState.SETTLING
                self._settle_start_time_s = now
                self._last_state_transition_time_s = now
                return 0.0, 0.0, self.state, f"Reverse complete (disp={disp:.2f}m); settling"

            if dur >= self.config.max_reverse_duration_s:
                # Timeout
                self.state = RecoveryState.SETTLING
                self._settle_start_time_s = now
                self._last_state_transition_time_s = now
                return 0.0, 0.0, self.state, f"Reverse duration timeout ({dur:.2f}s); settling"

            # Continue controlled reverse
            return self.config.reverse_velocity_mps, 0.0, self.state, f"Reversing (disp={disp:.2f}m/{self.config.target_reverse_distance_m:.2f}m)"

        # ---------------------------------------------------------------------
        # State: SETTLING
        # ---------------------------------------------------------------------
        if self.state == RecoveryState.SETTLING:
            dur = now - self._settle_start_time_s
            if dur >= self.config.settle_duration_s:
                self.state = RecoveryState.REPLAN_READY
                self._last_state_transition_time_s = now
                return 0.0, 0.0, self.state, "Settling complete; triggering global replan"
            return 0.0, 0.0, self.state, f"Settling at rest ({dur:.2f}s)"

        # ---------------------------------------------------------------------
        # State: REPLAN_READY
        # ---------------------------------------------------------------------
        if self.state == RecoveryState.REPLAN_READY:
            # Replan state is consumed by external caller
            return 0.0, 0.0, self.state, "Global replanning required"

        return None, None, self.state, "Unknown state"

    def mark_replan_completed(self) -> None:
        """Invoked after global replanner generates new path; resets state machine."""
        self.state = RecoveryState.INACTIVE
        self._consecutive_stop_ticks = 0

    def reset(self) -> None:
        """Full reset of recovery controller."""
        self.state = RecoveryState.INACTIVE
        self._reverse_start_time_s = 0.0
        self._reverse_start_x = 0.0
        self._reverse_start_y = 0.0
        self._settle_start_time_s = 0.0
        self._consecutive_stop_ticks = 0
