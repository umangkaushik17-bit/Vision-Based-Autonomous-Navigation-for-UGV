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
    TURNING = "TURNING"
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
    enable_reverse_turn: bool = True           # Apply angular escape steering during reverse
    reverse_turn_rate_radps: float = 0.35      # Angular velocity for reverse-and-turn disengagement
    enable_turn_in_place_phase: bool = False   # Optional explicit post-reverse turn-in-place state
    target_turn_angle_rad: float = 0.45        # Target heading change during TURNING phase
    max_turn_duration_s: float = 2.50          # Bounded timeout for TURNING phase


class RecoveryController:
    """Orchestrates controlled reverse recovery when deadlocked by an obstacle."""

    def __init__(self, config: Optional[RecoveryConfig] = None) -> None:
        self.config = config or RecoveryConfig()
        self.state = RecoveryState.INACTIVE

        # Tracking variables during recovery
        self._reverse_start_time_s: float = 0.0
        self._reverse_start_x: float = 0.0
        self._reverse_start_y: float = 0.0
        self._reverse_start_yaw: float = 0.0
        self._turn_start_time_s: float = 0.0
        self._turn_start_yaw: float = 0.0
        self._escape_turn_dir: float = 0.0
        self._settle_start_time_s: float = 0.0
        self._consecutive_stop_ticks: int = 0
        self._last_state_transition_time_s: float = 0.0

        # Safety Governor for continuous reverse monitoring
        self.safety_governor = SafetyGovernor()

    def compute_escape_turn_direction(
        self,
        robot_pose: Tuple[float, float, float],
        hazards: Sequence[Union[TrackedObstacleState, LatchedHazard]],
        costmap: Any = None,
    ) -> float:
        """Determine the freer lateral direction (+1.0 for LEFT, -1.0 for RIGHT, 0.0 if symmetric)
        to steer the vehicle heading away from forward/lateral obstructions during recovery.
        """
        rx, ry, ryaw = robot_pose
        cos_yaw = math.cos(ryaw)
        sin_yaw = math.sin(ryaw)

        min_left_clr = 5.0
        min_right_clr = 5.0

        # 1. Inspect forward/lateral hazards in robot frame
        for h in hazards:
            dx = float(h.x) - rx
            dy = float(h.y) - ry
            x_rel = dx * cos_yaw + dy * sin_yaw
            y_rel = -dx * sin_yaw + dy * cos_yaw
            if -0.20 <= x_rel <= 2.50 and abs(y_rel) <= 2.00:
                d = math.hypot(x_rel, y_rel)
                if y_rel >= 0.0:
                    min_left_clr = min(min_left_clr, d)
                else:
                    min_right_clr = min(min_right_clr, d)

        # 2. Sample static costmap left vs right flanks ahead of robot
        if costmap is not None and hasattr(costmap, "world_to_grid"):
            for x_rel in (0.40, 0.80, 1.20, 1.60):
                for y_mag in (0.35, 0.70, 1.05):
                    # Left sample (+y_mag)
                    lx = rx + x_rel * cos_yaw - y_mag * sin_yaw
                    ly = ry + x_rel * sin_yaw + y_mag * cos_yaw
                    if not hasattr(costmap, "is_in_bounds") or costmap.is_in_bounds(lx, ly):
                        lr, lc = costmap.world_to_grid(lx, ly)
                        if 0 <= lr < costmap.shape[0] and 0 <= lc < costmap.shape[1]:
                            if (hasattr(costmap, "obstacle") and costmap.obstacle[lr, lc]) or (
                                hasattr(costmap, "cost") and costmap.cost[lr, lc] >= self.config.lethal_cost_threshold
                            ):
                                min_left_clr = min(min_left_clr, math.hypot(x_rel, y_mag))
                    # Right sample (-y_mag)
                    rx_s = rx + x_rel * cos_yaw - (-y_mag) * sin_yaw
                    ry_s = ry + x_rel * sin_yaw + (-y_mag) * cos_yaw
                    if not hasattr(costmap, "is_in_bounds") or costmap.is_in_bounds(rx_s, ry_s):
                        rr, rc = costmap.world_to_grid(rx_s, ry_s)
                        if 0 <= rr < costmap.shape[0] and 0 <= rc < costmap.shape[1]:
                            if (hasattr(costmap, "obstacle") and costmap.obstacle[rr, rc]) or (
                                hasattr(costmap, "cost") and costmap.cost[rr, rc] >= self.config.lethal_cost_threshold
                            ):
                                min_right_clr = min(min_right_clr, math.hypot(x_rel, y_mag))

        if abs(min_left_clr - min_right_clr) < 0.05:
            return 0.0
        # Steer heading toward the side with strictly larger clearance
        return 1.0 if min_left_clr > min_right_clr else -1.0

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
            self._reverse_start_yaw = ryaw
            self._escape_turn_dir = self.compute_escape_turn_direction(robot_pose, hazards, costmap)
            self._last_state_transition_time_s = now
            rev_w = (self._escape_turn_dir * self.config.reverse_turn_rate_radps) if self.config.enable_reverse_turn else 0.0
            return self.config.reverse_velocity_mps, rev_w, self.state, "Starting controlled reverse recovery"

        # ---------------------------------------------------------------------
        # State: REVERSING
        # ---------------------------------------------------------------------
        if self.state == RecoveryState.REVERSING:
            rev_w = (self._escape_turn_dir * self.config.reverse_turn_rate_radps) if self.config.enable_reverse_turn else 0.0
            # 1. Continuous Safety Governor Inspection: check if rear became blocked
            gov_v, gov_w, gov_dec, _ = self.safety_governor.evaluate(
                requested_v=self.config.reverse_velocity_mps,
                requested_w=rev_w,
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

            if disp >= self.config.target_reverse_distance_m or dur >= self.config.max_reverse_duration_s:
                reason_prefix = (
                    f"Reverse complete (disp={disp:.2f}m)"
                    if disp >= self.config.target_reverse_distance_m
                    else f"Reverse duration timeout ({dur:.2f}s)"
                )
                if self.config.enable_turn_in_place_phase and self.config.target_turn_angle_rad > 0.0:
                    self.state = RecoveryState.TURNING
                    self._turn_start_time_s = now
                    self._turn_start_yaw = ryaw
                    if self._escape_turn_dir == 0.0:
                        self._escape_turn_dir = 1.0
                    self._last_state_transition_time_s = now
                    turn_w = self._escape_turn_dir * max(0.25, self.config.reverse_turn_rate_radps)
                    return 0.0, turn_w, self.state, f"{reason_prefix}; executing escape turn"

                self.state = RecoveryState.SETTLING
                self._settle_start_time_s = now
                self._last_state_transition_time_s = now
                return 0.0, 0.0, self.state, f"{reason_prefix}; settling"

            # Continue controlled reverse (with optional turn steering)
            return self.config.reverse_velocity_mps, rev_w, self.state, f"Reversing (disp={disp:.2f}m/{self.config.target_reverse_distance_m:.2f}m)"

        # ---------------------------------------------------------------------
        # State: TURNING (Post-Reverse Angular Escape)
        # ---------------------------------------------------------------------
        if self.state == RecoveryState.TURNING:
            yaw_diff = abs(math.atan2(math.sin(ryaw - self._turn_start_yaw), math.cos(ryaw - self._turn_start_yaw)))
            dur = now - self._turn_start_time_s
            if yaw_diff >= self.config.target_turn_angle_rad or dur >= self.config.max_turn_duration_s:
                self.state = RecoveryState.SETTLING
                self._settle_start_time_s = now
                self._last_state_transition_time_s = now
                return 0.0, 0.0, self.state, f"Turn escape complete (dyaw={yaw_diff:.2f}rad); settling"
            turn_w = self._escape_turn_dir * max(0.25, self.config.reverse_turn_rate_radps)
            return 0.0, turn_w, self.state, f"Turning to clear trap (dyaw={yaw_diff:.2f}/{self.config.target_turn_angle_rad:.2f}rad)"

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
        self._reverse_start_yaw = 0.0
        self._turn_start_time_s = 0.0
        self._turn_start_yaw = 0.0
        self._escape_turn_dir = 0.0
        self._settle_start_time_s = 0.0
        self._consecutive_stop_ticks = 0
