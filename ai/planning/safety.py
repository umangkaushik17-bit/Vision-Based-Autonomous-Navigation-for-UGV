"""Safety gating, command arbitration, watchdog monitoring, and emergency stop controller for Phase 12 UGV navigation."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from ai.planning.base import PlanningConfig, PlannerStatus, RobotState, VehicleKinematics
from ai.planning.collision import FootprintCollisionChecker


class SafetyController:
    """Multi-tiered safety supervisor and velocity command arbiter.

    Arbitration Priority:
      Emergency Stop > Safety Clamp > DWA Command
    """

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        self.config = config or PlanningConfig()
        self.vehicle: VehicleKinematics = self.config.vehicle
        self.collision_checker = FootprintCollisionChecker(self.vehicle)

        safety_cfg = getattr(self.config, "safety", {})
        self.max_map_age_s = float(safety_cfg.get("max_map_age_s", 2.00))
        self.max_loc_age_s = float(safety_cfg.get("max_localization_age_s", 1.00))
        self.cmd_timeout_s = float(safety_cfg.get("command_timeout_s", 0.25))
        self.stop_margin_m = float(getattr(self.vehicle, "stop_margin_m", 0.30))
        self.lethal_cost_threshold = float(getattr(self.config, "tolerances", {}).get("lethal_cost_threshold", 0.85))

        # Dynamic limits
        self.max_v = float(self.vehicle.max_linear_velocity_mps)
        self.min_v = float(self.vehicle.min_linear_velocity_mps)
        self.max_w = float(self.vehicle.max_angular_velocity_radps)
        self.max_accel_v = float(self.vehicle.max_linear_acceleration_mps2)
        self.max_accel_w = float(self.vehicle.max_angular_acceleration_radps2)
        self.degraded_max_v = float(getattr(self.vehicle, "degraded_max_linear_velocity_mps", 0.35))

        # State tracking
        self.last_valid_command_time: float = 0.0
        self.last_v: float = 0.0
        self.last_w: float = 0.0
        self.prev_status: PlannerStatus = PlannerStatus.IDLE

    def check_current_occupancy(
        self,
        current_state: RobotState,
        terrain_map: Any
    ) -> Tuple[bool, PlannerStatus, str]:
        """Verify the robot is not currently inside an obstacle or critical hazard."""
        if terrain_map is None:
            return False, PlannerStatus.MAP_UNAVAILABLE, "Terrain map is None"

        if not terrain_map.is_in_bounds(current_state.x, current_state.y):
            return False, PlannerStatus.RECOVERY_REQUIRED, "Robot current center is outside map extent"

        # Check oriented footprint at current position (Section 36: obstacle, critical hazard, map bounds)
        if self.collision_checker.is_footprint_collision(
            current_state.x, current_state.y, current_state.yaw, terrain_map,
            unknown_is_lethal=False,
            max_cost=self.lethal_cost_threshold,
        ):
            return False, PlannerStatus.RECOVERY_REQUIRED, "Robot footprint is intersecting lethal obstacle or hazard"

        return True, PlannerStatus.SUCCESS, "Current pose footprint is clear"

    def evaluate_safety_guards(
        self,
        current_time_s: float,
        current_state: RobotState,
        terrain_map: Any,
        tracking_state: Union[int, str] = 0,  # 0=TRACKING_OK, 1=DEGRADED, 2=LOST, 3=RELOCALIZING, 4=RECOVERED
        map_timestamp_s: Optional[float] = None,
        loc_timestamp_s: Optional[float] = None,
        tf_available: bool = True
    ) -> Tuple[bool, PlannerStatus, str, bool]:
        """Evaluate pre-planning safety triggers.

        Returns:
            (is_safe, status, reason, is_degraded)
        """
        # 1. Localization Tracking State Guard (Phase 8/10/12 Contract)
        state_str = str(tracking_state).upper()
        if state_str in ("2", "LOST", "TRACKINGSTATE.LOST"):
            return False, PlannerStatus.EMERGENCY_STOP, "Localization tracking state is LOST", False

        if state_str in ("3", "RELOCALIZING", "TRACKINGSTATE.RELOCALIZING"):
            return False, PlannerStatus.EMERGENCY_STOP, "Localization is currently RELOCALIZING (holding stationary)", False

        is_degraded = state_str in ("1", "DEGRADED", "TRACKINGSTATE.DEGRADED")

        # 2. TF Link Guard
        if not tf_available:
            return False, PlannerStatus.EMERGENCY_STOP, "TF coordinate transform tree unavailable", is_degraded

        # 3. Map Availability Guard
        if terrain_map is None:
            return False, PlannerStatus.MAP_UNAVAILABLE, "WorldTerrainMap is not available", is_degraded

        # 4. Current Footprint Occupancy Guard (Section 36)
        is_clear, occ_status, occ_reason = self.check_current_occupancy(current_state, terrain_map)
        if not is_clear:
            return False, occ_status, occ_reason, is_degraded

        # 5. Map Staleness Guard (Section 34)
        map_time = map_timestamp_s if map_timestamp_s is not None else getattr(terrain_map, "created_timestamp", None)
        if map_time is not None and map_time > 0.0:
            map_age = current_time_s - map_time
            if map_age > self.max_map_age_s:
                return False, PlannerStatus.MAP_STALE, f"WorldTerrainMap is stale: age {map_age:.3f}s > max {self.max_map_age_s:.3f}s", is_degraded

        # 6. Localization Staleness Guard (Section 35)
        loc_time = loc_timestamp_s if loc_timestamp_s is not None else (current_state.timestamp if current_state.timestamp > 0.0 else None)
        if loc_time is not None:
            loc_age = current_time_s - loc_time
            if loc_age > self.max_loc_age_s:
                return False, PlannerStatus.LOCALIZATION_INVALID, f"Localization estimate is stale: age {loc_age:.3f}s > max {self.max_loc_age_s:.3f}s", is_degraded

        return True, PlannerStatus.SUCCESS, "All safety guards nominal", is_degraded


    def arbitrate_command(
        self,
        requested_v: float,
        requested_w: float,
        dt_s: float,
        current_time_s: float,
        planner_status: PlannerStatus,
        is_degraded: bool = False,
        is_emergency: bool = False,
    ) -> Tuple[float, float, PlannerStatus, Dict[str, Any]]:
        """Arbitrate, clamp, and watchdog-monitor the final command velocity.

        Enforces:
          Emergency Stop > Safety Clamp > DWA Command
        """
        diagnostics: Dict[str, Any] = {
            "requested_v": round(float(requested_v), 3),
            "requested_w": round(float(requested_w), 3),
            "emergency_override": False,
            "watchdog_override": False,
            "clamped": False,
            "is_degraded": is_degraded,
        }

        # 1. Top Priority: Emergency Stop Triggered
        if is_emergency or planner_status in (
            PlannerStatus.EMERGENCY_STOP,
            PlannerStatus.RECOVERY_REQUIRED,
            PlannerStatus.MAP_STALE,
            PlannerStatus.MAP_UNAVAILABLE,
            PlannerStatus.LOCALIZATION_INVALID,
            PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
            PlannerStatus.NO_SAFE_TRAJECTORY,
            PlannerStatus.NO_PATH,
            PlannerStatus.GOAL_REACHED,
        ):
            self.last_v = 0.0
            self.last_w = 0.0
            self.last_valid_command_time = current_time_s
            self.prev_status = planner_status
            diagnostics["emergency_override"] = True
            return 0.0, 0.0, planner_status, diagnostics

        # 2. Check Command Watchdog (Section 40)
        if self.last_valid_command_time > 0.0:
            time_since_last_cmd = current_time_s - self.last_valid_command_time
            if time_since_last_cmd > self.cmd_timeout_s:
                self.last_v = 0.0
                self.last_w = 0.0
                diagnostics["watchdog_override"] = True
                return 0.0, 0.0, PlannerStatus.EMERGENCY_STOP, diagnostics

        # 3. Sanity Audit: Check Finite / Non-NaN (Section 86)
        if not (math.isfinite(requested_v) and math.isfinite(requested_w)):
            self.last_v = 0.0
            self.last_w = 0.0
            return 0.0, 0.0, PlannerStatus.EMERGENCY_STOP, {**diagnostics, "error": "NaN/Inf command rejected"}

        # 4. Apply Dynamic Velocity & Acceleration Safety Clamps (Section 39)
        eff_max_v = self.degraded_max_v if is_degraded else self.max_v
        v_target = float(np.clip(requested_v, self.min_v, eff_max_v))
        w_target = float(np.clip(requested_w, -self.max_w, self.max_w))

        dt = max(1e-3, dt_s)
        max_dv = self.max_accel_v * dt
        max_dw = self.max_accel_w * dt

        clamped_v = float(np.clip(v_target, self.last_v - max_dv, self.last_v + max_dv))
        clamped_w = float(np.clip(w_target, self.last_w - max_dw, self.last_w + max_dw))

        if abs(clamped_v - requested_v) > 1e-3 or abs(clamped_w - requested_w) > 1e-3:
            diagnostics["clamped"] = True

        self.last_v = clamped_v
        self.last_w = clamped_w
        self.last_valid_command_time = current_time_s
        self.prev_status = planner_status

        diagnostics["command_linear_x"] = round(clamped_v, 3)
        diagnostics["command_angular_z"] = round(clamped_w, 3)

        return clamped_v, clamped_w, planner_status, diagnostics
