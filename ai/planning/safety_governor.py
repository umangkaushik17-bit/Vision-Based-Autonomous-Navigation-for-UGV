"""Lost-Obstacle Safety Governor and Recent Hazard Memory (Phase 13.23).

Enforces the core principle: "NOT DETECTED != SAFE".

Prevents a recently observed close obstacle from becoming unsafe free space merely
because perception or tracking temporarily drops the track.

Architectural Guarantees:
1. Recent Hazard Memory (Latch): Preserves recently observed hazards within close proximity
   (< 2.5m) with a bounded, physics-justified Time-To-Live (TTL).
2. Authoritative Safety Governor: Positioned downstream of behavioral arbitration, immediately
   upstream of the command buffer.
3. Asymmetric Override: The Safety Governor can only REDUCE velocity or command STOP;
   it can NEVER increase risk.
4. Latched Hazard Synthesis: Latched hazards can be synthesized as stationary obstacle
   predictions so DWA continues to account for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
)


@dataclass
class LatchedHazard:
    """Persistent representation of a recently observed close hazard whose tracking dropped out."""
    hazard_id: str
    x: float                                  # World x (odom)
    y: float                                  # World y (odom)
    vx: float = 0.0                           # Last estimated velocity
    vy: float = 0.0
    bbox_size: Tuple[float, float] = (0.8, 0.8)
    bbox_yaw: float = 0.0
    first_observed_time_s: float = 0.0
    last_observed_time_s: float = 0.0
    last_distance_m: float = 0.0
    ttl_s: float = 3.5                        # Remaining or total validity window
    confidence: float = 0.90
    source_track_id: Optional[int] = None
    semantic_class: str = "obstacle"

    def is_expired(self, current_time_s: float) -> bool:
        """Check if hazard has exceeded its bounded time-to-live."""
        age = current_time_s - self.last_observed_time_s
        return age > self.ttl_s

    def to_predicted_trajectory(self, horizon_s: float = 2.0, dt_s: float = 0.25) -> PredictedObstacleTrajectory:
        """Convert latched hazard into a conservative stationary prediction for DWA consumption."""
        track_id_val = self.source_track_id if self.source_track_id is not None else 999
        track_state = TrackedObstacleState(
            track_id=track_id_val,
            timestamp=self.last_observed_time_s,
            frame_id="odom",
            x=self.x,
            y=self.y,
            vx=0.0,
            vy=0.0,
            covariance=np.eye(4, dtype=np.float64) * 0.10,
            bbox_size=self.bbox_size,
            bbox_yaw=self.bbox_yaw,
            track_age=10,
            coast_cycles=1,
            is_confirmed=True,
            status="TRACKING",
        )

        steps = int(round(horizon_s / dt_s))
        waypoints: List[TrajectoryWaypoint] = []
        bounding_polygons: List[np.ndarray] = []
        cov_2d = np.eye(2, dtype=np.float64) * 0.05

        for i in range(steps + 1):
            t_offset = i * dt_s
            unc_radius = 0.10 + 0.05 * t_offset
            waypoints.append(
                TrajectoryWaypoint(
                    time_offset=t_offset,
                    timestamp=self.last_observed_time_s + t_offset,
                    x=self.x,
                    y=self.y,
                    vx=0.0,
                    vy=0.0,
                    covariance_2d=cov_2d,
                    uncertainty_radius=unc_radius,
                )
            )
            poly = compute_obb_corners(
                (self.x, self.y),
                self.bbox_size,
                self.bbox_yaw,
                margin_m=unc_radius,
            )
            bounding_polygons.append(poly)

        return PredictedObstacleTrajectory(
            track_id=track_id_val,
            current_state=track_state,
            horizon_s=horizon_s,
            dt=dt_s,
            waypoints=waypoints,
            bounding_polygon_per_step=bounding_polygons,
        )


@dataclass
class RecentHazardMemoryConfig:
    """Parameters governing hazard latching and semantic-weighted TTL decay (WS-2B)."""
    proximity_threshold_m: float = 2.50       # Only latch hazards observed within this distance
    critical_proximity_m: float = 1.20        # Critical close zone
    base_ttl_s: float = 3.50                  # Standard TTL for hazards in proximity
    close_ttl_s: float = 5.00                 # Extended conservative TTL for very close hazards
    decay_rate_per_sec: float = 0.20          # Confidence decay
    near_field_blind_spot_max_x_m: float = 1.25  # Max forward x for costmap blind-spot latching
    near_field_blind_spot_half_width_m: float = 0.55  # Half-width for costmap blind-spot latching
    min_active_confidence: float = 0.20       # Prune latched hazard if decayed below threshold
    semantic_ttl_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "obstacle": 1.00,
            "boulder": 1.25,
            "rock": 1.25,
            "vehicle": 1.20,
            "pedestrian": 1.30,
            "water": 1.20,
            "ditch": 1.20,
            "grass": 0.25,
            "tall_grass": 0.25,
            "shadow": 0.20,
            "glare": 0.20,
            "foliage": 0.30,
        }
    )
    semantic_decay_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "obstacle": 1.00,
            "boulder": 0.80,
            "rock": 0.80,
            "vehicle": 0.90,
            "pedestrian": 0.80,
            "water": 0.85,
            "ditch": 0.85,
            "grass": 3.50,
            "tall_grass": 3.50,
            "shadow": 4.00,
            "glare": 4.00,
            "foliage": 3.00,
        }
    )


class RecentHazardMemory:
    """Maintains short-term memory of close hazards to prevent free-space hallucination on dropouts."""

    def __init__(self, config: Optional[RecentHazardMemoryConfig] = None) -> None:
        self.config = config or RecentHazardMemoryConfig()
        self._latched_hazards: Dict[str, LatchedHazard] = {}

    def latch_near_field_costmap_obstacles(
        self,
        costmap: Any,
        current_time_s: float,
        robot_pose: Tuple[float, float, float],
    ) -> Optional[LatchedHazard]:
        """Scans static/vision costmap in the near-field camera blind spot (< 1.25m ahead)
        and latches any detected obstacle cell so under-bumper FOV dropout cannot hallucinate free space.
        """
        if costmap is None or not hasattr(costmap, "obstacle") or not hasattr(costmap, "world_to_grid"):
            return None

        rx, ry, ryaw = robot_pose
        cos_yaw = math.cos(ryaw)
        sin_yaw = math.sin(ryaw)
        now = float(current_time_s)

        best_dist = 999.0
        best_wx = 0.0
        best_wy = 0.0
        found = False

        # Sample forward near-field blind-spot grid in robot frame
        for x_rel in (0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20):
            if x_rel > self.config.near_field_blind_spot_max_x_m:
                break
            for y_rel in (-0.30, -0.15, 0.0, 0.15, 0.30):
                wx = rx + x_rel * cos_yaw - y_rel * sin_yaw
                wy = ry + x_rel * sin_yaw + y_rel * cos_yaw
                if hasattr(costmap, "is_in_bounds") and not costmap.is_in_bounds(wx, wy):
                    continue
                r, c = costmap.world_to_grid(wx, wy)
                if 0 <= r < costmap.shape[0] and 0 <= c < costmap.shape[1]:
                    if costmap.obstacle[r, c]:
                        d = math.hypot(x_rel, y_rel)
                        if d < best_dist:
                            best_dist = d
                            best_wx = wx
                            best_wy = wy
                            found = True

        hid = "near_field_costmap_blindspot"
        if found:
            ttl = self.config.close_ttl_s if best_dist <= self.config.critical_proximity_m else self.config.base_ttl_s
            hazard = LatchedHazard(
                hazard_id=hid,
                x=float(best_wx),
                y=float(best_wy),
                vx=0.0,
                vy=0.0,
                bbox_size=(0.30, 0.30),
                bbox_yaw=float(ryaw),
                first_observed_time_s=now if hid not in self._latched_hazards else self._latched_hazards[hid].first_observed_time_s,
                last_observed_time_s=now,
                last_distance_m=best_dist,
                ttl_s=ttl,
                confidence=0.95,
                source_track_id=998,
            )
            self._latched_hazards[hid] = hazard
            return hazard
        return None

    def update(
        self,
        active_tracks: Sequence[TrackedObstacleState],
        current_time_s: float,
        robot_pose: Tuple[float, float, float],
        costmap: Any = None,
    ) -> List[LatchedHazard]:
        """Update hazard memory with latest tracks and optional near-field costmap; latch close hazards that disappear."""
        rx, ry, _ = robot_pose
        now = float(current_time_s)

        # 1. Inspect current active tracks
        active_track_ids = set()
        if costmap is not None:
            nf_h = self.latch_near_field_costmap_obstacles(costmap, now, robot_pose)
            if nf_h is not None:
                active_track_ids.add(nf_h.hazard_id)
        for track in active_tracks:
            tid = str(track.track_id)
            active_track_ids.add(tid)
            dist = math.hypot(track.x - rx, track.y - ry)

            # If within proximity threshold, refresh or register in memory
            if dist <= self.config.proximity_threshold_m:
                sem_cls = str(getattr(track, "semantic_class", getattr(track, "class_name", "obstacle"))).lower().strip()
                ttl_mult = float(self.config.semantic_ttl_multipliers.get(sem_cls, 1.0))
                raw_ttl = self.config.close_ttl_s if dist <= self.config.critical_proximity_m else self.config.base_ttl_s
                ttl = raw_ttl * ttl_mult
                hazard = LatchedHazard(
                    hazard_id=tid,
                    x=float(track.x),
                    y=float(track.y),
                    vx=float(track.vx),
                    vy=float(track.vy),
                    bbox_size=track.bbox_size,
                    bbox_yaw=float(track.bbox_yaw),
                    first_observed_time_s=now if tid not in self._latched_hazards else self._latched_hazards[tid].first_observed_time_s,
                    last_observed_time_s=now,
                    last_distance_m=dist,
                    ttl_s=ttl,
                    confidence=0.95,
                    source_track_id=track.track_id,
                    semantic_class=sem_cls,
                )
                self._latched_hazards[tid] = hazard

        # 2. For latched hazards NOT in active_tracks: they are currently DROPPED/LOST
        expired_ids = []
        for hid, hazard in self._latched_hazards.items():
            if hid not in active_track_ids:
                # Dropout occurring: check if expired or confidence decayed below threshold
                if hazard.is_expired(now):
                    expired_ids.append(hid)
                else:
                    dt = now - hazard.last_observed_time_s
                    decay_mult = float(self.config.semantic_decay_multipliers.get(hazard.semantic_class, 1.0))
                    eff_decay = self.config.decay_rate_per_sec * decay_mult
                    if decay_mult > 1.0:
                        hazard.confidence = 0.95 - dt * eff_decay
                        if hazard.confidence < self.config.min_active_confidence:
                            expired_ids.append(hid)
                    else:
                        hazard.confidence = max(0.20, 0.95 - dt * eff_decay)

        for hid in expired_ids:
            del self._latched_hazards[hid]

        return list(self._latched_hazards.values())

    def get_latched_dropped_hazards(self, active_track_ids: Sequence[Union[int, str]]) -> List[LatchedHazard]:
        """Return only hazards that are currently NOT actively tracked (pure dropouts)."""
        active_set = {str(tid) for tid in active_track_ids}
        return [h for hid, h in self._latched_hazards.items() if hid not in active_set]

    def clear(self) -> None:
        """Clear all latched hazards."""
        self._latched_hazards.clear()


@dataclass
class SafetyGovernorConfig:
    """Thresholds for physical safety governor override."""
    stop_distance_m: float = 0.40             # Hard stop distance to obstacle
    cautious_distance_m: float = 0.95         # Deceleration envelope distance
    cautious_max_v_mps: float = 0.15          # Max linear speed in cautious envelope
    degraded_max_v_mps: float = 0.35          # Conservative speed cap on stale camera / sensor dropout
    camera_stale_timeout_s: float = 0.50      # Camera frame age threshold for degraded speed clamping
    corridor_half_width_m: float = 0.35       # Vehicle half width (0.225m) + safety margin (0.125m)
    rear_clearance_min_m: float = 0.45        # Minimum clearance required for reverse
    max_linear_deceleration_mps2: float = 1.50 # Deceleration constraint
    front_bumper_offset_m: float = 0.0        # Optional chassis half-length bumper offset
    critical_tilt_deg: float = 26.0           # Authoritative rollover stop threshold (deg)
    cautious_tilt_deg: float = 18.0           # Slope speed derating threshold (deg)

    @property
    def degraded_max_v(self) -> float:
        return self.degraded_max_v_mps


class SafetyGovernor:
    """Authoritative downstream safety filter guarding the vehicle against collisions and rollover."""

    def __init__(self, config: Optional[SafetyGovernorConfig] = None) -> None:
        self.config = config or SafetyGovernorConfig()

    def compute_downhill_stopping_distance(
        self,
        downhill_pitch_rad: float,
        base_distance_m: Optional[float] = None,
    ) -> float:
        """Compute slope-compensated stopping distance on downhill grades (Subphase 15.7).

        d_stop(theta) = d_stop_0 * a_max / max(0.30, a_max - g * sin(theta_downhill))
        """
        d0 = float(self.config.stop_distance_m if base_distance_m is None else base_distance_m)
        theta = max(0.0, float(downhill_pitch_rad))
        if theta <= 1e-4:
            return d0
        a_max = float(self.config.max_linear_deceleration_mps2)
        g_sin = 9.81 * math.sin(theta)
        scale = a_max / max(0.30, a_max - g_sin)
        return float(d0 * scale)

    def evaluate(
        self,
        requested_v: float,
        requested_w: float,
        robot_pose: Tuple[float, float, float],
        hazards: Sequence[Union[TrackedObstacleState, LatchedHazard]],
        current_time_s: float = 0.0,
        robot_roll_pitch: Tuple[float, float] = (0.0, 0.0),
        costmap: Any = None,
        camera_frame_age_s: Optional[float] = None,
        camera_dropout: bool = False,
    ) -> Tuple[float, float, str, Dict[str, Any]]:
        """Evaluate commanded velocity against all active/latched hazards, static costmap, and 3D tilt limits.

        Returns:
            Tuple of (governed_v, governed_w, decision_reason, diagnostics)
        """
        rx, ry, ryaw = robot_pose
        cos_yaw = math.cos(ryaw)
        sin_yaw = math.sin(ryaw)

        governed_v = float(requested_v)
        governed_w = float(requested_w)
        decision = "ALLOW"
        closest_forward_dist = 999.0
        closest_rear_dist = 999.0
        blocking_hazard_id: Optional[str] = None

        # Check 3D chassis attitude tilt (roll, pitch) for rollover protection
        raw_roll_rad = float(robot_roll_pitch[0])
        raw_pitch_rad = float(robot_roll_pitch[1])
        roll_deg = abs(math.degrees(raw_roll_rad))
        pitch_deg = abs(math.degrees(raw_pitch_rad))
        max_tilt_deg = max(roll_deg, pitch_deg)

        # Subphase 15.7: Downhill Stopping Distance Compensation
        downhill_pitch_rad = max(0.0, raw_pitch_rad) if requested_v >= 0.0 else max(0.0, -raw_pitch_rad)
        eff_stop_distance_m = self.compute_downhill_stopping_distance(
            downhill_pitch_rad, self.config.stop_distance_m
        )
        eff_cautious_distance_m = self.compute_downhill_stopping_distance(
            downhill_pitch_rad, self.config.cautious_distance_m
        )

        # Check each hazard in robot local coordinate frame
        for h in hazards:
            hx = float(h.x)
            hy = float(h.y)
            dx_world = hx - rx
            dy_world = hy - ry

            # Transform to robot frame (x_rel forward, y_rel left)
            x_rel = dx_world * cos_yaw + dy_world * sin_yaw
            y_rel = -dx_world * sin_yaw + dy_world * cos_yaw

            # Obstacle radius approximation
            obs_radius = 0.40
            if hasattr(h, "bbox_size") and len(h.bbox_size) >= 2:
                obs_radius = max(float(h.bbox_size[0]), float(h.bbox_size[1])) / 2.0

            effective_dist_x = max(0.0, abs(x_rel) - obs_radius - self.config.front_bumper_offset_m)
            is_in_corridor = (abs(y_rel) - obs_radius) <= self.config.corridor_half_width_m

            # 1. Forward Path Check (requested_v > 0)
            if requested_v > 0.0 and x_rel > 0.0 and is_in_corridor:
                if effective_dist_x < closest_forward_dist:
                    closest_forward_dist = effective_dist_x
                    blocking_hazard_id = str(getattr(h, "hazard_id", getattr(h, "track_id", "unknown")))

            # 2. Rear Path Check (requested_v < 0)
            elif requested_v < 0.0 and x_rel < 0.0 and is_in_corridor:
                if effective_dist_x < closest_rear_dist:
                    closest_rear_dist = effective_dist_x
                    blocking_hazard_id = str(getattr(h, "hazard_id", getattr(h, "track_id", "unknown")))

        # Static & 2.5D terrain costmap forward corridor check (obstacle, >= 28 deg slope, >= 0.12 m step/ditch)
        if costmap is not None and hasattr(costmap, "world_to_grid"):
            if requested_v > 0.0:
                slope_layer = getattr(costmap, "slope_deg", getattr(costmap, "slope", None))
                hvar_layer = getattr(costmap, "height_var", None)
                obs_layer = getattr(costmap, "obstacle", None)
                max_scan_x = min(2.20, max(0.85, eff_cautious_distance_m + 0.10))
                x_samples = [0.20, 0.30, 0.40, 0.55, 0.70, 0.85]
                x_extra = 1.00
                while x_extra <= max_scan_x + 1e-4:
                    x_samples.append(round(x_extra, 2))
                    x_extra += 0.15

                for x_rel in x_samples:
                    for y_rel in (-0.20, 0.0, 0.20):
                        wx = rx + x_rel * cos_yaw - y_rel * sin_yaw
                        wy = ry + x_rel * sin_yaw + y_rel * cos_yaw
                        if hasattr(costmap, "is_in_bounds") and not costmap.is_in_bounds(wx, wy):
                            continue
                        r, c = costmap.world_to_grid(wx, wy)
                        if 0 <= r < costmap.shape[0] and 0 <= c < costmap.shape[1]:
                            is_obs = bool(obs_layer[r, c]) if obs_layer is not None else False
                            is_steep = bool(float(slope_layer[r, c]) >= 28.0) if slope_layer is not None else False
                            is_step = bool(float(hvar_layer[r, c]) >= 0.12) if hvar_layer is not None else False
                            if is_obs or is_steep or is_step:
                                eff_d = max(0.0, x_rel - self.config.front_bumper_offset_m)
                                if eff_d < closest_forward_dist:
                                    closest_forward_dist = eff_d
                                    if is_steep:
                                        blocking_hazard_id = "terrain_slope_dropoff"
                                    elif is_step:
                                        blocking_hazard_id = "terrain_step_ditch_hazard"
                                    else:
                                        blocking_hazard_id = "static_costmap_blindspot"

        # ---------------------------------------------------------------------
        # Rule 0: 3D Attitude Rollover Safety Governor
        # ---------------------------------------------------------------------
        if max_tilt_deg >= self.config.critical_tilt_deg:
            governed_v = 0.0
            governed_w = 0.0
            decision = f"STOP_TILT_ROLLOVER_HAZARD (tilt={max_tilt_deg:.1f}deg >= {self.config.critical_tilt_deg:.1f}deg)"
        elif max_tilt_deg >= self.config.cautious_tilt_deg and abs(governed_v) > self.config.cautious_max_v_mps:
            governed_v = math.copysign(self.config.cautious_max_v_mps, governed_v)
            decision = f"CAUTIOUS_TILT_SLOWDOWN (tilt={max_tilt_deg:.1f}deg, capped to {governed_v:.2f}m/s)"

        # ---------------------------------------------------------------------
        # Rule 1: Forward Motion Safety Governor (with Downhill Braking Expansion)
        # ---------------------------------------------------------------------
        if requested_v > 0.0 and "STOP_TILT" not in decision:
            if closest_forward_dist <= eff_stop_distance_m:
                # Immediate hard stop: collision / terrain drop-off danger
                governed_v = 0.0
                governed_w = 0.0
                decision = f"STOP_FORWARD_HAZARD (dist={closest_forward_dist:.2f}m <= {eff_stop_distance_m:.2f}m, obs={blocking_hazard_id})"
            elif closest_forward_dist <= eff_cautious_distance_m:
                # Cautious speed reduction
                governed_v = min(governed_v, self.config.cautious_max_v_mps)
                decision = f"CAUTIOUS_SLOWDOWN (dist={closest_forward_dist:.2f}m <= {eff_cautious_distance_m:.2f}m, capped to {governed_v:.2f}m/s, obs={blocking_hazard_id})"

        # ---------------------------------------------------------------------
        # Rule 2: Reverse Motion Safety Governor
        # ---------------------------------------------------------------------
        elif requested_v < 0.0 and "STOP_TILT" not in decision:
            if closest_rear_dist <= self.config.rear_clearance_min_m:
                # Immediate stop: rear blocked
                governed_v = 0.0
                governed_w = 0.0
                decision = f"STOP_REAR_BLOCKED (dist={closest_rear_dist:.2f}m <= {self.config.rear_clearance_min_m:.2f}m, obs={blocking_hazard_id})"

        # ---------------------------------------------------------------------
        # Rule 3: Camera Dropout / Stale Frame Degraded Velocity Clamping (Subphase 16.1)
        # ---------------------------------------------------------------------
        is_camera_stale = bool(camera_dropout) or (
            camera_frame_age_s is not None and float(camera_frame_age_s) > self.config.camera_stale_timeout_s
        )
        if is_camera_stale and "STOP" not in decision and abs(governed_v) > self.config.degraded_max_v_mps:
            governed_v = math.copysign(self.config.degraded_max_v_mps, governed_v)
            if decision == "ALLOW":
                decision = f"CAMERA_DROPOUT_DEGRADED_SPEED (capped to {governed_v:.2f}m/s)"

        diagnostics = {
            "requested_v": round(float(requested_v), 3),
            "requested_w": round(float(requested_w), 3),
            "governed_v": round(float(governed_v), 3),
            "governed_w": round(float(governed_w), 3),
            "decision": decision,
            "closest_forward_dist_m": round(float(closest_forward_dist), 3) if closest_forward_dist < 900.0 else None,
            "closest_rear_dist_m": round(float(closest_rear_dist), 3) if closest_rear_dist < 900.0 else None,
            "blocking_hazard_id": blocking_hazard_id,
            "max_tilt_deg": round(max_tilt_deg, 2),
            "effective_stop_distance_m": round(float(eff_stop_distance_m), 3),
            "effective_cautious_distance_m": round(float(eff_cautious_distance_m), 3),
            "camera_stale_or_dropout": is_camera_stale,
            "degraded_max_v_mps": round(float(self.config.degraded_max_v_mps), 3),
        }

        return governed_v, governed_w, decision, diagnostics
