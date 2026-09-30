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
    """Parameters governing hazard latching and TTL decay."""
    proximity_threshold_m: float = 2.50       # Only latch hazards observed within this distance
    critical_proximity_m: float = 1.20        # Critical close zone
    base_ttl_s: float = 3.50                  # Standard TTL for hazards in proximity
    close_ttl_s: float = 5.00                 # Extended conservative TTL for very close hazards
    decay_rate_per_sec: float = 0.20          # Confidence decay


class RecentHazardMemory:
    """Maintains short-term memory of close hazards to prevent free-space hallucination on dropouts."""

    def __init__(self, config: Optional[RecentHazardMemoryConfig] = None) -> None:
        self.config = config or RecentHazardMemoryConfig()
        self._latched_hazards: Dict[str, LatchedHazard] = {}

    def update(
        self,
        active_tracks: Sequence[TrackedObstacleState],
        current_time_s: float,
        robot_pose: Tuple[float, float, float],
    ) -> List[LatchedHazard]:
        """Update hazard memory with latest tracks; latch close tracks that disappear."""
        rx, ry, _ = robot_pose
        now = float(current_time_s)

        # 1. Inspect current active tracks
        active_track_ids = set()
        for track in active_tracks:
            tid = str(track.track_id)
            active_track_ids.add(tid)
            dist = math.hypot(track.x - rx, track.y - ry)

            # If within proximity threshold, refresh or register in memory
            if dist <= self.config.proximity_threshold_m:
                ttl = self.config.close_ttl_s if dist <= self.config.critical_proximity_m else self.config.base_ttl_s
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
                )
                self._latched_hazards[tid] = hazard

        # 2. For latched hazards NOT in active_tracks: they are currently DROPPED/LOST
        expired_ids = []
        for hid, hazard in self._latched_hazards.items():
            if hid not in active_track_ids:
                # Dropout occurring: check if expired
                if hazard.is_expired(now):
                    expired_ids.append(hid)
                else:
                    # Decay confidence over time during loss
                    dt = now - hazard.last_observed_time_s
                    hazard.confidence = max(0.20, 0.95 - dt * self.config.decay_rate_per_sec)

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
    corridor_half_width_m: float = 0.35       # Vehicle half width (0.225m) + safety margin (0.125m)
    rear_clearance_min_m: float = 0.45        # Minimum clearance required for reverse
    max_linear_deceleration_mps2: float = 1.50 # Deceleration constraint


class SafetyGovernor:
    """Authoritative downstream safety filter guarding the vehicle against collisions."""

    def __init__(self, config: Optional[SafetyGovernorConfig] = None) -> None:
        self.config = config or SafetyGovernorConfig()

    def evaluate(
        self,
        requested_v: float,
        requested_w: float,
        robot_pose: Tuple[float, float, float],
        hazards: Sequence[Union[TrackedObstacleState, LatchedHazard]],
        current_time_s: float = 0.0,
    ) -> Tuple[float, float, str, Dict[str, Any]]:
        """Evaluate commanded velocity against all active and latched hazards.

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

            effective_dist_x = max(0.0, abs(x_rel) - obs_radius)
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

        # ---------------------------------------------------------------------
        # Rule 1: Forward Motion Safety Governor
        # ---------------------------------------------------------------------
        if requested_v > 0.0:
            if closest_forward_dist <= self.config.stop_distance_m:
                # Immediate hard stop: collision danger
                governed_v = 0.0
                governed_w = 0.0
                decision = f"STOP_FORWARD_HAZARD (dist={closest_forward_dist:.2f}m <= {self.config.stop_distance_m:.2f}m, obs={blocking_hazard_id})"
            elif closest_forward_dist <= self.config.cautious_distance_m:
                # Cautious speed reduction
                governed_v = min(requested_v, self.config.cautious_max_v_mps)
                decision = f"CAUTIOUS_SLOWDOWN (dist={closest_forward_dist:.2f}m, capped to {governed_v:.2f}m/s, obs={blocking_hazard_id})"

        # ---------------------------------------------------------------------
        # Rule 2: Reverse Motion Safety Governor
        # ---------------------------------------------------------------------
        elif requested_v < 0.0:
            if closest_rear_dist <= self.config.rear_clearance_min_m:
                # Immediate stop: rear blocked
                governed_v = 0.0
                governed_w = 0.0
                decision = f"STOP_REAR_BLOCKED (dist={closest_rear_dist:.2f}m <= {self.config.rear_clearance_min_m:.2f}m, obs={blocking_hazard_id})"

        diagnostics = {
            "requested_v": round(float(requested_v), 3),
            "requested_w": round(float(requested_w), 3),
            "governed_v": round(float(governed_v), 3),
            "governed_w": round(float(governed_w), 3),
            "decision": decision,
            "closest_forward_dist_m": round(float(closest_forward_dist), 3) if closest_forward_dist < 900.0 else None,
            "closest_rear_dist_m": round(float(closest_rear_dist), 3) if closest_rear_dist < 900.0 else None,
            "blocking_hazard_id": blocking_hazard_id,
        }

        return governed_v, governed_w, decision, diagnostics
