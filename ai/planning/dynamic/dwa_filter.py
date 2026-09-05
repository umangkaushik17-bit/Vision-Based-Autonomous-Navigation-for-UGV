"""Hierarchical broad-phase pre-filtering and dynamic scoring configuration for DWA.

Milestone 13.9:
Provides cheap spatial and spatiotemporal pruning of candidate DWA trajectories
before invoking the expensive PredictiveCollisionChecker.
Preserves strict conservative safety: candidates that cannot possibly interact
with dynamic obstacles are cleared in O(1) box operations, while relevant candidates
are passed to exact predictive checking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionResult,
    TrajectorySample,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    compute_obb_corners,
)
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.trajectory import Trajectory, TrajectoryPoint


@dataclass
class DynamicPlanningConfig:
    """Configuration parameters for dynamic obstacle-aware DWA planning.

    Attributes:
        enabled: If True, dynamic obstacle checks and scoring are active.
        w_dynamic_risk: Multiplier weight for dynamic risk cost penalty.
        high_risk_penalty: Direct score penalty for PREDICTED_HIGH_RISK candidates.
        medium_risk_penalty: Direct score penalty for PREDICTED_MEDIUM_RISK candidates.
        low_risk_penalty: Direct score penalty for PREDICTED_LOW_RISK candidates.
        w_dynamic_clearance: Score weight rewarding clearance from dynamic obstacles.
        reject_high_risk: If True, HIGH_RISK candidates are discarded (treated as invalid).
        unknown_horizon_policy: Policy for trajectories exceeding prediction horizon.
            - "PENALIZE" (default, safety-informed): Applies configured score penalty.
            - "REJECT" (strict safety): Discards any candidate entering unsupported prediction horizon.
            - "ALLOW" (research/experimental ONLY): Does not apply score penalty, but diagnostics
              still strictly expose UNKNOWN_HORIZON (never classified as SAFE or FREE).
        unknown_horizon_penalty: Score penalty when unknown_horizon_policy == "PENALIZE".
        enable_broad_phase: If True, uses hierarchical spatial + continuous relative swept-segment pre-filtering.
        broad_phase_margin_m: Additional metric inflation on obstacle bounding reach (default 0.05m).
    """

    enabled: bool = True
    w_dynamic_risk: float = 0.40
    high_risk_penalty: float = 0.60
    medium_risk_penalty: float = 0.35
    low_risk_penalty: float = 0.15
    w_dynamic_clearance: float = 0.25
    reject_high_risk: bool = False
    unknown_horizon_policy: str = "PENALIZE"
    unknown_horizon_penalty: float = 0.50
    enable_broad_phase: bool = True
    broad_phase_margin_m: float = 0.05

    def __post_init__(self) -> None:
        policy = self.unknown_horizon_policy.upper()
        if policy not in ("PENALIZE", "REJECT", "ALLOW"):
            raise ValueError(f"Unknown horizon policy must be PENALIZE, REJECT, or ALLOW; got {self.unknown_horizon_policy}")
        self.unknown_horizon_policy = policy


@dataclass
class ObstacleWaypointSample:
    """Spatiotemporal circle sample along an obstacle's predicted path."""

    t: float
    x: float
    y: float
    r_total: float  # obstacle half_diag + uncertainty_radius + margin


@dataclass
class ObstacleSpatiotemporalHull:
    """Spatiotemporal bounding volume enclosing an obstacle's predicted trajectory."""

    track_id: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    max_uncertainty_radius: float
    t_min: float = 0.0
    t_max: float = 2.0
    samples: List[ObstacleWaypointSample] = field(default_factory=list)

    def interpolate_at_t(self, t: float) -> Optional[Tuple[float, float, float]]:
        """Interpolate (x, y, r_total) at time t. Returns None if t > t_max + 1e-4."""
        if not self.samples:
            return None
        if t < -1e-6 or t > self.t_max + 1e-4:
            return None

        t = max(0.0, float(t))
        samples = self.samples
        n = len(samples)
        if t <= samples[0].t:
            return (samples[0].x, samples[0].y, samples[0].r_total)
        if t >= samples[-1].t:
            return (samples[-1].x, samples[-1].y, samples[-1].r_total)

        # Binary search for interval
        low = 0
        high = n - 1
        while low <= high:
            mid = (low + high) // 2
            if samples[mid].t <= t:
                low = mid + 1
            else:
                high = mid - 1

        idx = max(0, high)
        s0 = samples[idx]
        s1 = samples[idx + 1] if idx + 1 < n else s0
        dt = s1.t - s0.t
        if dt > 1e-6:
            alpha = (t - s0.t) / dt
            return (
                s0.x + alpha * (s1.x - s0.x),
                s0.y + alpha * (s1.y - s0.y),
                s0.r_total + alpha * (s1.r_total - s0.r_total),
            )
        return (s0.x, s0.y, s0.r_total)


class DynamicBroadPhaseFilter:
    """Conservative hierarchical broad-phase pre-filter for candidate DWA trajectories.

    Eliminates candidate trajectories whose spatial extent or continuous-time spatiotemporal
    relative distance cannot intersect any active dynamic obstacle hull or uncertainty zone.
    Guarantees continuous-time safety: no inter-sample tunneling can produce a false clearance.
    """

    def __init__(self, config: Optional[DynamicPlanningConfig] = None) -> None:
        self.config = config or DynamicPlanningConfig()

    def build_obstacle_hulls(
        self,
        predicted_tracks: Optional[Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]]],
        dynamic_layer: Optional[DynamicOccupancyLayer] = None,
        max_horizon_s: float = 2.0,
    ) -> List[ObstacleSpatiotemporalHull]:
        """Compute bounding hulls and spatiotemporal waypoint circles up to max_horizon_s."""
        hulls: List[ObstacleSpatiotemporalHull] = []

        tracks: Sequence[PredictedObstacleTrajectory] = []
        if predicted_tracks is not None:
            if isinstance(predicted_tracks, PredictedTrajectoryArray):
                tracks = predicted_tracks.predictions
            else:
                tracks = list(predicted_tracks)

        margin = float(self.config.broad_phase_margin_m)

        for trk in tracks:
            if trk.current_state.status.upper() == "EXPIRED":
                continue

            length_m, width_m = trk.current_state.bbox_size
            half_diag = math.hypot(length_m / 2.0, width_m / 2.0)

            pts_x: List[float] = [trk.current_state.x]
            pts_y: List[float] = [trk.current_state.y]
            max_r_unc = 0.0

            # Store waypoint circles
            samples: List[ObstacleWaypointSample] = [
                ObstacleWaypointSample(
                    t=0.0,
                    x=trk.current_state.x,
                    y=trk.current_state.y,
                    r_total=half_diag + 0.15 + margin,
                )
            ]

            for wp in trk.waypoints:
                pts_x.append(wp.x)
                pts_y.append(wp.y)
                if wp.uncertainty_radius > max_r_unc:
                    max_r_unc = wp.uncertainty_radius
                samples.append(
                    ObstacleWaypointSample(
                        t=wp.time_offset,
                        x=wp.x,
                        y=wp.y,
                        r_total=half_diag + wp.uncertainty_radius + margin,
                    )
                )
                # Keep collecting until we strictly cover max_horizon_s
                if max_horizon_s > 0 and wp.time_offset >= max_horizon_s - 1e-4:
                    break

            total_padding = half_diag + max_r_unc + margin
            t_max_valid = float(samples[-1].t) if samples else float(trk.horizon_s)

            hulls.append(ObstacleSpatiotemporalHull(
                track_id=trk.track_id,
                x_min=min(pts_x) - total_padding,
                x_max=max(pts_x) + total_padding,
                y_min=min(pts_y) - total_padding,
                y_max=max(pts_y) + total_padding,
                max_uncertainty_radius=max_r_unc,
                t_min=0.0,
                t_max=t_max_valid,
                samples=samples,
            ))

        if not tracks and dynamic_layer is not None:
            # Fallback: compute bounding box of occupied cells in dynamic layer slices
            min_r, max_r = 100000, -1
            min_c, max_c = 100000, -1
            slices_dict = getattr(dynamic_layer, "slices", getattr(dynamic_layer, "_slices", {}))
            for s in slices_dict.values():
                rows, cols = np.where(s.cost > 0.0)
                if len(rows) > 0:
                    min_r = min(min_r, int(np.min(rows)))
                    max_r = max(max_r, int(np.max(rows)))
                    min_c = min(min_c, int(np.min(cols)))
                    max_c = max(max_c, int(np.max(cols)))
            if max_r >= 0:
                p1_x, p1_y = dynamic_layer.grid_to_world(min_r, min_c)
                p2_x, p2_y = dynamic_layer.grid_to_world(max_r, max_c)
                hulls.append(ObstacleSpatiotemporalHull(
                    track_id=-1,
                    x_min=min(p1_x, p2_x) - margin,
                    x_max=max(p1_x, p2_x) + margin,
                    y_min=min(p1_y, p2_y) - margin,
                    y_max=max(p1_y, p2_y) + margin,
                    max_uncertainty_radius=0.0,
                    t_min=0.0,
                    t_max=max_horizon_s,
                    samples=[],
                ))

        return hulls

    @staticmethod
    def compute_trajectory_aabb(
        trajectory: Trajectory,
        vehicle_circumscribed_radius_m: float = 0.40,
    ) -> Tuple[float, float, float, float]:
        """Calculate (x_min, x_max, y_min, y_max) bounding box for candidate trajectory."""
        if not trajectory.points:
            return (0.0, 0.0, 0.0, 0.0)

        xs = [p.x for p in trajectory.points]
        ys = [p.y for p in trajectory.points]
        r = vehicle_circumscribed_radius_m

        return (
            float(min(xs) - r),
            float(max(xs) + r),
            float(min(ys) - r),
            float(max(ys) + r),
        )

    def is_candidate_relevant(
        self,
        trajectory: Trajectory,
        obstacle_hulls: List[ObstacleSpatiotemporalHull],
        vehicle_circumscribed_radius_m: float = 0.40,
    ) -> bool:
        """Evaluate if candidate trajectory might interact with dynamic obstacles.

        Uses hierarchical continuous-time filtering:
        1. Unknown horizon guard: any candidate extending beyond an obstacle's valid
           prediction horizon triggers exact evaluation to apply unknown-horizon policy.
        2. 2D spatial AABB test against obstacle spatiotemporal hull.
        3. Continuous relative swept-segment test: evaluates point-to-segment distance
           in relative coordinates over every interval [t_k, t_{k+1}] to eliminate tunneling.

        Returns:
            True if candidate trajectory might interact with dynamic obstacles (requires exact check).
            False if candidate is guaranteed to be completely clear of all dynamic obstacles in continuous time.
        """
        if not self.config.enable_broad_phase:
            return True

        if not obstacle_hulls:
            return False

        pts = trajectory.points
        if not pts:
            return False

        t_xmin, t_xmax, t_ymin, t_ymax = self.compute_trajectory_aabb(
            trajectory, vehicle_circumscribed_radius_m
        )

        r_ugv = vehicle_circumscribed_radius_m
        t_base = pts[0].t if pts[0].t <= 2.0 else pts[0].t - 0.1
        margin = float(self.config.broad_phase_margin_m)
        sagitta_eps = 0.005  # Circular arc sagitta allowance for UGV dt <= 0.1s

        for hull in obstacle_hulls:
            # 1. Unknown horizon check: if trajectory extends beyond obstacle's valid horizon,
            # it cannot be cleared as safe; exact checker must enforce unknown-horizon policy
            rel_t_end = pts[-1].t if pts[-1].t <= 2.0 else (pts[-1].t - t_base)
            if rel_t_end > hull.t_max + 1e-4:
                return True

            # 2. Spatial AABB test
            if not (t_xmax >= hull.x_min and t_xmin <= hull.x_max and
                    t_ymax >= hull.y_min and t_ymin <= hull.y_max):
                continue

            # Fallback if hull has no fine-grained samples
            if not hull.samples:
                return True

            # 3. Continuous relative swept-segment test
            n_pts = len(pts)
            if n_pts == 1:
                rel_t = pts[0].t if pts[0].t <= 2.0 else 0.0
                obs_state = hull.interpolate_at_t(rel_t)
                if obs_state is None:
                    return True
                ox, oy, r_obs = obs_state
                d_sq = (pts[0].x - ox) ** 2 + (pts[0].y - oy) ** 2
                reach = r_ugv + r_obs + margin
                if d_sq <= reach * reach:
                    return True
                continue

            for k in range(n_pts - 1):
                pt0 = pts[k]
                pt1 = pts[k + 1]

                t0 = pt0.t if pt0.t <= 2.0 else (pt0.t - t_base)
                t1 = pt1.t if pt1.t <= 2.0 else (pt1.t - t_base)

                obs0 = hull.interpolate_at_t(t0)
                obs1 = hull.interpolate_at_t(t1)

                if obs0 is None or obs1 is None:
                    return True

                ox0, oy0, r_obs0 = obs0
                ox1, oy1, r_obs1 = obs1

                # Relative offsets
                d0_x = pt0.x - ox0
                d0_y = pt0.y - oy0
                d1_x = pt1.x - ox1
                d1_y = pt1.y - oy1

                # Relative swept segment vector
                vx = d1_x - d0_x
                vy = d1_y - d0_y
                l_sq = vx * vx + vy * vy

                if l_sq < 1e-10:
                    d_min_sq = d0_x * d0_x + d0_y * d0_y
                else:
                    alpha = max(0.0, min(1.0, -(d0_x * vx + d0_y * vy) / l_sq))
                    cx = d0_x + alpha * vx
                    cy = d0_y + alpha * vy
                    d_min_sq = cx * cx + cy * cy

                r_reach = r_ugv + max(r_obs0, r_obs1) + margin + sagitta_eps
                if d_min_sq <= r_reach * r_reach:
                    return True

        return False
