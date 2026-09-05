"""Predictive collision checking for dynamic obstacles across spatiotemporal horizons.

Milestone 13.8:
Evaluates future UGV trajectories against the time-indexed DynamicOccupancyLayer
and predicted obstacle trajectories. Implements strict safety semantics:
- FREE / SAFE vs RISK vs LETHAL/COLLISION vs UNKNOWN / OUT_OF_HORIZON
- Out-of-horizon queries (t > horizon) MUST NOT silently become FREE.
- LOST/expired tracks beyond their valid horizon MUST NOT create artificial free space.
- Distinguishes physical footprint overlap from uncertainty risk and unknown coverage.
- Full rectangular UGV footprint model (corners, perimeter, and interior samples).
- Exact 2D polygon clearance and Separating Axis Theorem (SAT) collision checking.
- Multi-obstacle conservative arbitration (earliest critical collision preserved).
"""

from __future__ import annotations

from enum import Enum
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ai.planning.base import VehicleKinematics
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrajectoryWaypoint,
    compute_obb_corners,
    polygon_intersection_sat,
)
from ai.planning.dynamic.occupancy_layer import (
    DynamicOccupancyGridSlice,
    DynamicOccupancyLayer,
    DynamicRiskLevel,
)
from ai.planning.trajectory import Trajectory, TrajectoryPoint


class CollisionStatus(str, Enum):
    """Categorical classification of predictive trajectory collision check."""

    SAFE = "SAFE"
    PREDICTED_COLLISION = "PREDICTED_COLLISION"
    PREDICTED_HIGH_RISK = "PREDICTED_HIGH_RISK"
    PREDICTED_MEDIUM_RISK = "PREDICTED_MEDIUM_RISK"
    PREDICTED_LOW_RISK = "PREDICTED_LOW_RISK"
    UNKNOWN_HORIZON = "UNKNOWN_HORIZON"
    INVALID_INPUT = "INVALID_INPUT"


# Severity rank for deterministic arbitration (higher = more critical)
STATUS_SEVERITY_ORDER: Dict[CollisionStatus, int] = {
    CollisionStatus.INVALID_INPUT: 100,
    CollisionStatus.PREDICTED_COLLISION: 90,
    CollisionStatus.UNKNOWN_HORIZON: 80,
    CollisionStatus.PREDICTED_HIGH_RISK: 70,
    CollisionStatus.PREDICTED_MEDIUM_RISK: 60,
    CollisionStatus.PREDICTED_LOW_RISK: 50,
    CollisionStatus.SAFE: 10,
}


@dataclass
class TrajectorySample:
    """Discrete spatiotemporal sample along a candidate UGV trajectory.

    Attributes:
        time_offset: Relative future time in seconds (t >= 0.0).
        x: World/odom X coordinate in meters.
        y: World/odom Y coordinate in meters.
        yaw: Heading angle in radians.
        v: Forward linear velocity in m/s.
    """

    time_offset: float
    x: float
    y: float
    yaw: float = 0.0
    v: float = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def pose(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.yaw)


@dataclass
class PredictiveCollisionConfig:
    """Configuration parameters for predictive collision checking.

    Attributes:
        vehicle_length_m: UGV longitudinal dimension (default 0.65m matches Phase 12).
        vehicle_width_m: UGV lateral dimension (default 0.45m matches Phase 12).
        collision_margin_m: Outer safety expansion margin (default 0.05m matches Phase 12).
        max_horizon_s: Upper prediction horizon bound (default 2.00s).
        collision_cost_threshold: Numerical dynamic cost considered lethal collision (0.65).
        high_risk_cost_threshold: Dynamic cost for HIGH_RISK classification (0.70).
        medium_risk_cost_threshold: Dynamic cost for MEDIUM_RISK classification (0.40).
        low_risk_cost_threshold: Dynamic cost for LOW_RISK classification (0.15).
        perimeter_step_m: Metric step size for sampling UGV footprint edges (0.05m).
        interior_grid_density: Grid resolution (rows, cols) for sampling UGV interior points.
        conservative_sub_bin_union: If True, queries between layer time bins use conservative union.
        check_track_horizons: If True, tracks with horizon < sample time are treated as UNKNOWN.
        max_clearance_search_m: Maximum search distance for clearance reporting (5.0m).
        raise_on_invalid: If True, raises exceptions on invalid input rather than returning INVALID_INPUT.
    """

    vehicle_length_m: float = 0.65
    vehicle_width_m: float = 0.45
    collision_margin_m: float = 0.05
    max_horizon_s: float = 2.00
    collision_cost_threshold: float = 0.65
    high_risk_cost_threshold: float = 0.70
    medium_risk_cost_threshold: float = 0.40
    low_risk_cost_threshold: float = 0.15
    perimeter_step_m: float = 0.05
    interior_grid_density: Tuple[int, int] = (7, 5)
    conservative_sub_bin_union: bool = True
    check_track_horizons: bool = True
    max_clearance_search_m: float = 5.00
    raise_on_invalid: bool = False

    def __post_init__(self) -> None:
        if self.vehicle_length_m <= 0.0 or self.vehicle_width_m <= 0.0:
            raise ValueError("Vehicle dimensions must be strictly positive.")
        if self.max_horizon_s <= 0.0:
            raise ValueError("max_horizon_s must be strictly positive.")


@dataclass
class PredictiveCollisionResult:
    """Structured deterministic outcome of a predictive trajectory collision check.

    Attributes:
        collision: True strictly if a physical footprint overlap is predicted.
        collision_time: Time offset in seconds of earliest collision or primary critical event.
        collision_position: (x, y) coordinates of UGV center at collision/critical event.
        obstacle_id: Identifier of obstacle track involved in collision or primary risk.
        minimum_clearance: Minimum metric geometric clearance to obstacle physical footprint.
        maximum_cost: Maximum dynamic occupancy cost [0.0, 1.0] encountered.
        risk_classification: Highest DynamicRiskLevel encountered across trajectory.
        unknown_horizon_encountered: True if any sample queried beyond valid prediction horizon.
        samples_checked: Number of discrete trajectory samples evaluated.
        reason: Descriptive explanation of collision/risk classification.
        status: Overall CollisionStatus classification enum.
        details: Optional diagnostic payload.
    """

    collision: bool
    collision_time: Optional[float]
    collision_position: Optional[Tuple[float, float]]
    obstacle_id: Optional[int]
    minimum_clearance: float
    maximum_cost: float
    risk_classification: DynamicRiskLevel
    unknown_horizon_encountered: bool
    samples_checked: int
    reason: str
    status: CollisionStatus
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Geometric Distance and Footprint Utilities
# =============================================================================

def point_to_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Calculate exact Euclidean distance from point P(px, py) to line segment AB."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    seg_len_sq = abx * abx + aby * aby
    if seg_len_sq < 1e-12:
        return float(math.hypot(apx, apy))

    t = (apx * abx + apy * aby) / seg_len_sq
    if t <= 0.0:
        return float(math.hypot(apx, apy))
    elif t >= 1.0:
        return float(math.hypot(px - bx, py - by))

    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return float(math.hypot(px - proj_x, py - proj_y))


def compute_polygon_clearance(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Compute exact 2D Euclidean clearance between two convex polygons.

    If the polygons intersect, returns 0.0 immediately via Separating Axis Theorem.
    Otherwise, evaluates minimum distance across all vertex-to-edge pairs.

    Args:
        poly1: (N, 2) vertices of first polygon in sequential boundary order.
        poly2: (M, 2) vertices of second polygon in sequential boundary order.

    Returns:
        Exact minimum metric distance (m), or 0.0 if intersecting.
    """
    if polygon_intersection_sat(poly1, poly2):
        return 0.0

    min_dist = float("inf")
    n1 = len(poly1)
    n2 = len(poly2)

    # 1. Distances from vertices of poly1 to edges of poly2
    for i in range(n1):
        px = float(poly1[i, 0])
        py = float(poly1[i, 1])
        for j in range(n2):
            ax = float(poly2[j, 0])
            ay = float(poly2[j, 1])
            bx = float(poly2[(j + 1) % n2, 0])
            by = float(poly2[(j + 1) % n2, 1])
            d = point_to_segment_distance(px, py, ax, ay, bx, by)
            if d < min_dist:
                min_dist = d

    # 2. Distances from vertices of poly2 to edges of poly1
    for j in range(n2):
        px = float(poly2[j, 0])
        py = float(poly2[j, 1])
        for i in range(n1):
            ax = float(poly1[i, 0])
            ay = float(poly1[i, 1])
            bx = float(poly1[(i + 1) % n1, 0])
            by = float(poly1[(i + 1) % n1, 1])
            d = point_to_segment_distance(px, py, ax, ay, bx, by)
            if d < min_dist:
                min_dist = d

    return float(min_dist)


def sample_ugv_footprint_points(
    x_m: float,
    y_m: float,
    yaw_rad: float,
    length_m: float,
    width_m: float,
    margin_m: float = 0.0,
    perimeter_step_m: float = 0.05,
    interior_density: Tuple[int, int] = (7, 5),
) -> np.ndarray:
    """Generate comprehensive 2D sample points across oriented rectangular UGV footprint.

    Reuses the Phase 12 FootprintCollisionChecker representation:
    - 4 oriented bounding box corners (with margin)
    - Perimeter boundary samples along all 4 edges at perimeter_step_m
    - Interior grid points (interior_density[0] x interior_density[1])

    Args:
        x_m: Center X coordinate in world/odom frame.
        y_m: Center Y coordinate in world/odom frame.
        yaw_rad: Heading angle in radians.
        length_m: Vehicle length in meters.
        width_m: Vehicle width in meters.
        margin_m: Safety expansion margin in meters.
        perimeter_step_m: Maximum spacing between edge samples.
        interior_density: (n_rows, n_cols) grid for interior coverage.

    Returns:
        np.ndarray of shape (N, 2) containing metric sample coordinates.
    """
    hl = (length_m / 2.0) + margin_m
    hw = (width_m / 2.0) + margin_m

    corners = compute_obb_corners((x_m, y_m), (2.0 * hl, 2.0 * hw), yaw_rad, margin_m=0.0)

    points: List[Tuple[float, float]] = []

    # 1. Corners and perimeter samples
    for i in range(4):
        cx, cy = float(corners[i, 0]), float(corners[i, 1])
        points.append((cx, cy))

        next_i = (i + 1) % 4
        nx, ny = float(corners[next_i, 0]), float(corners[next_i, 1])
        edge_dist = float(math.hypot(nx - cx, ny - cy))
        n_sub = max(1, int(math.ceil(edge_dist / perimeter_step_m)))
        for k in range(1, n_sub):
            alpha = k / float(n_sub)
            points.append((cx + alpha * (nx - cx), cy + alpha * (ny - cy)))

    # 2. Interior sample grid
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    dx_samples = np.linspace(-hl * 0.85, hl * 0.85, interior_density[0])
    dy_samples = np.linspace(-hw * 0.80, hw * 0.80, interior_density[1])
    for dx in dx_samples:
        for dy in dy_samples:
            ix = x_m + (dx * c - dy * s)
            iy = y_m + (dx * s + dy * c)
            points.append((float(ix), float(iy)))

    return np.array(points, dtype=np.float64)


# =============================================================================
# Predictive Collision Checker Implementation
# =============================================================================

class PredictiveCollisionChecker:
    """Predictive dynamic collision checker between future UGV trajectories and dynamic obstacles.

    Standalone contract for Milestone 13.8. Does not modify DWA, A*, or Phase 12 safety clamps.
    Consumes DynamicOccupancyLayer and predicted obstacle trajectories.
    """

    def __init__(
        self,
        config: Optional[PredictiveCollisionConfig] = None,
        vehicle: Optional[VehicleKinematics] = None,
    ) -> None:
        if config is not None:
            self.config = config
        elif vehicle is not None:
            self.config = PredictiveCollisionConfig(
                vehicle_length_m=float(vehicle.length_m),
                vehicle_width_m=float(vehicle.width_m),
                collision_margin_m=float(vehicle.collision_margin_m),
            )
        else:
            self.config = PredictiveCollisionConfig()

        self.length = float(self.config.vehicle_length_m)
        self.width = float(self.config.vehicle_width_m)
        self.margin = float(self.config.collision_margin_m)
        self.r_inscribed = min(self.length, self.width) / 2.0
        self.r_circumscribed = math.hypot(self.length / 2.0, self.width / 2.0) + self.margin

    # -------------------------------------------------------------------------
    # Trajectory Normalization
    # -------------------------------------------------------------------------
    def _normalize_trajectory(
        self,
        trajectory: Any,
    ) -> Tuple[List[TrajectorySample], Optional[str]]:
        """Normalize heterogeneous trajectory inputs into standardized TrajectorySample list.

        Supports:
        - Trajectory (from ai.planning.trajectory)
        - Sequence[TrajectoryPoint]
        - Sequence[TrajectorySample]
        - Sequence[Tuple[float, ...]] (e.g. (t, x, y, yaw) or (t, x, y, yaw, v))

        Returns:
            Tuple of (sample_list, error_message_if_any).
        """
        if trajectory is None:
            return [], "Trajectory input is None"

        raw_points: Sequence[Any]
        if isinstance(trajectory, Trajectory):
            raw_points = trajectory.points
        elif isinstance(trajectory, (list, tuple)):
            raw_points = trajectory
        else:
            return [], f"Unsupported trajectory type: {type(trajectory)}"

        if not raw_points:
            return [], "Trajectory is empty (0 samples)"

        samples: List[TrajectorySample] = []
        prev_t = -1e-9

        for idx, pt in enumerate(raw_points):
            if isinstance(pt, TrajectorySample):
                t, x, y, yaw, v = pt.time_offset, pt.x, pt.y, pt.yaw, pt.v
            elif isinstance(pt, TrajectoryPoint):
                t, x, y, yaw, v = pt.t, pt.x, pt.y, pt.yaw, pt.v
            elif isinstance(pt, (tuple, list)):
                if len(pt) < 3:
                    return [], f"Tuple sample {idx} has fewer than 3 elements: {pt}"
                t = float(pt[0])
                x = float(pt[1])
                y = float(pt[2])
                yaw = float(pt[3]) if len(pt) > 3 else 0.0
                v = float(pt[4]) if len(pt) > 4 else 0.0
            else:
                return [], f"Unknown sample point type at index {idx}: {type(pt)}"

            # Validate numeric values
            if not (math.isfinite(t) and math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
                return [], f"Non-finite value encountered in sample {idx}: t={t}, x={x}, y={y}, yaw={yaw}"

            if t < -1e-6:
                return [], f"Negative time offset encountered in sample {idx}: t={t}"

            t = max(0.0, float(t))
            if idx > 0 and t < prev_t - 1e-6:
                return [], f"Non-monotonic timestamps in trajectory at index {idx}: t={t} < prev={prev_t}"

            prev_t = t
            samples.append(TrajectorySample(time_offset=t, x=float(x), y=float(y), yaw=float(yaw), v=float(v)))

        return samples, None

    # -------------------------------------------------------------------------
    # Waypoint Interpolation for Tracked Obstacles
    # -------------------------------------------------------------------------
    def _interpolate_track_at_time(
        self,
        track: PredictedObstacleTrajectory,
        target_t: float,
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Extract or linearly interpolate obstacle state at target_t.

        Returns:
            (x, y, vx, vy, uncertainty_radius, bbox_yaw) or None if target_t > track.horizon_s.
        """
        if target_t < -1e-6 or target_t > track.horizon_s + 1e-4:
            return None

        if not track.waypoints:
            return None

        # Check exact or nearest waypoint
        for wp in track.waypoints:
            if abs(wp.time_offset - target_t) < 1e-4:
                return (wp.x, wp.y, wp.vx, wp.vy, wp.uncertainty_radius, track.current_state.bbox_yaw)

        if target_t <= track.waypoints[0].time_offset:
            wp0 = track.waypoints[0]
            return (wp0.x, wp0.y, wp0.vx, wp0.vy, wp0.uncertainty_radius, track.current_state.bbox_yaw)

        if target_t >= track.waypoints[-1].time_offset:
            wp1 = track.waypoints[-1]
            return (wp1.x, wp1.y, wp1.vx, wp1.vy, wp1.uncertainty_radius, track.current_state.bbox_yaw)

        for i in range(len(track.waypoints) - 1):
            w0 = track.waypoints[i]
            w1 = track.waypoints[i + 1]
            if w0.time_offset <= target_t <= w1.time_offset:
                span = w1.time_offset - w0.time_offset
                if span < 1e-6:
                    return (w0.x, w0.y, w0.vx, w0.vy, w0.uncertainty_radius, track.current_state.bbox_yaw)
                ratio = (target_t - w0.time_offset) / span
                ix = w0.x + ratio * (w1.x - w0.x)
                iy = w0.y + ratio * (w1.y - w0.y)
                ivx = w0.vx + ratio * (w1.vx - w0.vx)
                ivy = w0.vy + ratio * (w1.vy - w0.vy)
                ir_unc = w0.uncertainty_radius + ratio * (w1.uncertainty_radius - w0.uncertainty_radius)
                return (ix, iy, ivx, ivy, ir_unc, track.current_state.bbox_yaw)

        return None

    # -------------------------------------------------------------------------
    # Core Trajectory Checking Logic
    # -------------------------------------------------------------------------
    def check_trajectory(
        self,
        trajectory: Any,
        dynamic_layer: DynamicOccupancyLayer,
        predicted_tracks: Optional[Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]]] = None,
    ) -> PredictiveCollisionResult:
        """Evaluate future UGV candidate trajectory against the dynamic occupancy layer.

        Args:
            trajectory: Sequence of future trajectory waypoints/samples.
            dynamic_layer: Spatiotemporal DynamicOccupancyLayer from Milestone 13.7.
            predicted_tracks: Optional sequence of PredictedObstacleTrajectory for exact
                geometric clearance calculation and track-level attribution.

        Returns:
            Structured PredictiveCollisionResult detailing collision, risk, clearance, and horizon status.

        Raises:
            ValueError: If input is invalid and config.raise_on_invalid is True.
        """
        samples, err_msg = self._normalize_trajectory(trajectory)
        if err_msg is not None:
            if self.config.raise_on_invalid:
                raise ValueError(f"Invalid trajectory input: {err_msg}")
            return PredictiveCollisionResult(
                collision=False,
                collision_time=None,
                collision_position=None,
                obstacle_id=None,
                minimum_clearance=0.0,
                maximum_cost=0.0,
                risk_classification=DynamicRiskLevel.FREE,
                unknown_horizon_encountered=False,
                samples_checked=0,
                reason=f"Invalid trajectory input: {err_msg}",
                status=CollisionStatus.INVALID_INPUT,
            )

        # Normalize predicted tracks list
        tracks_list: List[PredictedObstacleTrajectory] = []
        if predicted_tracks is not None:
            if isinstance(predicted_tracks, PredictedTrajectoryArray):
                tracks_list = list(predicted_tracks.predictions)
            else:
                tracks_list = list(predicted_tracks)

        # Check for EXPIRED tracks in input
        for trk in tracks_list:
            if trk.current_state.status.upper() == "EXPIRED":
                msg = f"Obstacle track {trk.track_id} is EXPIRED; expired tracks must be pruned before collision checking."
                if self.config.raise_on_invalid:
                    raise ValueError(msg)
                return PredictiveCollisionResult(
                    collision=False,
                    collision_time=None,
                    collision_position=None,
                    obstacle_id=trk.track_id,
                    minimum_clearance=0.0,
                    maximum_cost=0.0,
                    risk_classification=DynamicRiskLevel.FREE,
                    unknown_horizon_encountered=False,
                    samples_checked=0,
                    reason=msg,
                    status=CollisionStatus.INVALID_INPUT,
                )

        # Tracking state accumulators
        global_has_collision: bool = False
        earliest_collision_time: Optional[float] = None
        earliest_collision_pos: Optional[Tuple[float, float]] = None
        earliest_collision_obs_id: Optional[int] = None
        earliest_collision_reason: str = ""

        # Event tracking by severity
        earliest_high_risk: Optional[Tuple[float, Tuple[float, float], Optional[int], float]] = None
        earliest_med_risk: Optional[Tuple[float, Tuple[float, float], Optional[int], float]] = None
        earliest_low_risk: Optional[Tuple[float, Tuple[float, float], Optional[int], float]] = None
        earliest_unknown: Optional[Tuple[float, Tuple[float, float], Optional[int], str]] = None

        global_min_clearance: float = float(self.config.max_clearance_search_m)
        global_max_cost: float = 0.0
        global_highest_risk: DynamicRiskLevel = DynamicRiskLevel.FREE
        unknown_horizon_encountered: bool = False

        total_samples = len(samples)

        # Evaluate each trajectory sample
        for sample in samples:
            t_s = sample.time_offset
            x_s = sample.x
            y_s = sample.y
            yaw_s = sample.yaw

            # 1. Horizon checks
            sample_is_unknown: bool = False
            unknown_reason: str = ""

            # Check global prediction horizon
            if t_s > self.config.max_horizon_s + 1e-4:
                sample_is_unknown = True
                unknown_reason = (
                    f"Sample time t={t_s:.2f}s exceeds layer max horizon ({self.config.max_horizon_s:.2f}s)"
                )
                unknown_horizon_encountered = True

            # Check individual track horizons (e.g. LOST tracks capped at 1.0s)
            if self.config.check_track_horizons and tracks_list:
                for trk in tracks_list:
                    if t_s > trk.horizon_s + 1e-4:
                        sample_is_unknown = True
                        unknown_horizon_encountered = True
                        if not unknown_reason:
                            unknown_reason = (
                                f"Sample time t={t_s:.2f}s exceeds valid horizon ({trk.horizon_s:.2f}s) "
                                f"for track {trk.track_id} (status={trk.current_state.status})"
                            )

            if sample_is_unknown and earliest_unknown is None:
                earliest_unknown = (t_s, (x_s, y_s), None, unknown_reason)

            # 2. UGV Footprint geometry
            ugv_obb = compute_obb_corners(
                (x_s, y_s),
                (self.length, self.width),
                yaw_s,
                margin_m=self.margin,
            )

            # Check world boundary clipping
            if not dynamic_layer.is_in_bounds(x_s, y_s):
                global_has_collision = True
                if earliest_collision_time is None:
                    earliest_collision_time = t_s
                    earliest_collision_pos = (x_s, y_s)
                    earliest_collision_reason = f"UGV center ({x_s:.2f}, {y_s:.2f}) out of grid bounds at t={t_s:.2f}s"
                global_min_clearance = 0.0
                global_max_cost = 1.0
                global_highest_risk = DynamicRiskLevel.LETHAL
                continue

            # Check boundary clipping for all corners
            corner_out_of_bounds = False
            for ci in range(4):
                if not dynamic_layer.is_in_bounds(float(ugv_obb[ci, 0]), float(ugv_obb[ci, 1])):
                    corner_out_of_bounds = True
                    break
            if corner_out_of_bounds:
                global_has_collision = True
                if earliest_collision_time is None:
                    earliest_collision_time = t_s
                    earliest_collision_pos = (x_s, y_s)
                    earliest_collision_reason = f"UGV footprint corner clipped grid bounds at t={t_s:.2f}s"
                global_min_clearance = 0.0
                global_max_cost = 1.0
                global_highest_risk = DynamicRiskLevel.LETHAL
                continue

            # 3. Dynamic Occupancy Raster Evaluation
            # Sample UGV footprint points (corners, perimeter, interior)
            footprint_points = sample_ugv_footprint_points(
                x_s, y_s, yaw_s,
                self.length, self.width,
                margin_m=self.margin,
                perimeter_step_m=self.config.perimeter_step_m,
                interior_density=self.config.interior_grid_density,
            )

            # Query dynamic layer slice at t_s
            grid_slice = dynamic_layer.get_slice(
                t_s, conservative_union=self.config.conservative_sub_bin_union
            )

            sample_max_cost: float = 0.0
            sample_highest_risk: DynamicRiskLevel = DynamicRiskLevel.FREE
            sample_raster_nominal_hit: bool = False

            for pt in footprint_points:
                r, c = dynamic_layer.world_to_grid(float(pt[0]), float(pt[1]))
                if not dynamic_layer.is_grid_in_bounds(r, c):
                    sample_raster_nominal_hit = True
                    sample_max_cost = 1.0
                    sample_highest_risk = DynamicRiskLevel.LETHAL
                    break

                c_val = float(grid_slice.cost[r, c])
                r_val = DynamicRiskLevel(int(grid_slice.risk_level[r, c]))
                is_nom = bool(grid_slice.nominal_obstacle[r, c])

                if c_val > sample_max_cost:
                    sample_max_cost = c_val
                if r_val > sample_highest_risk:
                    sample_highest_risk = r_val
                if is_nom or c_val >= 1.0 or r_val == DynamicRiskLevel.LETHAL:
                    sample_raster_nominal_hit = True

            # 4. Exact Geometric Obstacle Track Evaluation (if tracks provided)
            sample_physical_collision: bool = False
            sample_colliding_obs_id: Optional[int] = None
            sample_min_clearance: float = float(self.config.max_clearance_search_m)

            if tracks_list:
                for trk in tracks_list:
                    wp_data = self._interpolate_track_at_time(trk, t_s)
                    if wp_data is None:
                        # Target time exceeds obstacle valid prediction horizon
                        continue

                    ox, oy, ovx, ovy, or_unc, oyaw = wp_data
                    obs_size = trk.current_state.bbox_size

                    # Obstacle nominal physical OBB (zero margin)
                    obs_obb = compute_obb_corners((ox, oy), obs_size, oyaw, margin_m=0.0)

                    # Quick bounding circle filter before exact polygon checks
                    obs_radius = math.hypot(obs_size[0] / 2.0, obs_size[1] / 2.0)
                    dist_centers = math.hypot(x_s - ox, y_s - oy)

                    if dist_centers > (self.r_circumscribed + obs_radius + self.config.max_clearance_search_m):
                        # Far away; cannot collide or affect clearance below search threshold
                        continue

                    # Exact Separating Axis Theorem (SAT) polygon intersection
                    is_sat_overlap = polygon_intersection_sat(ugv_obb, obs_obb)

                    if is_sat_overlap:
                        sample_physical_collision = True
                        sample_colliding_obs_id = trk.track_id
                        sample_min_clearance = 0.0
                        sample_max_cost = 1.0
                        sample_highest_risk = DynamicRiskLevel.LETHAL
                        break
                    else:
                        # Calculate exact metric geometric clearance
                        geom_dist = compute_polygon_clearance(ugv_obb, obs_obb)
                        if geom_dist < sample_min_clearance:
                            sample_min_clearance = geom_dist

                        # Evaluate uncertainty penetration
                        if geom_dist <= or_unc:
                            sigma_pos = max(1e-4, or_unc / 3.0)
                            if geom_dist <= sigma_pos:
                                # High-risk zone [0, 1*sigma]
                                if DynamicRiskLevel.HIGH_RISK > sample_highest_risk:
                                    sample_highest_risk = DynamicRiskLevel.HIGH_RISK
                                cost_est = 0.95 - 0.25 * (geom_dist / sigma_pos)
                                if cost_est > sample_max_cost:
                                    sample_max_cost = float(cost_est)
                                if earliest_high_risk is None or (earliest_high_risk[0] > t_s):
                                    earliest_high_risk = (t_s, (x_s, y_s), trk.track_id, sample_max_cost)
                            elif geom_dist <= 2.0 * sigma_pos:
                                # Medium-risk zone (1*sigma, 2*sigma]
                                if DynamicRiskLevel.MEDIUM_RISK > sample_highest_risk:
                                    sample_highest_risk = DynamicRiskLevel.MEDIUM_RISK
                                cost_est = 0.70 - 0.30 * ((geom_dist - sigma_pos) / sigma_pos)
                                if cost_est > sample_max_cost:
                                    sample_max_cost = float(cost_est)
                                if earliest_med_risk is None or (earliest_med_risk[0] > t_s):
                                    earliest_med_risk = (t_s, (x_s, y_s), trk.track_id, sample_max_cost)
                            else:
                                # Low-risk zone (2*sigma, 3*sigma]
                                if DynamicRiskLevel.LOW_RISK > sample_highest_risk:
                                    sample_highest_risk = DynamicRiskLevel.LOW_RISK
                                cost_est = 0.40 - 0.25 * ((geom_dist - 2.0 * sigma_pos) / sigma_pos)
                                if cost_est > sample_max_cost:
                                    sample_max_cost = float(cost_est)
                                if earliest_low_risk is None or (earliest_low_risk[0] > t_s):
                                    earliest_low_risk = (t_s, (x_s, y_s), trk.track_id, sample_max_cost)

                # Fallback: if no track matched SAT collision, check if layer contains untracked obstacles
                if not sample_physical_collision and sample_raster_nominal_hit:
                    tracked_ids = {trk.track_id for trk in tracks_list}
                    layer_ids = set(grid_slice.active_obstacle_ids)
                    untracked = layer_ids - tracked_ids
                    if untracked:
                        sample_physical_collision = True
                        sample_colliding_obs_id = list(untracked)[0]
                        sample_min_clearance = 0.0
            else:
                # If no explicit tracks provided, check active obstacles from layer slice
                if sample_raster_nominal_hit:
                    sample_physical_collision = True
                    sample_min_clearance = 0.0
                    sample_colliding_obs_id = (
                        grid_slice.active_obstacle_ids[0] if grid_slice.active_obstacle_ids else None
                    )

            # Accumulate sample outcomes into global trackers
            if sample_physical_collision:
                global_has_collision = True
                if earliest_collision_time is None or (earliest_collision_time > t_s):
                    earliest_collision_time = t_s
                    earliest_collision_pos = (x_s, y_s)
                    earliest_collision_obs_id = sample_colliding_obs_id
                    earliest_collision_reason = (
                        f"Predicted physical footprint collision with obstacle "
                        f"{sample_colliding_obs_id if sample_colliding_obs_id is not None else 'unknown'} "
                        f"at t={t_s:.2f}s"
                    )

            if sample_min_clearance < global_min_clearance:
                global_min_clearance = sample_min_clearance
            if sample_max_cost > global_max_cost:
                global_max_cost = sample_max_cost
            if sample_highest_risk > global_highest_risk:
                global_highest_risk = sample_highest_risk

            # Track risk events if not collision
            if not sample_physical_collision:
                if sample_highest_risk == DynamicRiskLevel.HIGH_RISK or sample_max_cost >= self.config.high_risk_cost_threshold:
                    if earliest_high_risk is None or (earliest_high_risk[0] > t_s):
                        obs_id_hint = sample_colliding_obs_id or (
                            grid_slice.active_obstacle_ids[0] if grid_slice.active_obstacle_ids else None
                        )
                        earliest_high_risk = (t_s, (x_s, y_s), obs_id_hint, sample_max_cost)
                elif sample_highest_risk == DynamicRiskLevel.MEDIUM_RISK or sample_max_cost >= self.config.medium_risk_cost_threshold:
                    if earliest_med_risk is None or (earliest_med_risk[0] > t_s):
                        obs_id_hint = sample_colliding_obs_id or (
                            grid_slice.active_obstacle_ids[0] if grid_slice.active_obstacle_ids else None
                        )
                        earliest_med_risk = (t_s, (x_s, y_s), obs_id_hint, sample_max_cost)
                elif sample_highest_risk == DynamicRiskLevel.LOW_RISK or sample_max_cost >= self.config.low_risk_cost_threshold:
                    if earliest_low_risk is None or (earliest_low_risk[0] > t_s):
                        obs_id_hint = sample_colliding_obs_id or (
                            grid_slice.active_obstacle_ids[0] if grid_slice.active_obstacle_ids else None
                        )
                        earliest_low_risk = (t_s, (x_s, y_s), obs_id_hint, sample_max_cost)

        # ---------------------------------------------------------------------
        # Final Decision Logic & Classification Arbitration
        # ---------------------------------------------------------------------
        if global_has_collision:
            # Lethal physical collision strictly takes precedence
            return PredictiveCollisionResult(
                collision=True,
                collision_time=earliest_collision_time,
                collision_position=earliest_collision_pos,
                obstacle_id=earliest_collision_obs_id,
                minimum_clearance=0.0,
                maximum_cost=float(max(1.0, global_max_cost)),
                risk_classification=DynamicRiskLevel.LETHAL,
                unknown_horizon_encountered=unknown_horizon_encountered,
                samples_checked=total_samples,
                reason=earliest_collision_reason,
                status=CollisionStatus.PREDICTED_COLLISION,
                details={"event_type": "physical_collision"},
            )

        # No physical collision: evaluate unknown horizon vs risk
        # Safety rule: Unknown horizon beyond valid prediction MUST NOT silently become SAFE/FREE
        if unknown_horizon_encountered:
            u_t, u_pos, u_id, u_reason = earliest_unknown or (0.0, (0.0, 0.0), None, "Unknown horizon")
            return PredictiveCollisionResult(
                collision=False,
                collision_time=u_t,
                collision_position=u_pos,
                obstacle_id=u_id,
                minimum_clearance=float(global_min_clearance),
                maximum_cost=float(global_max_cost),
                risk_classification=global_highest_risk,
                unknown_horizon_encountered=True,
                samples_checked=total_samples,
                reason=f"{u_reason}; prediction unavailable beyond valid horizon is UNKNOWN, not FREE",
                status=CollisionStatus.UNKNOWN_HORIZON,
                details={"event_type": "unknown_horizon", "unknown_time": u_t},
            )

        if global_highest_risk == DynamicRiskLevel.HIGH_RISK or global_max_cost >= self.config.high_risk_cost_threshold:
            h_t, h_pos, h_id, h_cost = earliest_high_risk or (0.0, (0.0, 0.0), None, global_max_cost)
            return PredictiveCollisionResult(
                collision=False,
                collision_time=h_t,
                collision_position=h_pos,
                obstacle_id=h_id,
                minimum_clearance=float(global_min_clearance),
                maximum_cost=float(global_max_cost),
                risk_classification=DynamicRiskLevel.HIGH_RISK,
                unknown_horizon_encountered=False,
                samples_checked=total_samples,
                reason=f"Predicted high-risk threshold penetration (cost={global_max_cost:.2f}) at t={h_t:.2f}s",
                status=CollisionStatus.PREDICTED_HIGH_RISK,
                details={"event_type": "high_risk", "risk_cost": global_max_cost},
            )

        if global_highest_risk == DynamicRiskLevel.MEDIUM_RISK or global_max_cost >= self.config.medium_risk_cost_threshold:
            m_t, m_pos, m_id, m_cost = earliest_med_risk or (0.0, (0.0, 0.0), None, global_max_cost)
            return PredictiveCollisionResult(
                collision=False,
                collision_time=m_t,
                collision_position=m_pos,
                obstacle_id=m_id,
                minimum_clearance=float(global_min_clearance),
                maximum_cost=float(global_max_cost),
                risk_classification=DynamicRiskLevel.MEDIUM_RISK,
                unknown_horizon_encountered=False,
                samples_checked=total_samples,
                reason=f"Predicted medium-risk intersection (cost={global_max_cost:.2f}) at t={m_t:.2f}s",
                status=CollisionStatus.PREDICTED_MEDIUM_RISK,
                details={"event_type": "medium_risk", "risk_cost": global_max_cost},
            )

        if global_highest_risk == DynamicRiskLevel.LOW_RISK or global_max_cost >= self.config.low_risk_cost_threshold:
            l_t, l_pos, l_id, l_cost = earliest_low_risk or (0.0, (0.0, 0.0), None, global_max_cost)
            return PredictiveCollisionResult(
                collision=False,
                collision_time=l_t,
                collision_position=l_pos,
                obstacle_id=l_id,
                minimum_clearance=float(global_min_clearance),
                maximum_cost=float(global_max_cost),
                risk_classification=DynamicRiskLevel.LOW_RISK,
                unknown_horizon_encountered=False,
                samples_checked=total_samples,
                reason=f"Predicted low-risk intersection (cost={global_max_cost:.2f}) at t={l_t:.2f}s",
                status=CollisionStatus.PREDICTED_LOW_RISK,
                details={"event_type": "low_risk", "risk_cost": global_max_cost},
            )

        # Clear trajectory within valid horizon
        return PredictiveCollisionResult(
            collision=False,
            collision_time=None,
            collision_position=None,
            obstacle_id=None,
            minimum_clearance=float(global_min_clearance),
            maximum_cost=float(global_max_cost),
            risk_classification=DynamicRiskLevel.FREE,
            unknown_horizon_encountered=False,
            samples_checked=total_samples,
            reason="Trajectory is predicted safe within valid dynamic prediction horizon",
            status=CollisionStatus.SAFE,
            details={"event_type": "safe"},
        )

    # -------------------------------------------------------------------------
    # Single Point Query Helper
    # -------------------------------------------------------------------------
    def check_point(
        self,
        x: float,
        y: float,
        yaw: float,
        time_offset: float,
        dynamic_layer: DynamicOccupancyLayer,
        predicted_tracks: Optional[Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]]] = None,
    ) -> PredictiveCollisionResult:
        """Check a single UGV pose and future time offset for predictive collision."""
        sample = TrajectorySample(time_offset=time_offset, x=x, y=y, yaw=yaw)
        return self.check_trajectory([sample], dynamic_layer, predicted_tracks)
