"""Mathematical data models and representations for dynamic obstacles in Phase 13.

Provides standardized data structures across detection, tracking, estimation,
prediction, and dynamic occupancy query interfaces:
- DynamicObstacleDetection: Single-frame detection bounding box and confidence
- TrackedObstacleState: Multi-frame filtered state with velocity and 4x4 covariance
- TrajectoryWaypoint: Future spatiotemporal waypoint with 2x2 spatial uncertainty
- PredictedObstacleTrajectory: Multi-horizon extrapolated trajectory and bounding hulls
- Container arrays for batched messaging and ROS2-style serialization
- Geometric and coordinate transformation utilities (base_link <-> odom)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def normalize_angle(angle_rad: float) -> float:
    """Normalize an angle to the interval [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def compute_obb_corners(
    center: Tuple[float, float],
    size: Tuple[float, float],
    yaw_rad: float,
    margin_m: float = 0.0,
) -> np.ndarray:
    """Calculate the 4 metric corners of an oriented bounding box (OBB).

    Matches the project convention in ai.planning.collision.FootprintCollisionChecker.
    Order of corners: [front-left, front-right, rear-right, rear-left].

    Args:
        center: (x, y) center position in meters.
        size: (length, width) dimensions along longitudinal and lateral axes in meters.
        yaw_rad: Heading angle in radians.
        margin_m: Safety expansion margin in meters added to half-dimensions.

    Returns:
        np.ndarray of shape (4, 2) containing metric corner coordinates.
    """
    length, width = size
    hl = (float(length) / 2.0) + float(margin_m)
    hw = (float(width) / 2.0) + float(margin_m)

    local_corners = np.array([
        [hl, hw],
        [hl, -hw],
        [-hl, -hw],
        [-hl, hw],
    ], dtype=np.float64)

    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)

    rotated = (R @ local_corners.T).T
    rotated[:, 0] += center[0]
    rotated[:, 1] += center[1]
    return rotated


def polygon_intersection_sat(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """Evaluate 2D convex polygon collision using the Separating Axis Theorem (SAT).

    Args:
        poly1: (N, 2) array of vertices in clockwise or counter-clockwise order.
        poly2: (M, 2) array of vertices in clockwise or counter-clockwise order.

    Returns:
        True if the two convex polygons overlap, False otherwise.
    """
    for polygon in (poly1, poly2):
        num_vertices = len(polygon)
        for i in range(num_vertices):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % num_vertices]
            edge = p2 - p1
            # Normal perpendicular vector (-dy, dx)
            normal = np.array([-edge[1], edge[0]], dtype=np.float64)
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal /= norm

            # Project both polygons onto the axis
            proj1 = poly1 @ normal
            proj2 = poly2 @ normal

            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)

            # Check for separation along this axis
            if max1 < min2 or max2 < min1:
                return False

    return True


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """Determine if a 2D point lies inside a 2D polygon using standard ray casting.

    Args:
        point: (x, y) query coordinate.
        polygon: (N, 2) array of polygon vertices.

    Returns:
        True if the point is strictly inside or on the polygon boundary.
    """
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        p1x, p1y = polygon[i]
        p2x, p2y = polygon[(i + 1) % n]

        # Check if point is on horizontal edge
        if p1y == p2y == y and min(p1x, p2x) <= x <= max(p1x, p2x):
            return True

        # Check ray crossing
        if min(p1y, p2y) < y <= max(p1y, p2y):
            x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if x <= x_inters:
                inside = not inside

    return inside


@dataclass
class DynamicObstacleDetection:
    """Standardized single-frame dynamic obstacle observation.

    Decoupled from perception source (optical flow, clustering, simulation sensor).
    """

    detection_id: str
    timestamp: float
    frame_id: str
    bbox_center: Tuple[float, float]
    bbox_size: Tuple[float, float]
    bbox_yaw: float
    confidence: float
    raw_cluster_points: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")
        if self.bbox_size[0] <= 0.0 or self.bbox_size[1] <= 0.0:
            raise ValueError(f"bbox_size dimensions must be positive, got {self.bbox_size}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")

        self.bbox_yaw = normalize_angle(self.bbox_yaw)
        self.bbox_center = (float(self.bbox_center[0]), float(self.bbox_center[1]))
        self.bbox_size = (float(self.bbox_size[0]), float(self.bbox_size[1]))

        if self.raw_cluster_points is not None:
            self.raw_cluster_points = np.asarray(self.raw_cluster_points, dtype=np.float64)
            if self.raw_cluster_points.ndim != 2 or self.raw_cluster_points.shape[1] != 2:
                raise ValueError(
                    f"raw_cluster_points must have shape (N, 2), got {self.raw_cluster_points.shape}"
                )

    def get_corners(self, margin_m: float = 0.0) -> np.ndarray:
        """Return (4, 2) corner array of the oriented bounding box."""
        return compute_obb_corners(self.bbox_center, self.bbox_size, self.bbox_yaw, margin_m=margin_m)

    def get_axis_aligned_bbox(self, margin_m: float = 0.0) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) bounding box enclosing the OBB."""
        corners = self.get_corners(margin_m=margin_m)
        return (
            float(np.min(corners[:, 0])),
            float(np.min(corners[:, 1])),
            float(np.max(corners[:, 0])),
            float(np.max(corners[:, 1])),
        )

    def transform_to_frame(
        self,
        target_frame: str,
        robot_pose: Tuple[float, float, float],
    ) -> DynamicObstacleDetection:
        """Transform detection between 'base_link' and 'odom' coordinates.

        Args:
            target_frame: Target coordinate frame name ('odom' or 'base_link').
            robot_pose: (robot_x, robot_y, robot_yaw_rad) in odom frame.

        Returns:
            New DynamicObstacleDetection instance in target_frame.
        """
        if self.frame_id == target_frame:
            return self

        rx, ry, ryaw = robot_pose
        cx, cy = self.bbox_center

        if self.frame_id == "base_link" and target_frame == "odom":
            c = math.cos(ryaw)
            s = math.sin(ryaw)
            new_cx = c * cx - s * cy + rx
            new_cy = s * cx + c * cy + ry
            new_yaw = normalize_angle(self.bbox_yaw + ryaw)

            new_cluster = None
            if self.raw_cluster_points is not None and len(self.raw_cluster_points) > 0:
                R = np.array([[c, -s], [s, c]], dtype=np.float64)
                new_cluster = (R @ self.raw_cluster_points.T).T
                new_cluster[:, 0] += rx
                new_cluster[:, 1] += ry

        elif self.frame_id == "odom" and target_frame == "base_link":
            dx = cx - rx
            dy = cy - ry
            c = math.cos(-ryaw)
            s = math.sin(-ryaw)
            new_cx = c * dx - s * dy
            new_cy = s * dx + c * dy
            new_yaw = normalize_angle(self.bbox_yaw - ryaw)

            new_cluster = None
            if self.raw_cluster_points is not None and len(self.raw_cluster_points) > 0:
                R = np.array([[c, -s], [s, c]], dtype=np.float64)
                shifted = self.raw_cluster_points.copy()
                shifted[:, 0] -= rx
                shifted[:, 1] -= ry
                new_cluster = (R @ shifted.T).T
        else:
            raise ValueError(
                f"Unsupported frame transformation from '{self.frame_id}' to '{target_frame}'"
            )

        return DynamicObstacleDetection(
            detection_id=self.detection_id,
            timestamp=self.timestamp,
            frame_id=target_frame,
            bbox_center=(new_cx, new_cy),
            bbox_size=self.bbox_size,
            bbox_yaw=new_yaw,
            confidence=self.confidence,
            raw_cluster_points=new_cluster,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize detection to a JSON-compatible dictionary."""
        return {
            "detection_id": self.detection_id,
            "timestamp": float(self.timestamp),
            "frame_id": str(self.frame_id),
            "bbox_center": [float(self.bbox_center[0]), float(self.bbox_center[1])],
            "bbox_size": [float(self.bbox_size[0]), float(self.bbox_size[1])],
            "bbox_yaw": float(self.bbox_yaw),
            "confidence": float(self.confidence),
            "raw_cluster_points": (
                self.raw_cluster_points.tolist()
                if self.raw_cluster_points is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DynamicObstacleDetection:
        """Construct detection from serialized dictionary."""
        cluster = d.get("raw_cluster_points")
        return cls(
            detection_id=str(d["detection_id"]),
            timestamp=float(d["timestamp"]),
            frame_id=str(d["frame_id"]),
            bbox_center=(float(d["bbox_center"][0]), float(d["bbox_center"][1])),
            bbox_size=(float(d["bbox_size"][0]), float(d["bbox_size"][1])),
            bbox_yaw=float(d["bbox_yaw"]),
            confidence=float(d["confidence"]),
            raw_cluster_points=np.array(cluster, dtype=np.float64) if cluster is not None else None,
        )


@dataclass
class TrackedObstacleState:
    """Multi-frame tracked obstacle state with filtered velocity and covariance.

    Maintained strictly in the world ('odom') frame to decouple tracking from robot egomotion.
    State vector: [x, y, vx, vy]^T.
    """

    track_id: int
    timestamp: float
    x: float
    y: float
    vx: float
    vy: float
    covariance: np.ndarray
    bbox_size: Tuple[float, float]
    bbox_yaw: float
    frame_id: str = "odom"
    track_age: int = 1
    coast_cycles: int = 0
    is_confirmed: bool = False
    status: str = "NEW"
    history: List[Tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError(f"track_id must be non-negative, got {self.track_id}")
        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")
        if self.bbox_size[0] <= 0.0 or self.bbox_size[1] <= 0.0:
            raise ValueError(f"bbox_size must be positive, got {self.bbox_size}")
        if self.track_age < 1:
            raise ValueError(f"track_age must be >= 1, got {self.track_age}")
        if self.coast_cycles < 0:
            raise ValueError(f"coast_cycles must be >= 0, got {self.coast_cycles}")

        self.bbox_yaw = normalize_angle(self.bbox_yaw)
        self.bbox_size = (float(self.bbox_size[0]), float(self.bbox_size[1]))
        self.x = float(self.x)
        self.y = float(self.y)
        self.vx = float(self.vx)
        self.vy = float(self.vy)
        self.status = str(self.status)

        # Validate covariance matrix
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        if self.covariance.shape != (4, 4):
            raise ValueError(f"covariance must have shape (4, 4), got {self.covariance.shape}")

        # Enforce symmetry and check positive semi-definiteness
        sym_cov = 0.5 * (self.covariance + self.covariance.T)
        self.covariance = sym_cov
        eigvals = np.linalg.eigvalsh(self.covariance)
        if np.any(eigvals < -1e-6):
            raise ValueError(
                f"covariance matrix must be positive semi-definite; minimum eigenvalue: {np.min(eigvals)}"
            )

        if not self.history:
            self.history = [(self.timestamp, self.x, self.y)]

    @property
    def speed(self) -> float:
        """Linear speed magnitude in m/s."""
        return math.hypot(self.vx, self.vy)

    @property
    def velocity_heading(self) -> float:
        """Course heading angle of velocity vector in radians [-pi, pi]."""
        return math.atan2(self.vy, self.vx)

    @property
    def position(self) -> Tuple[float, float]:
        """Current 2D position (x, y) in meters."""
        return (self.x, self.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current 2D velocity (vx, vy) in m/s."""
        return (self.vx, self.vy)

    def position_covariance(self) -> np.ndarray:
        """Return the 2x2 position covariance submatrix."""
        return self.covariance[:2, :2].copy()

    def velocity_covariance(self) -> np.ndarray:
        """Return the 2x2 velocity covariance submatrix."""
        return self.covariance[2:, 2:].copy()

    def position_uncertainty_radius(self, sigma_scale: float = 3.0) -> float:
        """Calculate the spatial uncertainty radius (semi-major axis) at given sigma scale.

        Args:
            sigma_scale: Confidence scale factor (e.g. 3.0 for ~99.7% confidence ellipse).

        Returns:
            Semi-major axis length in meters.
        """
        pos_cov = self.position_covariance()
        eigvals = np.linalg.eigvalsh(pos_cov)
        max_eigval = max(0.0, float(np.max(eigvals)))
        return float(sigma_scale * math.sqrt(max_eigval))

    def get_corners(self, margin_m: float = 0.0) -> np.ndarray:
        """Return (4, 2) corner array of the oriented bounding box in odom frame."""
        return compute_obb_corners(
            (self.x, self.y), self.bbox_size, self.bbox_yaw, margin_m=margin_m
        )

    def get_axis_aligned_bbox(self, margin_m: float = 0.0) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) bounding box enclosing the OBB."""
        corners = self.get_corners(margin_m=margin_m)
        return (
            float(np.min(corners[:, 0])),
            float(np.min(corners[:, 1])),
            float(np.max(corners[:, 0])),
            float(np.max(corners[:, 1])),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize track state to a JSON-compatible dictionary."""
        return {
            "track_id": int(self.track_id),
            "timestamp": float(self.timestamp),
            "frame_id": str(self.frame_id),
            "x": float(self.x),
            "y": float(self.y),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "covariance": self.covariance.tolist(),
            "bbox_size": [float(self.bbox_size[0]), float(self.bbox_size[1])],
            "bbox_yaw": float(self.bbox_yaw),
            "track_age": int(self.track_age),
            "coast_cycles": int(self.coast_cycles),
            "is_confirmed": bool(self.is_confirmed),
            "status": str(self.status),
            "history": [(float(t), float(hx), float(hy)) for (t, hx, hy) in self.history],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrackedObstacleState:
        """Construct track state from dictionary."""
        return cls(
            track_id=int(d["track_id"]),
            timestamp=float(d["timestamp"]),
            frame_id=str(d.get("frame_id", "odom")),
            x=float(d["x"]),
            y=float(d["y"]),
            vx=float(d["vx"]),
            vy=float(d["vy"]),
            covariance=np.array(d["covariance"], dtype=np.float64),
            bbox_size=(float(d["bbox_size"][0]), float(d["bbox_size"][1])),
            bbox_yaw=float(d["bbox_yaw"]),
            track_age=int(d.get("track_age", 1)),
            coast_cycles=int(d.get("coast_cycles", 0)),
            is_confirmed=bool(d.get("is_confirmed", False)),
            status=str(d.get("status", "NEW")),
            history=[(float(h[0]), float(h[1]), float(h[2])) for h in d.get("history", [])],
        )


@dataclass
class TrajectoryWaypoint:
    """Spatiotemporal waypoint along a forward-projected obstacle trajectory."""

    time_offset: float
    timestamp: float
    x: float
    y: float
    vx: float
    vy: float
    covariance_2d: np.ndarray
    uncertainty_radius: float

    def __post_init__(self) -> None:
        if self.time_offset < 0.0:
            raise ValueError(f"time_offset must be non-negative, got {self.time_offset}")
        if self.uncertainty_radius < 0.0:
            raise ValueError(f"uncertainty_radius must be non-negative, got {self.uncertainty_radius}")

        self.x = float(self.x)
        self.y = float(self.y)
        self.vx = float(self.vx)
        self.vy = float(self.vy)
        self.uncertainty_radius = float(self.uncertainty_radius)

        self.covariance_2d = np.asarray(self.covariance_2d, dtype=np.float64)
        if self.covariance_2d.shape != (2, 2):
            raise ValueError(f"covariance_2d must have shape (2, 2), got {self.covariance_2d.shape}")

        sym_cov = 0.5 * (self.covariance_2d + self.covariance_2d.T)
        self.covariance_2d = sym_cov
        eigvals = np.linalg.eigvalsh(self.covariance_2d)
        if np.any(eigvals < -1e-6):
            raise ValueError(
                f"covariance_2d must be positive semi-definite; min eigenvalue: {np.min(eigvals)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize waypoint to dictionary."""
        return {
            "time_offset": float(self.time_offset),
            "timestamp": float(self.timestamp),
            "x": float(self.x),
            "y": float(self.y),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "covariance_2d": self.covariance_2d.tolist(),
            "uncertainty_radius": float(self.uncertainty_radius),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrajectoryWaypoint:
        """Construct waypoint from dictionary."""
        return cls(
            time_offset=float(d["time_offset"]),
            timestamp=float(d["timestamp"]),
            x=float(d["x"]),
            y=float(d["y"]),
            vx=float(d["vx"]),
            vy=float(d["vy"]),
            covariance_2d=np.array(d["covariance_2d"], dtype=np.float64),
            uncertainty_radius=float(d["uncertainty_radius"]),
        )


@dataclass
class PredictedObstacleTrajectory:
    """Forward-extrapolated trajectory and dynamic collision hulls for an obstacle."""

    track_id: int
    current_state: TrackedObstacleState
    horizon_s: float
    dt: float
    waypoints: List[TrajectoryWaypoint]
    bounding_polygon_per_step: List[np.ndarray]

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError(f"track_id must be non-negative, got {self.track_id}")
        if self.horizon_s < 0.0:
            raise ValueError(f"horizon_s must be non-negative, got {self.horizon_s}")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if len(self.waypoints) != len(self.bounding_polygon_per_step):
            raise ValueError(
                f"waypoints length ({len(self.waypoints)}) must match bounding_polygon_per_step "
                f"length ({len(self.bounding_polygon_per_step)})"
            )

    def get_waypoint_at_offset(self, time_offset: float) -> Optional[TrajectoryWaypoint]:
        """Return the closest waypoint for a given future time offset."""
        if not self.waypoints or time_offset < 0.0 or time_offset > self.horizon_s + self.dt:
            return None
        idx = int(round(time_offset / self.dt))
        idx = max(0, min(len(self.waypoints) - 1, idx))
        return self.waypoints[idx]

    def get_polygon_at_offset(self, time_offset: float) -> Optional[np.ndarray]:
        """Return the bounding polygon for a given future time offset."""
        if not self.bounding_polygon_per_step or time_offset < 0.0 or time_offset > self.horizon_s + self.dt:
            return None
        idx = int(round(time_offset / self.dt))
        idx = max(0, min(len(self.bounding_polygon_per_step) - 1, idx))
        return self.bounding_polygon_per_step[idx]

    def interpolate_state_at(
        self,
        time_offset: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Linearly interpolate state (x, y, vx, vy) at an arbitrary time offset."""
        if not self.waypoints or time_offset < 0.0 or time_offset > self.horizon_s:
            return None

        if time_offset <= self.waypoints[0].time_offset:
            wp = self.waypoints[0]
            return (wp.x, wp.y, wp.vx, wp.vy)
        if time_offset >= self.waypoints[-1].time_offset:
            wp = self.waypoints[-1]
            return (wp.x, wp.y, wp.vx, wp.vy)

        for i in range(len(self.waypoints) - 1):
            w0 = self.waypoints[i]
            w1 = self.waypoints[i + 1]
            if w0.time_offset <= time_offset <= w1.time_offset:
                denom = w1.time_offset - w0.time_offset
                if denom < 1e-6:
                    return (w0.x, w0.y, w0.vx, w0.vy)
                ratio = (time_offset - w0.time_offset) / denom
                ix = w0.x + ratio * (w1.x - w0.x)
                iy = w0.y + ratio * (w1.y - w0.y)
                ivx = w0.vx + ratio * (w1.vx - w0.vx)
                ivy = w0.vy + ratio * (w1.vy - w0.vy)
                return (ix, iy, ivx, ivy)

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize predicted trajectory to dictionary."""
        return {
            "track_id": int(self.track_id),
            "current_state": self.current_state.to_dict(),
            "horizon_s": float(self.horizon_s),
            "dt": float(self.dt),
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "bounding_polygon_per_step": [poly.tolist() for poly in self.bounding_polygon_per_step],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PredictedObstacleTrajectory:
        """Construct predicted trajectory from dictionary."""
        return cls(
            track_id=int(d["track_id"]),
            current_state=TrackedObstacleState.from_dict(d["current_state"]),
            horizon_s=float(d["horizon_s"]),
            dt=float(d["dt"]),
            waypoints=[TrajectoryWaypoint.from_dict(wp) for wp in d["waypoints"]],
            bounding_polygon_per_step=[
                np.array(poly, dtype=np.float64) for poly in d["bounding_polygon_per_step"]
            ],
        )


@dataclass
class DynamicObstacleDetectionArray:
    """Array container for single-frame detections, matching ROS2 array message conventions."""

    header_timestamp: float
    frame_id: str
    detections: List[DynamicObstacleDetection] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.detections)

    def __getitem__(self, idx: int) -> DynamicObstacleDetection:
        return self.detections[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header_timestamp": float(self.header_timestamp),
            "frame_id": str(self.frame_id),
            "detections": [det.to_dict() for det in self.detections],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DynamicObstacleDetectionArray:
        return cls(
            header_timestamp=float(d["header_timestamp"]),
            frame_id=str(d["frame_id"]),
            detections=[DynamicObstacleDetection.from_dict(det) for det in d.get("detections", [])],
        )


@dataclass
class TrackedObstacleArray:
    """Array container for active multi-frame tracks in the world ('odom') frame."""

    header_timestamp: float
    frame_id: str = "odom"
    tracks: List[TrackedObstacleState] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> TrackedObstacleState:
        return self.tracks[idx]

    def get_track_by_id(self, track_id: int) -> Optional[TrackedObstacleState]:
        """Find active track with specified ID."""
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    def confirmed_tracks(self) -> List[TrackedObstacleState]:
        """Filter tracks that have achieved confirmed status."""
        return [t for t in self.tracks if t.is_confirmed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header_timestamp": float(self.header_timestamp),
            "frame_id": str(self.frame_id),
            "tracks": [t.to_dict() for t in self.tracks],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrackedObstacleArray:
        return cls(
            header_timestamp=float(d["header_timestamp"]),
            frame_id=str(d.get("frame_id", "odom")),
            tracks=[TrackedObstacleState.from_dict(t) for t in d.get("tracks", [])],
        )


@dataclass
class PredictedTrajectoryArray:
    """Array container for predicted trajectories across all active tracks."""

    header_timestamp: float
    frame_id: str = "odom"
    predictions: List[PredictedObstacleTrajectory] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.predictions)

    def __getitem__(self, idx: int) -> PredictedObstacleTrajectory:
        return self.predictions[idx]

    def get_prediction_by_id(self, track_id: int) -> Optional[PredictedObstacleTrajectory]:
        """Find prediction for track with specified ID."""
        for p in self.predictions:
            if p.track_id == track_id:
                return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header_timestamp": float(self.header_timestamp),
            "frame_id": str(self.frame_id),
            "predictions": [p.to_dict() for p in self.predictions],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PredictedTrajectoryArray:
        return cls(
            header_timestamp=float(d["header_timestamp"]),
            frame_id=str(d.get("frame_id", "odom")),
            predictions=[PredictedObstacleTrajectory.from_dict(p) for p in d.get("predictions", [])],
        )
