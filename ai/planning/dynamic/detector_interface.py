"""Dynamic obstacle detector interface and deterministic synthetic detector for Phase 13.

Provides an abstract perception interface decoupled from:
- tracker
- Kalman filter
- predictor
- dynamic occupancy
- DWA / A* planners
- safety supervisor
- ROS2 / Gazebo

Canonical contract:
    detector.detect(frame, timestamp_s) -> List[DynamicObstacleDetection]

All outputs strictly conform to the Phase 13.1 DynamicObstacleDetection data model.
Ground-truth isolation is strictly maintained: runtime `detect()` calls never consume
ground-truth poses, future velocities, or ground-truth track IDs.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    compute_obb_corners,
    normalize_angle,
)


class BaseDynamicObstacleDetector(abc.ABC):
    """Abstract base class for all dynamic obstacle detectors.

    Decouples sensor perception backends (YOLO, stereo depth, optical flow,
    semantic discrepancy, or synthetic fixtures) from downstream tracking.
    """

    def __init__(
        self,
        frame_id: str = "base_link",
        min_confidence: float = 0.30,
    ) -> None:
        """Initialize base detector parameters.

        Args:
            frame_id: Target coordinate frame of the generated detections (e.g., 'base_link' or 'odom').
            min_confidence: Minimum confidence threshold [0.0, 1.0] for emitting detections.
        """
        if not frame_id:
            raise ValueError("frame_id must be a non-empty string")
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0.0, 1.0], got {min_confidence}")

        self.frame_id = frame_id
        self.min_confidence = min_confidence

    @abc.abstractmethod
    def detect(
        self,
        frame: Any,
        timestamp_s: float,
    ) -> List[DynamicObstacleDetection]:
        """Detect dynamic obstacles from a single perception frame.

        Args:
            frame: Sensor perception data (RGB image, depth map, point cloud, feature map,
                   or None for synthetic simulation fixtures).
            timestamp_s: Observation timestamp in seconds (non-negative).

        Returns:
            List of standardized DynamicObstacleDetection objects.
            Must be empty if no dynamic obstacles are detected or if inputs are below threshold.
        """
        pass

    def filter_detections(
        self,
        detections: List[DynamicObstacleDetection],
    ) -> List[DynamicObstacleDetection]:
        """Filter detections against minimum confidence and sanity constraints."""
        valid: List[DynamicObstacleDetection] = []
        for det in detections:
            if det.confidence >= self.min_confidence:
                valid.append(det)
        return valid


@dataclass
class SyntheticObstacleScript:
    """Offline scenario specification for deterministic synthetic obstacle generation.

    Configured during simulation/fixture setup; NOT accessible to runtime planner or tracker.
    """

    obstacle_id: str
    size: Tuple[float, float]  # (length, width) in meters
    position_fn: Callable[[float], Optional[Tuple[float, float]]]
    yaw_fn: Optional[Callable[[float], float]] = None
    confidence: float = 0.95
    active_time_range: Tuple[float, float] = (0.0, float("inf"))
    measurement_noise_std: float = 0.0
    dropout_probability: float = 0.0
    generate_cluster_points: bool = False
    cluster_point_count: int = 16

    def __post_init__(self) -> None:
        if self.size[0] <= 0.0 or self.size[1] <= 0.0:
            raise ValueError(f"size dimensions must be positive, got {self.size}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not (0.0 <= self.dropout_probability <= 1.0):
            raise ValueError(f"dropout_probability must be in [0.0, 1.0], got {self.dropout_probability}")
        if self.measurement_noise_std < 0.0:
            raise ValueError(f"measurement_noise_std must be non-negative, got {self.measurement_noise_std}")


class SyntheticDynamicObstacleDetector(BaseDynamicObstacleDetector):
    """Deterministic synthetic detector for unit testing and SIL simulation benchmarks.

    Generates realistic, noisy, or dropping DynamicObstacleDetection observations
    based on offline scenario scripts.

    Enforces strict Ground-Truth Isolation:
    - Runtime `detect()` only takes `(frame, timestamp_s)`
    - Generates ephemeral, unassociated detection IDs (e.g. 'det_frame001_0')
    - Emits NO ground-truth track IDs or future velocities
    """

    def __init__(
        self,
        frame_id: str = "base_link",
        min_confidence: float = 0.30,
        random_seed: Optional[int] = 42,
    ) -> None:
        super().__init__(frame_id=frame_id, min_confidence=min_confidence)
        self._scripts: List[SyntheticObstacleScript] = []
        self._frame_counter: int = 0
        self._rng = np.random.default_rng(random_seed)

    def add_obstacle_script(self, script: SyntheticObstacleScript) -> None:
        """Register an obstacle motion script for testing."""
        self._scripts.append(script)

    def clear_obstacle_scripts(self) -> None:
        """Remove all obstacle motion scripts."""
        self._scripts.clear()

    def set_seed(self, seed: Optional[int]) -> None:
        """Set or reset the random number generator seed for deterministic replay."""
        self._rng = np.random.default_rng(seed)

    def detect(
        self,
        frame: Any,
        timestamp_s: float,
    ) -> List[DynamicObstacleDetection]:
        """Produce standardized DynamicObstacleDetection objects at timestamp_s.

        Args:
            frame: Sensor perception payload (can be None or metadata dict).
            timestamp_s: Observation timestamp in seconds (must be >= 0.0).

        Returns:
            List of DynamicObstacleDetection instances.
        """
        if timestamp_s < 0.0:
            raise ValueError(f"timestamp_s must be non-negative, got {timestamp_s}")

        self._frame_counter += 1
        detections: List[DynamicObstacleDetection] = []

        for idx, script in enumerate(self._scripts):
            t_start, t_end = script.active_time_range
            if not (t_start <= timestamp_s <= t_end):
                continue

            # Simulate sensor dropout
            if script.dropout_probability > 0.0:
                if self._rng.random() < script.dropout_probability:
                    continue

            # Evaluate scripted position
            pos = script.position_fn(timestamp_s)
            if pos is None:
                continue

            cx, cy = pos
            if script.measurement_noise_std > 0.0:
                cx += float(self._rng.normal(0.0, script.measurement_noise_std))
                cy += float(self._rng.normal(0.0, script.measurement_noise_std))

            # Evaluate heading
            if script.yaw_fn is not None:
                yaw = script.yaw_fn(timestamp_s)
            else:
                yaw = 0.0

            # Generate synthetic cluster points if requested
            cluster_pts = None
            if script.generate_cluster_points and script.cluster_point_count > 0:
                l, w = script.size
                local_xs = self._rng.uniform(-l / 2.0, l / 2.0, script.cluster_point_count)
                local_ys = self._rng.uniform(-w / 2.0, w / 2.0, script.cluster_point_count)
                c_yaw = math.cos(yaw)
                s_yaw = math.sin(yaw)
                rot_xs = c_yaw * local_xs - s_yaw * local_ys + cx
                rot_ys = s_yaw * local_xs + c_yaw * local_ys + cy
                cluster_pts = np.column_stack([rot_xs, rot_ys])

            # Generate an ephemeral detection ID (strictly unassociated with GT track ID)
            detection_id = f"det_{self._frame_counter:06d}_{idx:02d}"

            det = DynamicObstacleDetection(
                detection_id=detection_id,
                timestamp=float(timestamp_s),
                frame_id=self.frame_id,
                bbox_center=(float(cx), float(cy)),
                bbox_size=script.size,
                bbox_yaw=float(normalize_angle(yaw)),
                confidence=float(script.confidence),
                raw_cluster_points=cluster_pts,
            )
            detections.append(det)

        return self.filter_detections(detections)


# ==============================================================================
# Future Vision Detector Extension Interfaces (Stubs / Protocols)
# ==============================================================================

class VisionDynamicObstacleDetector(BaseDynamicObstacleDetector):
    """Base class for future computer-vision object detectors (YOLO, Stereo, Semantic).

    Enables plug-and-play replacement of the synthetic detector with neural network
    or depth-discrepancy backends without changing any tracker or planner interfaces.
    """

    def __init__(
        self,
        model_weights_path: Optional[str] = None,
        frame_id: str = "base_link",
        min_confidence: float = 0.50,
        device: str = "cpu",
    ) -> None:
        super().__init__(frame_id=frame_id, min_confidence=min_confidence)
        self.model_weights_path = model_weights_path
        self.device = device
        self._is_initialized = False

    def initialize_model(self) -> None:
        """Load vision model weights and prepare inference backend."""
        self._is_initialized = True

    def detect(
        self,
        frame: Any,
        timestamp_s: float,
    ) -> List[DynamicObstacleDetection]:
        """Inference stub to be populated when deep models are deployed."""
        if not self._is_initialized:
            self.initialize_model()

        if frame is None:
            return []

        # Future integration point for YOLO / Stereo Bounding Box inference
        # Emits DynamicObstacleDetection objects in frame_id ("base_link")
        return []
