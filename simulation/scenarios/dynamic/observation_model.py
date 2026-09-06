"""Sensor observation generator synthesizing realistic, noisy, unassociated dynamic detections."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ai.planning.dynamic.data_model import DynamicObstacleDetection, normalize_angle
from simulation.scenarios.dynamic.data_model import GroundTruthObstacleState
from simulation.scenarios.dynamic.motion_models import BaseObstacleMotionModel


class DynamicObservationGenerator:
    """Generates synthetic sensor observations from ground truth with strict isolation.

    Enforces:
    - Zero leakage of ground-truth IDs or true velocities.
    - Ephemeral, unassociated detection IDs (e.g. 'det_f0012_0').
    - Configurable zero-mean Gaussian measurement noise.
    - Stochastic or scheduled detection dropout.
    - Deterministic random number generator state.
    """

    def __init__(
        self,
        motion_models: List[BaseObstacleMotionModel],
        frame_id: str = "odom",
        max_detection_range_m: float = 15.0,
        random_seed: int = 42,
    ) -> None:
        self.motion_models = list(motion_models)
        self.frame_id = frame_id
        self.max_detection_range_m = float(max_detection_range_m)
        self.random_seed = int(random_seed)
        self._rng = np.random.default_rng(self.random_seed)
        self._frame_index = 0

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Reset the PRNG seed for deterministic replay."""
        s = self.random_seed if seed is None else int(seed)
        self._rng = np.random.default_rng(s)
        self._frame_index = 0

    def generate_observations(
        self,
        timestamp_s: float,
        robot_pose: Optional[Tuple[float, float, float]] = None,
        force_dropout_all: bool = False,
    ) -> List[DynamicObstacleDetection]:
        """Generate perception detections from active obstacles at timestamp_s.

        Args:
            timestamp_s: Current simulation timestamp.
            robot_pose: Optional (x, y, yaw) of UGV for sensor range filtering.
            force_dropout_all: Failure injection flag to simulate complete sensor blackout.

        Returns:
            List of unassociated DynamicObstacleDetection objects.
        """
        self._frame_index += 1
        if force_dropout_all:
            return []

        detections: List[DynamicObstacleDetection] = []
        det_idx = 0

        for model in self.motion_models:
            gt_state: Optional[GroundTruthObstacleState] = model.evaluate(timestamp_s)
            if gt_state is None or not gt_state.is_active:
                continue

            # Sensor maximum detection range check
            if robot_pose is not None:
                rx, ry, _ = robot_pose
                dist = math.hypot(gt_state.x - rx, gt_state.y - ry)
                if dist > self.max_detection_range_m:
                    continue

            # Scheduled contiguous dropout check
            if any(t_start <= timestamp_s <= t_end for t_start, t_end in model.spec.dropout_windows):
                continue

            # Stochastic sensor dropout check
            p_drop = model.spec.dropout_probability
            if p_drop > 0.0 and self._rng.random() < p_drop:
                continue

            # Apply measurement noise
            noise_std = model.spec.measurement_noise_std
            meas_x = gt_state.x
            meas_y = gt_state.y
            if noise_std > 0.0:
                meas_x += float(self._rng.normal(0.0, noise_std))
                meas_y += float(self._rng.normal(0.0, noise_std))

            # Timestamp calculation (supports deliberate stale timestamp injection)
            det_timestamp = max(0.0, float(timestamp_s - model.spec.stale_timestamp_offset_s))

            # Ephemeral detection ID (strictly unassociated)
            ephemeral_id = f"det_f{self._frame_index:04d}_{det_idx}"
            det_idx += 1

            det = DynamicObstacleDetection(
                detection_id=ephemeral_id,
                timestamp=det_timestamp,
                frame_id=self.frame_id,
                bbox_center=(meas_x, meas_y),
                bbox_size=(gt_state.length_m, gt_state.width_m),
                bbox_yaw=normalize_angle(gt_state.yaw),
                confidence=float(model.spec.confidence),
            )
            detections.append(det)

        return detections
