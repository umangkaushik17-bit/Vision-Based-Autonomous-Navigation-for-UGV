"""Temporal synchronization and motion-compensated pose interpolation for spatial fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from collections import deque
import numpy as np

from ai.localization.base import TrackingState
from ai.localization.result import LocalizationEstimate
from ai.spatial_fusion.transforms import euler_to_rotation_matrix, rotation_matrix_to_euler, create_se3_matrix


@dataclass
class SyncResult:
    """Result of attempting to associate a perception observation timestamp with a 6-DoF pose."""

    is_valid: bool
    pose: Optional[np.ndarray] = None           # Compensated 4x4 SE(3) world-to-body matrix
    tracking_state: TrackingState = TrackingState.LOST
    confidence: float = 0.0
    time_delta_s: float = 0.0                   # t_obs - t_pose (seconds)
    is_extrapolated: bool = False
    source_estimate: Optional[LocalizationEstimate] = None
    reason: str = "ok"


class TemporalSynchronizer:
    """Deterministic timestamp association and motion compensation engine.
    
    Buffers recent LocalizationEstimate updates and matches perception observations
    with bounded temporal extrapolation to prevent stale pose propagation.
    """

    def __init__(
        self,
        max_sync_delta_s: float = 0.10,
        max_extrapolation_s: float = 0.05,
        buffer_size: int = 100
    ) -> None:
        self.max_sync_delta_s = max_sync_delta_s
        self.max_extrapolation_s = max_extrapolation_s
        self.buffer: Deque[LocalizationEstimate] = deque(maxlen=buffer_size)

        # Performance counters
        self.sync_queries = 0
        self.sync_successes = 0
        self.sync_rejected_gap = 0
        self.sync_rejected_untrusted = 0

    def add_localization_estimate(self, estimate: LocalizationEstimate) -> None:
        """Add incoming localization pose estimate to the temporal buffer."""
        # Enforce temporal monotonic order if buffer exists
        if self.buffer and estimate.timestamp < self.buffer[-1].timestamp:
            # Handle out-of-order by inserting into sorted position
            temp_list = list(self.buffer)
            temp_list.append(estimate)
            temp_list.sort(key=lambda e: e.timestamp)
            self.buffer = deque(temp_list, maxlen=self.buffer.maxlen)
        else:
            self.buffer.append(estimate)

    def synchronize(self, obs_timestamp: float) -> SyncResult:
        """Find the nearest localization estimate for an observation timestamp and apply motion compensation."""
        self.sync_queries += 1

        if not self.buffer:
            self.sync_rejected_gap += 1
            return SyncResult(is_valid=False, reason="empty_localization_buffer")

        # Find closest estimate in time
        best_est: Optional[LocalizationEstimate] = None
        best_dt = float("inf")

        for est in self.buffer:
            dt = abs(obs_timestamp - est.timestamp)
            if dt < best_dt:
                best_dt = dt
                best_est = est

        if best_est is None:
            self.sync_rejected_gap += 1
            return SyncResult(is_valid=False, reason="no_candidate_pose")

        signed_dt = obs_timestamp - best_est.timestamp

        # 1. Zero-Trust Check on Localization State
        if not best_est.allows_spatial_fusion or best_est.trusted_pose is None:
            self.sync_rejected_untrusted += 1
            return SyncResult(
                is_valid=False,
                tracking_state=best_est.tracking_state,
                confidence=best_est.confidence,
                time_delta_s=signed_dt,
                source_estimate=best_est,
                reason=f"untrusted_localization_state_{best_est.tracking_state.value}"
            )

        # 2. Check maximum allowable temporal gap
        if abs(signed_dt) > self.max_sync_delta_s:
            self.sync_rejected_gap += 1
            return SyncResult(
                is_valid=False,
                tracking_state=best_est.tracking_state,
                confidence=best_est.confidence,
                time_delta_s=signed_dt,
                source_estimate=best_est,
                reason=f"timestamp_gap_exceeded_{abs(signed_dt):.4f}s"
            )

        # 3. Motion Compensation (interpolation / short extrapolation)
        raw_pose = best_est.trusted_pose.copy()
        compensated_pose = raw_pose.copy()
        is_extrapolated = False

        if abs(signed_dt) > 1e-4:
            if abs(signed_dt) <= self.max_extrapolation_s:
                # Apply velocity-based motion projection:
                # dt_clamped = signed_dt
                dt = signed_dt
                v_body = best_est.linear_velocity
                w_body = best_est.angular_velocity

                # Position compensation in world frame:
                # p_world_new = p_world_old + R_world_body @ (v_body * dt)
                R_wb = raw_pose[:3, :3]
                dp_world = R_wb @ (v_body * dt)
                compensated_pose[:3, 3] += dp_world

                # Orientation compensation:
                # delta_yaw = w_body[2] * dt
                if abs(w_body[2]) > 1e-4:
                    roll, pitch, yaw = rotation_matrix_to_euler(R_wb)
                    compensated_R = euler_to_rotation_matrix(roll, pitch, yaw + w_body[2] * dt)
                    compensated_pose[:3, :3] = compensated_R

                is_extrapolated = True
            else:
                # Delta exceeds extrapolation limit, use raw pose without extrapolation
                is_extrapolated = False

        self.sync_successes += 1
        return SyncResult(
            is_valid=True,
            pose=compensated_pose,
            tracking_state=best_est.tracking_state,
            confidence=best_est.confidence,
            time_delta_s=signed_dt,
            is_extrapolated=is_extrapolated,
            source_estimate=best_est,
            reason="ok"
        )
