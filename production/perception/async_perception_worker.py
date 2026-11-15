"""Decoupled Asynchronous Perception & Attitude-IPM Worker (P1-3 / Defect #7).

Prevents synchronous neural network inference (12–110 ms) inside ROS 2 camera
subscription callbacks from blocking the 10 Hz DWA planning timer and 20 Hz
SafeCommandBuffer actuator dispatch timer on single-threaded executors.

Architectural Guarantees:
1. Bounded Depth-1 Latest-Frame Slot: Camera callback deposits the newest frame in O(1)
   time (< 0.1 ms) and returns immediately. Stale intermediate frames are overwritten
   rather than queued, eliminating perception lag buildup.
2. Off-Thread Inference & IPM Projection: Both neural segmentation (`predict`) and
   3D SE(3) attitude-compensated BEV projection (`project_traversability`) execute on
   the background worker thread.
3. Atomic Result Swap: The control/camera loop retrieves completed perception + BEV
   outputs in O(1) time (< 0.05 ms) via `pop_latest_result()`.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Optional, Tuple
import numpy as np

from ai.traversability.base import TraversabilityResult
from production.perception.vision_costmap_integrator import (
    LocalBEVCostmap,
    VisionCostmapIntegrator,
)


@dataclass
class PendingFrameSlot:
    """Single-slot latest camera frame input with synchronized robot pose & attitude."""

    rgb: np.ndarray
    timestamp_s: float
    ugv_x: float = 0.0
    ugv_y: float = 0.0
    ugv_yaw: float = 0.0
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    header: Any = None
    sequence_id: int = 0


@dataclass
class AsyncPerceptionOutput:
    """Completed perception and optional BEV projection output from background worker."""

    traversability_result: TraversabilityResult
    bev_costmap: Optional[LocalBEVCostmap]
    timestamp_s: float
    ugv_pose_xy_yaw: Tuple[float, float, float]
    roll_rad: float
    pitch_rad: float
    inference_ms: float
    projection_ms: float
    header: Any = None
    sequence_id: int = 0


class AsyncPerceptionWorker:
    """Non-blocking background worker for real-time traversability inference and IPM projection."""

    def __init__(
        self,
        predictor: Any,
        costmap_integrator: Optional[VisionCostmapIntegrator] = None,
        enable_bev_projection: bool = True,
    ) -> None:
        self.predictor = predictor
        self.costmap_integrator = costmap_integrator
        self.enable_bev_projection = enable_bev_projection

        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()

        self._pending_slot: Optional[PendingFrameSlot] = None
        self._latest_output: Optional[AsyncPerceptionOutput] = None
        self._submitted_seq: int = 0
        self._processed_count: int = 0
        self._dropped_stale_count: int = 0

        self._thread = threading.Thread(
            target=self._worker_loop,
            name="ugv_async_perception_worker",
            daemon=True,
        )
        self._thread.start()

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def dropped_stale_count(self) -> int:
        return self._dropped_stale_count

    def submit_frame(
        self,
        rgb: np.ndarray,
        timestamp_s: float,
        ugv_pose_xy_yaw: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        header: Any = None,
    ) -> int:
        """Deposit latest camera frame in O(1) time (< 0.1 ms) without blocking caller."""
        with self._lock:
            self._submitted_seq += 1
            if self._pending_slot is not None:
                self._dropped_stale_count += 1
            self._pending_slot = PendingFrameSlot(
                rgb=rgb,
                timestamp_s=float(timestamp_s),
                ugv_x=float(ugv_pose_xy_yaw[0]),
                ugv_y=float(ugv_pose_xy_yaw[1]),
                ugv_yaw=float(ugv_pose_xy_yaw[2]),
                roll_rad=float(roll_rad),
                pitch_rad=float(pitch_rad),
                header=header,
                sequence_id=self._submitted_seq,
            )
            seq = self._submitted_seq
        self._wake_event.set()
        return seq

    def pop_latest_result(self) -> Optional[AsyncPerceptionOutput]:
        """Atomically pop the most recent completed perception result (or None if no new result)."""
        with self._lock:
            out = self._latest_output
            self._latest_output = None
            return out

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=0.10)
            if self._stop_event.is_set():
                break

            with self._lock:
                slot = self._pending_slot
                self._pending_slot = None
                self._wake_event.clear()

            if slot is None or self.predictor is None:
                continue

            try:
                t0 = time.perf_counter()
                res = self.predictor.predict(slot.rgb)
                infer_ms = (time.perf_counter() - t0) * 1000.0

                bev: Optional[LocalBEVCostmap] = None
                proj_ms = 0.0
                if self.enable_bev_projection and self.costmap_integrator is not None and res is not None:
                    t_p0 = time.perf_counter()
                    bev = self.costmap_integrator.project_traversability(
                        res,
                        current_time_s=slot.timestamp_s,
                        roll_rad=slot.roll_rad,
                        pitch_rad=slot.pitch_rad,
                    )
                    proj_ms = (time.perf_counter() - t_p0) * 1000.0

                output = AsyncPerceptionOutput(
                    traversability_result=res,
                    bev_costmap=bev,
                    timestamp_s=slot.timestamp_s,
                    ugv_pose_xy_yaw=(slot.ugv_x, slot.ugv_y, slot.ugv_yaw),
                    roll_rad=slot.roll_rad,
                    pitch_rad=slot.pitch_rad,
                    inference_ms=infer_ms,
                    projection_ms=proj_ms,
                    header=slot.header,
                    sequence_id=slot.sequence_id,
                )
                with self._lock:
                    self._latest_output = output
                    self._processed_count += 1
            except Exception:
                pass

    def stop(self, timeout_s: float = 1.0) -> None:
        """Signal worker thread to exit cleanly."""
        self._stop_event.set()
        self._wake_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_s)
