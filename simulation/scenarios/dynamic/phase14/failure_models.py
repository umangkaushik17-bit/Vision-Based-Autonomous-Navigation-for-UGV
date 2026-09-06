"""Non-invasive fault injection interceptors for Phase 13.14 validation."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ai.planning.base import PlannerStatus
from ai.planning.trajectory import Trajectory, TrajectoryPoint
from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
)
from ai.planning.dynamic.global_replanner import DynamicReplanResult, DynamicReplanTrigger
from simulation.scenarios.dynamic.phase14.data_model import (
    Phase14FaultSpec,
    Phase14FaultType,
)


class Phase14FaultInjector:
    """Non-invasive adapter intercepting data at component boundaries outside frozen algorithms."""

    def __init__(self, fault_specs: Optional[List[Phase14FaultSpec]] = None, random_seed: int = 42) -> None:
        self.fault_specs: List[Phase14FaultSpec] = list(fault_specs) if fault_specs else []
        self.rng = np.random.RandomState(random_seed)
        self.fault_event_log: List[Dict[str, Any]] = []
        self._active_fault_ids: set[str] = set()

    def get_active_faults(self, timestamp_s: float) -> List[Phase14FaultSpec]:
        """Return all faults currently active at simulation timestamp."""
        active = [f for f in self.fault_specs if f.is_active(timestamp_s)]
        for f in active:
            if f.fault_id not in self._active_fault_ids:
                self._active_fault_ids.add(f.fault_id)
                self.fault_event_log.append({
                    "event": "FAULT_ACTIVATED",
                    "timestamp": timestamp_s,
                    "fault_id": f.fault_id,
                    "fault_type": f.fault_type.value,
                    "target_component": f.target_component,
                })
        # Check deactivated faults
        currently_active_ids = {f.fault_id for f in active}
        deactivated = self._active_fault_ids - currently_active_ids
        for fid in list(deactivated):
            self._active_fault_ids.remove(fid)
            self.fault_event_log.append({
                "event": "FAULT_DEACTIVATED",
                "timestamp": timestamp_s,
                "fault_id": fid,
            })
        return active

    def calculate_total_fault_duration(self) -> float:
        """Calculate union of active fault durations in seconds."""
        if not self.fault_specs:
            return 0.0
        intervals = sorted([(f.start_time, f.end_time) for f in self.fault_specs])
        merged = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return float(sum(end - start for start, end in merged))

    # --- 1. Perception Interceptors ---

    def intercept_observations(
        self,
        raw_detections: List[DynamicObstacleDetection],
        timestamp_s: float,
    ) -> List[DynamicObstacleDetection]:
        """Apply sensor dropouts, stale timestamps, or observation noise."""
        active = self.get_active_faults(timestamp_s)
        if not active:
            return raw_detections

        filtered_detections: List[DynamicObstacleDetection] = []

        # Find active perception faults
        dropout_fault = next((f for f in active if f.fault_type == Phase14FaultType.DROPOUT), None)
        stale_fault = next((f for f in active if f.fault_type == Phase14FaultType.STALE_TIMESTAMP), None)
        noise_fault = next((f for f in active if f.fault_type == Phase14FaultType.NOISY_OBSERVATIONS), None)

        for det in raw_detections:
            # Check dropout
            if dropout_fault is not None:
                prob = float(dropout_fault.parameters.get("dropout_probability", 1.0))
                target_obs = dropout_fault.parameters.get("obstacle_id", None)
                if target_obs is None or det.detection_id.startswith(str(target_obs)):
                    if self.rng.uniform(0.0, 1.0) < prob:
                        continue  # Dropped!

            # Check stale timestamp
            mod_det = det
            if stale_fault is not None:
                target_obs = stale_fault.parameters.get("obstacle_id", None)
                if target_obs is None or det.detection_id.startswith(str(target_obs)):
                    offset_s = float(stale_fault.parameters.get("offset_s", 2.0))
                    mod_det = DynamicObstacleDetection(
                        detection_id=det.detection_id,
                        timestamp=det.timestamp - offset_s,
                        frame_id=det.frame_id,
                        bbox_center=det.bbox_center,
                        bbox_size=det.bbox_size,
                        bbox_yaw=det.bbox_yaw,
                        confidence=det.confidence,
                    )

            # Check noise injection
            if noise_fault is not None:
                pos_std = float(noise_fault.parameters.get("pos_std", 0.15))
                conf_std = float(noise_fault.parameters.get("conf_std", 0.10))
                dx = float(self.rng.normal(0.0, pos_std))
                dy = float(self.rng.normal(0.0, pos_std))
                dconf = float(self.rng.normal(0.0, conf_std))
                new_conf = max(0.1, min(1.0, mod_det.confidence + dconf))
                new_center = (mod_det.bbox_center[0] + dx, mod_det.bbox_center[1] + dy)
                mod_det = DynamicObstacleDetection(
                    detection_id=mod_det.detection_id,
                    timestamp=mod_det.timestamp,
                    frame_id=mod_det.frame_id,
                    bbox_center=new_center,
                    bbox_size=mod_det.bbox_size,
                    bbox_yaw=mod_det.bbox_yaw,
                    confidence=new_conf,
                )

            filtered_detections.append(mod_det)

        return filtered_detections

    # --- 2. Tracking Interceptors ---

    def intercept_tracks(
        self,
        confirmed_tracks: List[TrackedObstacleState],
        timestamp_s: float,
    ) -> List[TrackedObstacleState]:
        """Apply track loss or tracker state clearing."""
        active = self.get_active_faults(timestamp_s)
        track_loss_fault = next((f for f in active if f.fault_type == Phase14FaultType.TRACK_LOSS), None)
        track_reset_fault = next((f for f in active if f.fault_type == Phase14FaultType.TRACK_RESET), None)

        if track_reset_fault is not None:
            return []

        if track_loss_fault is not None:
            target_track = track_loss_fault.parameters.get("track_id", None)
            if target_track is None:
                return []
            return [t for t in confirmed_tracks if str(t.track_id) != str(target_track)]

        return confirmed_tracks

    # --- 3. Prediction Interceptors ---

    def intercept_predictions(
        self,
        active_predictions: List[PredictedObstacleTrajectory],
        timestamp_s: float,
    ) -> List[PredictedObstacleTrajectory]:
        """Apply prediction dropouts, horizon truncation, or confidence degradation."""
        active = self.get_active_faults(timestamp_s)
        if not active:
            return active_predictions

        pred_dropout = next((f for f in active if f.fault_type == Phase14FaultType.PREDICTION_DROPOUT), None)
        pred_shorten = next((f for f in active if f.fault_type == Phase14FaultType.PREDICTION_HORIZON_SHORTEN), None)
        pred_degrade = next((f for f in active if f.fault_type == Phase14FaultType.PREDICTION_CONFIDENCE_DEGRADE), None)

        if pred_dropout is not None:
            target_id = pred_dropout.parameters.get("track_id", None)
            if target_id is None:
                return []
            return [p for p in active_predictions if str(p.track_id) != str(target_id)]

        modified: List[PredictedObstacleTrajectory] = []
        for pred in active_predictions:
            cur_waypoints = list(pred.waypoints)
            cur_polys = list(pred.bounding_polygon_per_step)
            cur_horizon = pred.horizon_s

            if pred_shorten is not None:
                max_pts = int(pred_shorten.parameters.get("max_points", 3))
                cur_waypoints = cur_waypoints[:max_pts]
                cur_polys = cur_polys[:max_pts]
                cur_horizon = min(cur_horizon, len(cur_waypoints) * pred.dt)

            if pred_degrade is not None:
                rad_mult = float(pred_degrade.parameters.get("radius_multiplier", 3.0))
                for i, wp in enumerate(cur_waypoints):
                    cur_waypoints[i] = copy.copy(wp)
                    cur_waypoints[i].uncertainty_radius *= rad_mult

            modified.append(PredictedObstacleTrajectory(
                track_id=pred.track_id,
                current_state=pred.current_state,
                horizon_s=cur_horizon,
                dt=pred.dt,
                waypoints=cur_waypoints,
                bounding_polygon_per_step=cur_polys,
            ))

        return modified

    # --- 4. Global Replanner Interceptors ---

    def intercept_replan_execution(
        self,
        replan_call: Callable[[], DynamicReplanResult],
        timestamp_s: float,
    ) -> DynamicReplanResult:
        """Simulate replanner failures (no path, exception, invalid path, timeout)."""
        active = self.get_active_faults(timestamp_s)
        no_path = next((f for f in active if f.fault_type == Phase14FaultType.REPLANNER_NO_PATH), None)
        exc_fault = next((f for f in active if f.fault_type == Phase14FaultType.REPLANNER_EXCEPTION), None)
        inv_fault = next((f for f in active if f.fault_type == Phase14FaultType.REPLANNER_INVALID_RESULT), None)
        timeout_fault = next((f for f in active if f.fault_type == Phase14FaultType.REPLANNER_TIMEOUT), None)

        if no_path is not None:
            return DynamicReplanResult(
                replan_triggered=True,
                trigger_type=DynamicReplanTrigger.PERSISTENT_BLOCKAGE,
                success=False,
                status=PlannerStatus.NO_PATH,
                path=None,
                is_path_stale=True,
                failure_reason="Injected Fault: Global replanner found no path",
            )

        if exc_fault is not None:
            return DynamicReplanResult(
                replan_triggered=True,
                trigger_type=DynamicReplanTrigger.MANUAL,
                success=False,
                status=PlannerStatus.NO_PATH,
                path=None,
                is_path_stale=True,
                failure_reason="Injected Fault: Global replanner internal solver failure",
            )

        if timeout_fault is not None:
            return DynamicReplanResult(
                replan_triggered=True,
                trigger_type=DynamicReplanTrigger.MANUAL,
                success=False,
                status=PlannerStatus.TIMEOUT,
                path=None,
                is_path_stale=True,
                failure_reason="Injected Fault: Global replanner request timed out",
            )

        res = replan_call()

        if inv_fault is not None:
            return DynamicReplanResult(
                replan_triggered=res.replan_triggered,
                trigger_type=res.trigger_type,
                success=False,
                status=PlannerStatus.NO_PATH,
                path=None,
                is_path_stale=True,
                failure_reason="Injected Fault: Replan result corrupted or invalid",
            )

        return res

    # --- 5. Local Planner / DWA Interceptors ---

    def intercept_dwa_output(
        self,
        cand_traj: Optional[Trajectory],
        cand_v: float,
        cand_w: float,
        planner_status: PlannerStatus,
        dwa_diag: Dict[str, Any],
        timestamp_s: float,
    ) -> Tuple[Optional[Trajectory], float, float, PlannerStatus, Dict[str, Any]]:
        """Inject zero valid trajectories or malformed planner outputs."""
        active = self.get_active_faults(timestamp_s)
        zero_valid = next((f for f in active if f.fault_type == Phase14FaultType.ZERO_VALID_TRAJECTORIES), None)
        malformed = next((f for f in active if f.fault_type == Phase14FaultType.MALFORMED_PLANNER_OUTPUT), None)

        if zero_valid is not None:
            diag_mod = dict(dwa_diag)
            diag_mod["valid_candidates"] = 0
            diag_mod["rejected_candidates"] = diag_mod.get("total_evaluated", 375)
            diag_mod["status"] = "ALL_COLLIDE"
            return None, 0.0, 0.0, PlannerStatus.NO_PATH, diag_mod

        if malformed is not None:
            # Inject malformed candidate with non-finite or out-of-bounds trajectory
            nan_traj = Trajectory(
                points=[TrajectoryPoint(t=0.0, x=float("nan"), y=float("nan"), v=float("nan"), omega=0.0)]
            )
            return nan_traj, float("nan"), float("nan"), PlannerStatus.SUCCESS, dwa_diag

        return cand_traj, cand_v, cand_w, planner_status, dwa_diag
