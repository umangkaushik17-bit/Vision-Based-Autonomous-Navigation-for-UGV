"""Unified Hardware-Independent Dynamic Window Approach (DWA) Local Planner.

Coordinates backend selection (CPU, CUDA, INTEL_GPU, NPU, AUTO), enforces deterministic
explicit backend selection (with clear failure on missing hardware), performs verified
AUTO backend arbitration, and guarantees safe runtime fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch

from ai.planning.backends.backend_interface import (
    BackendHealth,
    DWABackend,
    DWABackendType,
)
from ai.planning.backends.cpu_backend import CPUDWABackend
from ai.planning.backends.cpp_backend import (
    CPPDWABackend,
    CPPSimdDWABackend,
    CPPScalarDWABackend,
)
from ai.planning.backends.cuda_backend import CUDADWABackend
from ai.planning.backends.intel_gpu_backend import IntelGPUDWABackend
from ai.planning.backends.npu_backend import NPUDWABackend
from ai.planning.backends.vectorized_cpu_backend import VectorizedCPUDWABackend
from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralStateMachine,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
)
from ai.planning.dynamic.dwa_filter import DynamicPlanningConfig
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.local_planners.base import LocalPlanner
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory, TrajectoryPoint

logger = logging.getLogger("UnifiedDWAPlanner")


class UnifiedDWAPlanner(LocalPlanner):
    """Hardware-independent DWA planner supporting CPU, CUDA, Intel GPU, and AUTO selection."""

    def __init__(
        self,
        config: Optional[PlanningConfig] = None,
        backend: Union[str, DWABackendType] = DWABackendType.AUTO,
    ) -> None:
        super().__init__(name="UnifiedDWA", config=config)

        if isinstance(backend, str):
            self.requested_backend_type = DWABackendType.from_str(backend)
        else:
            self.requested_backend_type = backend

        # Behavioral state machine arbitration
        behavior_cfg = getattr(self.config, "behavior", None)
        self.behavioral_state_machine = BehavioralStateMachine(behavior_cfg)
        self.enable_behavioral_arbitration = bool(getattr(self.config, "enable_behavioral_arbitration", False))

        # Initialize active backend
        self.active_backend: DWABackend = self._select_backend(self.requested_backend_type)
        self.fallback_backend: DWABackend = CPUDWABackend(self.config)

    def _select_backend(self, requested: DWABackendType) -> DWABackend:
        """Resolve backend based on explicit selection or AUTO qualification."""
        if requested == DWABackendType.CPU:
            return CPUDWABackend(self.config)

        elif requested == DWABackendType.VECTORIZED_CPU:
            return VectorizedCPUDWABackend(self.config)

        elif requested == DWABackendType.CUDA:
            if not CUDADWABackend.is_available():
                raise RuntimeError(
                    "CUDA backend explicitly requested, but NVIDIA CUDA is unavailable on this system "
                    "(torch.cuda.is_available() returned False). System will NOT silently switch to CPU."
                )
            return CUDADWABackend(self.config)

        elif requested == DWABackendType.INTEL_GPU:
            if not IntelGPUDWABackend.is_available():
                raise RuntimeError(
                    "INTEL_GPU backend explicitly requested, but Intel GPU acceleration runtime "
                    "(torch.xpu or torch-directml) is unavailable on this environment. "
                    "System will NOT silently switch to CPU."
                )
            return IntelGPUDWABackend(self.config)

        elif requested == DWABackendType.CPP_SIMD:
            if not CPPSimdDWABackend.is_available():
                raise RuntimeError(
                    "CPP_SIMD backend explicitly requested, but compiled native C++ SIMD library "
                    "(AVX2) is unavailable or unsupported on this CPU. System will NOT silently switch to CPU."
                )
            return CPPSimdDWABackend(self.config)

        elif requested == DWABackendType.CPP_SCALAR:
            if not CPPScalarDWABackend.is_available():
                raise RuntimeError(
                    "CPP_SCALAR backend explicitly requested, but compiled native C++ scalar library "
                    "(dwa_cpp_scalar.dll or libdwa_cpp_scalar.so) is unavailable. System will NOT silently switch to CPU."
                )
            return CPPScalarDWABackend(self.config)

        elif requested == DWABackendType.CPP_CPU:
            if not CPPDWABackend.is_available():
                raise RuntimeError(
                    "CPP_CPU backend explicitly requested, but compiled native C++ library "
                    "(dwa_cpp.dll or libdwa_cpp.so) is unavailable. System will NOT silently switch to CPU."
                )
            return CPPDWABackend(self.config)

        elif requested == DWABackendType.NPU:
            # Explicit NPU request fails clearly with Phase 5 documentation
            raise RuntimeError(
                "NPU backend is not supported: NPU is not an appropriate backend for the current DWA "
                "computational structure (branch-heavy polygon SAT collision checking and dynamic raycasts "
                "cannot be mapped to fixed-shape NPU neural matrix engines)."
            )

        elif requested == DWABackendType.AUTO:
            return self._arbitrate_auto_backend()

        raise ValueError(f"Unsupported backend type: {requested}")

    def _arbitrate_auto_backend(self) -> DWABackend:
        """Arbitrate fastest VALIDATED backend with health verification."""
        # 1. NVIDIA CUDA: fastest validated backend if GPU hardware present
        if CUDADWABackend.is_available():
            try:
                cuda_be = CUDADWABackend(self.config)
                health = cuda_be.check_health()
                if health.is_available and health.safety_validated:
                    logger.info("AUTO mode selected validated NVIDIA CUDA backend.")
                    return cuda_be
            except Exception as e:
                logger.warning("CUDA available but initialization health check failed: %s", e)

        # 2. Native C++ SIMD (AVX2/OpenMP) or Scalar Fallback
        if CPPSimdDWABackend.is_available():
            try:
                cpp_simd_be = CPPSimdDWABackend(self.config)
                health = cpp_simd_be.check_health()
                if health.is_available and health.safety_validated:
                    logger.info("AUTO mode selected validated Native C++ SIMD backend (AVX2/OpenMP).")
                    return cpp_simd_be
            except Exception as e:
                logger.warning("Native C++ SIMD backend check failed: %s", e)

        if CPPScalarDWABackend.is_available():
            try:
                cpp_scalar_be = CPPScalarDWABackend(self.config)
                health = cpp_scalar_be.check_health()
                if health.is_available and health.safety_validated:
                    logger.info("AUTO mode selected validated Native C++ Scalar backend (Portable).")
                    return cpp_scalar_be
            except Exception as e:
                logger.warning("Native C++ Scalar backend check failed: %s", e)

        # 3. Intel GPU: accelerated iGPU/Arc if runtime available and validated
        if IntelGPUDWABackend.is_available():
            try:
                intel_be = IntelGPUDWABackend(self.config)
                health = intel_be.check_health()
                if health.is_available and health.safety_validated:
                    logger.info("AUTO mode selected validated Intel GPU backend.")
                    return intel_be
            except Exception as e:
                logger.warning("Intel GPU available but initialization failed: %s", e)

        # 4. Vectorized CPU: PyTorch batched tensor evaluation on host CPU
        try:
            vec_be = VectorizedCPUDWABackend(self.config)
            health = vec_be.check_health()
            if health.is_available and health.safety_validated:
                logger.info("AUTO mode selected validated Vectorized CPU backend (PyTorch SIMD).")
                return vec_be
        except Exception as e:
            logger.warning("Vectorized CPU backend check failed: %s", e)

        # 5. Canonical CPU Reference: infallible reference fallback
        logger.info("AUTO mode selected canonical Reference CPU backend.")
        return CPUDWABackend(self.config)

    @property
    def backend_type(self) -> DWABackendType:
        return self.active_backend.backend_type

    @property
    def backend_name(self) -> str:
        return self.active_backend.name

    @property
    def device_name(self) -> str:
        return self.active_backend.device_name

    def compute_trajectory(
        self,
        current_state: RobotState,
        global_path: Path,
        costmap: Any,
        is_degraded: bool = False,
        dynamic_layer: Optional[DynamicOccupancyLayer] = None,
        predicted_tracks: Optional[Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]]] = None,
        dynamic_config: Optional[DynamicPlanningConfig] = None,
    ) -> Tuple[Optional[Trajectory], float, float, PlannerStatus, Dict[str, Any]]:
        """Compute DWA trajectory with safe execution and AUTO fallback."""
        try:
            traj, v, w, status, diag = self.active_backend.compute_trajectory(
                current_state=current_state,
                global_path=global_path,
                costmap=costmap,
                is_degraded=is_degraded,
                dynamic_layer=dynamic_layer,
                predicted_tracks=predicted_tracks,
                dynamic_config=dynamic_config,
            )
            diag["active_backend"] = self.active_backend.name
            diag["active_device"] = self.active_backend.device_name
            diag["requested_backend"] = self.requested_backend_type.value
            return traj, v, w, status, diag

        except Exception as exc:
            logger.error("DWA backend '%s' encountered runtime failure: %s", self.active_backend.name, exc)

            # In explicit mode, FAIL SAFELY and trigger STOP
            if self.requested_backend_type != DWABackendType.AUTO:
                zero_traj = Trajectory(
                    points=[TrajectoryPoint(
                        t=current_state.timestamp + 0.10,
                        x=current_state.x, y=current_state.y, yaw=current_state.yaw,
                        v=0.0, omega=0.0, cost=1.0, clearance_m=0.0
                    )],
                    is_valid=False, total_cost=1.0, min_clearance_m=0.0
                )
                fail_diag = {
                    "active_backend": self.active_backend.name,
                    "requested_backend": self.requested_backend_type.value,
                    "runtime_error": str(exc),
                    "emergency_stop_triggered": True,
                }
                return zero_traj, 0.0, 0.0, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, fail_diag

            # In AUTO mode, safely fall back to the validated reference CPU backend
            logger.warning("AUTO mode safely falling back to canonical Reference CPU backend.")
            traj, v, w, status, diag = self.fallback_backend.compute_trajectory(
                current_state=current_state,
                global_path=global_path,
                costmap=costmap,
                is_degraded=is_degraded,
                dynamic_layer=dynamic_layer,
                predicted_tracks=predicted_tracks,
                dynamic_config=dynamic_config,
            )
            diag["active_backend"] = f"{self.active_backend.name} -> Fallback to Reference CPU"
            diag["active_device"] = self.fallback_backend.device_name
            diag["requested_backend"] = self.requested_backend_type.value
            diag["fallback_triggered"] = True
            return traj, v, w, status, diag

    def plan_with_behavior(
        self,
        current_state: RobotState,
        global_path: Path,
        costmap: Any,
        target_v: Optional[float] = None,
        dynamic_layer: Optional[DynamicOccupancyLayer] = None,
        predicted_tracks: Optional[Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]]] = None,
        dynamic_config: Optional[DynamicPlanningConfig] = None,
        behavioral_config: Optional[BehavioralConfig] = None,
    ) -> Tuple[Trajectory, float, float, PlannerStatus, Dict[str, Any], BehavioralDecision]:
        """Compute trajectory and arbitrate command through BehavioralStateMachine."""
        if behavioral_config is not None:
            self.behavioral_state_machine.config = behavioral_config

        traj, raw_v, raw_w, status, diag = self.compute_trajectory(
            current_state=current_state,
            global_path=global_path,
            costmap=costmap,
            dynamic_layer=dynamic_layer,
            predicted_tracks=predicted_tracks,
            dynamic_config=dynamic_config,
        )

        decision = self.behavioral_state_machine.evaluate(
            current_state=current_state,
            best_trajectory=traj,
            best_v=raw_v,
            best_w=raw_w,
            planner_status=status,
            dwa_diagnostics=diag,
            predicted_tracks=predicted_tracks,
            global_path=global_path,
            current_time_s=current_state.timestamp,
        )
        diag["behavioral_state"] = decision.state.value
        diag["behavioral_decision"] = decision.to_dict()
        diag["deadlock_event"] = decision.deadlock_event

        return traj, decision.command_v, decision.command_omega, status, diag, decision

    def plan(
        self,
        current_state: RobotState,
        global_path: Path,
        costmap: Any,
        is_degraded: bool = False,
        dynamic_layer: Optional[DynamicOccupancyLayer] = None,
        predicted_tracks: Optional[Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]]] = None,
        dynamic_config: Optional[DynamicPlanningConfig] = None,
    ) -> Tuple[Optional[Trajectory], float, float, PlannerStatus, Dict[str, Any]]:
        """Canonical planning entry point (alias to compute_trajectory)."""
        return self.compute_trajectory(
            current_state=current_state,
            global_path=global_path,
            costmap=costmap,
            is_degraded=is_degraded,
            dynamic_layer=dynamic_layer,
            predicted_tracks=predicted_tracks,
            dynamic_config=dynamic_config,
        )
