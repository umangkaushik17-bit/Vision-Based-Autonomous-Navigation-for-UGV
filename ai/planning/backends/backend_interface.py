"""Hardware-independent DWA backend abstraction and common result interfaces.

Defines the contract for CPU, NVIDIA CUDA, Intel GPU, and experimental NPU backends,
ensuring 100% mathematical, behavioral, and safety interface compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ai.planning.base import PlannerStatus, RobotState
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
)
from ai.planning.dynamic.dwa_filter import DynamicPlanningConfig
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory


class DWABackendType(str, Enum):
    """Enumeration of supported DWA acceleration backends."""

    CPU = "CPU"
    CUDA = "CUDA"
    INTEL_GPU = "INTEL_GPU"
    NPU = "NPU"
    AUTO = "AUTO"
    VECTORIZED_CPU = "VECTORIZED_CPU"
    CPP_CPU = "CPP_CPU"
    CPP_SIMD = "CPP_SIMD"
    CPP_SCALAR = "CPP_SCALAR"

    @classmethod
    def from_str(cls, val: str) -> DWABackendType:
        normalized = val.strip().upper()
        if normalized in ("INTEL", "INTEL_GPU", "IGPU", "XPU"):
            return cls.INTEL_GPU
        elif normalized in ("NVIDIA", "CUDA"):
            return cls.CUDA
        elif normalized in ("NPU", "INTEL_NPU", "VPU"):
            return cls.NPU
        elif normalized in ("CPU", "REFERENCE_CPU"):
            return cls.CPU
        elif normalized in ("VECTORIZED_CPU", "VEC_CPU", "TORCH_CPU"):
            return cls.VECTORIZED_CPU
        elif normalized in ("CPP_SIMD", "C++_SIMD", "SIMD_CPU", "AVX2"):
            return cls.CPP_SIMD
        elif normalized in ("CPP_SCALAR", "C++_SCALAR", "SCALAR_CPU", "PORTABLE_CPU"):
            return cls.CPP_SCALAR
        elif normalized in ("CPP", "CPP_CPU", "NATIVE_CPP", "C++"):
            return cls.CPP_CPU
        elif normalized in ("AUTO", "AUTOMATIC"):
            return cls.AUTO
        raise ValueError(f"Unknown DWA backend type: '{val}'. Supported: CPU, CUDA, INTEL_GPU, NPU, AUTO, VECTORIZED_CPU, CPP_CPU, CPP_SIMD, CPP_SCALAR")


@dataclass
class BackendHealth:
    """Backend availability and safety qualification record."""

    backend_type: DWABackendType
    is_available: bool
    device_name: str
    safety_validated: bool
    benchmark_latency_ms: Optional[float] = None
    error_message: Optional[str] = None


class DWABackend(ABC):
    """Abstract base class for all hardware-specific DWA planning backends."""

    def __init__(self, backend_type: DWABackendType) -> None:
        self.backend_type = backend_type

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @property
    @abstractmethod
    def device_name(self) -> str:
        """Physical device or runtime identifier."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Query if hardware and required software runtime are present."""
        ...

    @abstractmethod
    def check_health(self) -> BackendHealth:
        """Perform backend health and parity validation check."""
        ...

    @abstractmethod
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
        """Compute DWA trajectory and score candidate velocities.

        Returns:
            (best_trajectory, cmd_v, cmd_w, planner_status, diagnostics_dict)
        """
        ...
