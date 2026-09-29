"""Intel NPU (AI Boost / VPU) Backend Evaluator.

Documented outcome (Phase 5):
"NPU is not an appropriate backend for the current DWA computational structure."
DWA consists of non-uniform raycasts, SAT polygon projections, branch-heavy early exits,
and dynamic waypoint interpolation, which cannot be mapped efficiently to the Intel NPU.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

from ai.planning.backends.backend_interface import BackendHealth, DWABackend, DWABackendType
from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
)
from ai.planning.dynamic.dwa_filter import DynamicPlanningConfig
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory


class NPUDWABackend(DWABackend):
    """Intel NPU DWA backend stub (unsupported due to non-neural branching architecture)."""

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        super().__init__(DWABackendType.NPU)
        raise RuntimeError(
            "NPU backend is not supported: NPU is not an appropriate backend for the current DWA "
            "computational structure (branch-heavy polygon SAT collision checking and dynamic raycasts "
            "cannot be mapped to fixed-shape NPU neural matrix engines)."
        )

    @property
    def name(self) -> str:
        return "Intel NPU (Unsupported for DWA)"

    @property
    def device_name(self) -> str:
        return "Intel AI Boost NPU"

    @classmethod
    def is_available(cls) -> bool:
        # NPU hardware exists on laptop, but is technically unsuitable for DWA
        return False

    def check_health(self) -> BackendHealth:
        return BackendHealth(
            backend_type=self.backend_type,
            is_available=False,
            device_name=self.device_name,
            safety_validated=False,
            benchmark_latency_ms=None,
            error_message="NPU is not an appropriate backend for the current DWA computational structure.",
        )

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
        raise NotImplementedError("NPU cannot compute DWA trajectories.")
