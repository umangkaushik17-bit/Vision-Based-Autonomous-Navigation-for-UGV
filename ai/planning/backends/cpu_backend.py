"""Reference CPU DWA Backend.

Wraps the canonical DWAPlanner implementation, preserving reference safety semantics,
dynamic obstacle evaluation, and behavioral state machine arbitration.
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
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory


class CPUDWABackend(DWABackend):
    """Canonical CPU reference backend for DWA planning."""

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        super().__init__(DWABackendType.CPU)
        self.config = config or PlanningConfig()
        self.planner = DWAPlanner(self.config)

    @property
    def name(self) -> str:
        return "Reference CPU"

    @property
    def device_name(self) -> str:
        return "Host CPU (Reference Python/NumPy)"

    def is_available(self) -> bool:
        return True

    def check_health(self) -> BackendHealth:
        return BackendHealth(
            backend_type=self.backend_type,
            is_available=True,
            device_name=self.device_name,
            safety_validated=True,
            benchmark_latency_ms=2165.8,
            error_message=None,
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
        return self.planner.compute_trajectory(
            current_state=current_state,
            global_path=global_path,
            costmap=costmap,
            is_degraded=is_degraded,
            dynamic_layer=dynamic_layer,
            predicted_tracks=predicted_tracks,
            dynamic_config=dynamic_config,
        )
