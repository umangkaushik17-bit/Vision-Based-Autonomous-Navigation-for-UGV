"""Hardware-Adaptive Production DWA Planner.

Uses Native C++ DWA as the authoritative production planner (with SIMD or Scalar CPU,
or CUDA / Intel GPU where empirically validated).
Python DWA is strictly quarantined as a reference/regression implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from ai.planning.backends.unified_planner import UnifiedDWAPlanner
from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
)
from ai.planning.dynamic.dwa_filter import DynamicPlanningConfig
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory
from production.runtime.auto_selector import AutoBackendSelector, SelectionResult

logger = logging.getLogger("production.planning.adaptive_planner")


class AdaptiveDWAPlanner:
    """Production Hardware-Adaptive DWA Local Planner."""

    def __init__(
        self,
        config: Optional[PlanningConfig] = None,
        backend: str = "AUTO",
        selector: Optional[AutoBackendSelector] = None,
    ) -> None:
        self.config = config or PlanningConfig()
        self.requested_backend = backend.upper().strip()
        self.selector = selector or AutoBackendSelector()

        # Run empirical AUTO calibration or validate explicit backend
        self.selection_result: SelectionResult = self.selector.select_dwa_backend(
            requested=self.requested_backend
        )
        self.active_backend_name = self.selection_result.selected_name
        self.benchmark_latency_ms = self.selection_result.benchmark_latency_ms

        logger.info(
            f"Initializing AdaptiveDWAPlanner with validated backend '{self.active_backend_name}' "
            f"(benchmark: {self.benchmark_latency_ms:.2f} ms)"
        )
        self._planner = UnifiedDWAPlanner(
            config=self.config,
            backend=self.active_backend_name,
        )

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
        """Compute optimal collision-free trajectory using the selected backend."""
        return self._planner.plan(
            current_state=current_state,
            global_path=global_path,
            costmap=costmap,
            is_degraded=is_degraded,
            dynamic_layer=dynamic_layer,
            predicted_tracks=predicted_tracks,
            dynamic_config=dynamic_config,
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
        """Alias to plan() for backwards-compatibility."""
        return self.plan(
            current_state=current_state,
            global_path=global_path,
            costmap=costmap,
            is_degraded=is_degraded,
            dynamic_layer=dynamic_layer,
            predicted_tracks=predicted_tracks,
            dynamic_config=dynamic_config,
        )

    @property
    def backend_name(self) -> str:
        return self._planner.backend_name

    @property
    def device_name(self) -> str:
        return self._planner.device_name

    def get_selection_info(self) -> Dict[str, Any]:
        return {
            "active_backend": self.active_backend_name,
            "device": self.device_name,
            "benchmark_latency_ms": self.benchmark_latency_ms,
            "selection_reason": self.selection_result.selection_reason,
        }
