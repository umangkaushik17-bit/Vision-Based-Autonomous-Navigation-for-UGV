"""Abstract base interface for terrain-aware global path planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, Optional, Tuple

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import GoalState, PlanningConfig, PlannerStatus, RobotState
from ai.planning.path import Path
from ai.planning.validation import PathValidator


class GlobalPlanner(ABC):
    """Base interface for all global route planning algorithms."""

    def __init__(self, name: str, config: Optional[PlanningConfig] = None) -> None:
        self.name = name
        self.config = config or PlanningConfig()
        self.validator = PathValidator(
            vehicle=self.config.vehicle,
            max_safe_cost=float(self.config.tolerances.get("max_safe_cost", 0.65))
        )

    @abstractmethod
    def _search(
        self,
        costmap: BEVTerrainCostmap,
        start: RobotState,
        goal: GoalState
    ) -> Tuple[Optional[Path], PlannerStatus, Dict[str, Any]]:
        """Algorithm-specific graph or sampling search implementation."""
        raise NotImplementedError

    def plan(
        self,
        costmap: BEVTerrainCostmap,
        start: RobotState,
        goal: GoalState
    ) -> Tuple[Optional[Path], PlannerStatus, Dict[str, Any]]:
        """Execute start/goal validation, algorithm search, and post-search path validation."""
        t0 = time.perf_counter()

        # 1. Pre-validation
        is_valid, status, reason = self.validator.validate_start_goal(start, goal, costmap)
        if not is_valid:
            return None, status, {
                "planner": self.name,
                "planning_time_ms": (time.perf_counter() - t0) * 1000.0,
                "reason": reason,
                "nodes_explored": 0
            }

        # 2. Algorithm Search
        path, status, diagnostics = self._search(costmap, start, goal)
        t1 = time.perf_counter()
        diagnostics["planner"] = self.name
        diagnostics["planning_time_ms"] = round((t1 - t0) * 1000.0, 2)

        # 3. Post-validation if path found
        if path is not None and status == PlannerStatus.SUCCESS:
            valid, v_reason, v_diag = self.validator.validate_path(path, costmap)
            diagnostics.update(v_diag)
            if not valid:
                diagnostics["validation_failure_reason"] = v_reason
                return None, PlannerStatus.NO_PATH, diagnostics

        return path, status, diagnostics
