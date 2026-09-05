"""Abstract base interface for real-time local trajectory planners and path followers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory


class LocalPlanner(ABC):
    """Abstract interface for local trajectory selection and velocity command generation."""

    def __init__(self, name: str, config: Optional[PlanningConfig] = None) -> None:
        self.name = name
        self.config = config or PlanningConfig()

    @abstractmethod
    def compute_trajectory(
        self,
        current_state: RobotState,
        global_path: Path,
        costmap: BEVTerrainCostmap
    ) -> Tuple[Optional[Trajectory], float, float, PlannerStatus, Dict[str, Any]]:
        """Compute the best local trajectory and desired (v, omega) command.

        Returns:
            (best_trajectory, desired_linear_v, desired_angular_omega, status, diagnostics)
        """
        raise NotImplementedError
