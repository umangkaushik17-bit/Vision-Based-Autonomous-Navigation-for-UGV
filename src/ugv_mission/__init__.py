"""Mission subsystem: High-level mission state machine, waypoint sequencing, and recovery orchestration."""

from __future__ import annotations

from ugv_mission.mission_executive import MissionExecutive, MissionExecutiveConfig
from ugv_mission.mission_state import (
    MissionAbortReason,
    MissionState,
    MissionStatus,
    is_valid_mission_transition,
)
from ugv_mission.recovery_orchestrator import (
    MissionRecoveryOrchestrator,
    RecoveryOrchestratorConfig,
)
from ugv_mission.waypoint import Waypoint
from ugv_mission.waypoint_manager import WaypointManager

__version__ = "0.2.0"

__all__ = [
    "Waypoint",
    "WaypointManager",
    "MissionState",
    "MissionAbortReason",
    "MissionStatus",
    "is_valid_mission_transition",
    "MissionRecoveryOrchestrator",
    "RecoveryOrchestratorConfig",
    "MissionExecutive",
    "MissionExecutiveConfig",
]
