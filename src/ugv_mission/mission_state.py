"""Mission lifecycle states, data contracts, and transition invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Set, Tuple


class MissionState(str, Enum):
    """Core lifecycle states for the autonomous UGV mission executive."""

    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    NAVIGATING_TO_WAYPOINT = "NAVIGATING_TO_WAYPOINT"
    WAYPOINT_REACHED = "WAYPOINT_REACHED"
    RECOVERY = "RECOVERY"
    REPLANNING = "REPLANNING"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    MISSION_ABORTED = "MISSION_ABORTED"


class MissionAbortReason(str, Enum):
    """Categorized root causes for mission abort."""

    NONE = "NONE"
    INVALID_MISSION = "INVALID_MISSION"
    EMPTY_WAYPOINTS = "EMPTY_WAYPOINTS"
    RECOVERY_LIMIT_EXCEEDED = "RECOVERY_LIMIT_EXCEEDED"
    RECOVERY_PERMANENTLY_BLOCKED = "RECOVERY_PERMANENTLY_BLOCKED"
    PLANNER_FAILURE = "PLANNER_FAILURE"
    EMERGENCY_STOP_TRIGGERED = "EMERGENCY_STOP_TRIGGERED"
    USER_CANCELLED = "USER_CANCELLED"
    TIMEOUT_EXCEEDED = "TIMEOUT_EXCEEDED"
    CRITICAL_ERROR = "CRITICAL_ERROR"


# Explicitly permitted state transitions
VALID_MISSION_TRANSITIONS: Dict[MissionState, Set[MissionState]] = {
    MissionState.IDLE: {
        MissionState.INITIALIZING,
        MissionState.MISSION_ABORTED,
    },
    MissionState.INITIALIZING: {
        MissionState.NAVIGATING_TO_WAYPOINT,
        MissionState.MISSION_ABORTED,
        MissionState.IDLE,
    },
    MissionState.NAVIGATING_TO_WAYPOINT: {
        MissionState.WAYPOINT_REACHED,
        MissionState.RECOVERY,
        MissionState.REPLANNING,
        MissionState.MISSION_ABORTED,
        MissionState.IDLE,
    },
    MissionState.WAYPOINT_REACHED: {
        MissionState.NAVIGATING_TO_WAYPOINT,
        MissionState.MISSION_COMPLETE,
        MissionState.MISSION_ABORTED,
        MissionState.IDLE,
    },
    MissionState.RECOVERY: {
        MissionState.REPLANNING,
        MissionState.NAVIGATING_TO_WAYPOINT,
        MissionState.MISSION_ABORTED,
        MissionState.IDLE,
    },
    MissionState.REPLANNING: {
        MissionState.NAVIGATING_TO_WAYPOINT,
        MissionState.RECOVERY,
        MissionState.MISSION_ABORTED,
        MissionState.IDLE,
    },
    MissionState.MISSION_COMPLETE: {
        MissionState.IDLE,  # Requires explicit reset
    },
    MissionState.MISSION_ABORTED: {
        MissionState.IDLE,  # Requires explicit reset
    },
}


def is_valid_mission_transition(from_state: MissionState, to_state: MissionState) -> bool:
    """Check if the transition between two mission states is strictly allowed."""
    if from_state == to_state:
        return True
    allowed = VALID_MISSION_TRANSITIONS.get(from_state, set())
    return to_state in allowed


@dataclass
class MissionStatus:
    """Immutable snapshot telemetry contract of the active mission state."""

    mission_name: str
    state: MissionState
    active_waypoint_id: Optional[str]
    active_waypoint_index: int
    total_waypoints: int
    distance_to_active_waypoint: float
    recovery_attempt_count: int
    last_recovery_result: str
    is_complete: bool
    is_aborted: bool
    abort_reason: MissionAbortReason
    timestamp: float = field(default_factory=time.time)
    reached_waypoint_ids: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert status to a serializable dictionary."""
        return {
            "mission_name": self.mission_name,
            "state": self.state.value,
            "active_waypoint_id": self.active_waypoint_id,
            "active_waypoint_index": int(self.active_waypoint_index),
            "total_waypoints": int(self.total_waypoints),
            "distance_to_active_waypoint": round(float(self.distance_to_active_waypoint), 3),
            "recovery_attempt_count": int(self.recovery_attempt_count),
            "last_recovery_result": self.last_recovery_result,
            "is_complete": bool(self.is_complete),
            "is_aborted": bool(self.is_aborted),
            "abort_reason": self.abort_reason.value,
            "timestamp": round(float(self.timestamp), 4),
            "reached_waypoint_ids": list(self.reached_waypoint_ids),
            "diagnostics": dict(self.diagnostics),
        }
