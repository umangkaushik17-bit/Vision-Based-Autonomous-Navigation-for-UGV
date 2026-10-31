"""Waypoint sequencing, mission validation, and route progression manager."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import yaml

from ai.planning.base import RobotState
from ugv_mission.waypoint import Waypoint


class WaypointManager:
    """Manages ordered waypoint progression and arrival validation for autonomous missions."""

    def __init__(self, mission_name: str = "unnamed_mission") -> None:
        self.mission_name = mission_name
        self._waypoints: List[Waypoint] = []
        self._active_index: int = 0
        self._reached_ids: List[str] = []
        self._is_validated: bool = False

    @property
    def waypoints(self) -> List[Waypoint]:
        """Return shallow copy of all loaded waypoints."""
        return list(self._waypoints)

    @property
    def active_index(self) -> int:
        """Current 0-based waypoint index."""
        return self._active_index

    @property
    def total_count(self) -> int:
        """Total number of waypoints in current mission."""
        return len(self._waypoints)

    @property
    def active_waypoint(self) -> Optional[Waypoint]:
        """Return the currently active target waypoint, or None if mission empty/completed."""
        if 0 <= self._active_index < len(self._waypoints):
            return self._waypoints[self._active_index]
        return None

    @property
    def is_final_waypoint(self) -> bool:
        """Check if active waypoint is the final waypoint in the sequence."""
        if not self._waypoints:
            return False
        return self._active_index == (len(self._waypoints) - 1)

    @property
    def reached_ids(self) -> List[str]:
        """List of IDs of waypoints reached so far."""
        return list(self._reached_ids)

    @property
    def is_complete(self) -> bool:
        """Check if all waypoints have been reached."""
        return len(self._waypoints) > 0 and len(self._reached_ids) == len(self._waypoints)

    def load_waypoints(self, waypoints: Sequence[Union[Waypoint, Dict[str, Any]]]) -> Tuple[bool, str]:
        """Load ordered waypoints from a sequence of Waypoint objects or dictionaries."""
        parsed: List[Waypoint] = []
        for i, item in enumerate(waypoints):
            if isinstance(item, Waypoint):
                wp = item
            elif isinstance(item, dict):
                try:
                    wp = Waypoint.from_dict(item)
                except Exception as e:
                    return False, f"Malformed waypoint dictionary at index {i}: {e}"
            else:
                return False, f"Unsupported waypoint item type at index {i}: {type(item)}"

            is_valid, reason = wp.validate()
            if not is_valid:
                return False, f"Invalid waypoint at index {i}: {reason}"

            parsed.append(wp)

        if not parsed:
            return False, "Cannot load empty waypoint sequence"

        self._waypoints = parsed
        self._active_index = 0
        self._reached_ids = []
        self._is_validated = True
        return True, "ok"

    def load_from_dict(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Load mission specification from a dictionary containing a 'waypoints' list."""
        if not isinstance(data, dict):
            return False, "Mission data must be a dictionary"

        self.mission_name = str(data.get("name", data.get("mission_name", self.mission_name)))

        raw_wps = data.get("waypoints")
        if raw_wps is None:
            return False, "Mission data missing required 'waypoints' field"

        if not isinstance(raw_wps, (list, tuple)):
            return False, f"'waypoints' field must be a list, got {type(raw_wps)}"

        if len(raw_wps) == 0:
            return False, "Mission contains 0 waypoints"

        default_tol = float(data.get("default_xy_tolerance_m", 0.35))
        converted: List[Waypoint] = []
        for i, item in enumerate(raw_wps):
            if isinstance(item, Waypoint):
                converted.append(item)
            elif isinstance(item, dict):
                try:
                    wp = Waypoint.from_dict(item, default_tolerance_m=default_tol)
                    converted.append(wp)
                except Exception as e:
                    return False, f"Failed parsing waypoint at index {i}: {e}"
            else:
                return False, f"Invalid waypoint type at index {i}: {type(item)}"

        return self.load_waypoints(converted)

    def load_from_yaml(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """Load and parse mission from a YAML file."""
        p = Path(file_path)
        if not p.exists():
            return False, f"Mission file not found: {p}"

        try:
            with open(p, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except Exception as e:
            return False, f"YAML parse error: {e}"

        # If wrapped under a 'mission' root key
        if isinstance(content, dict) and "mission" in content and isinstance(content["mission"], dict):
            content = content["mission"]

        return self.load_from_dict(content)

    def check_active_reached(
        self,
        pose_or_state: Union[RobotState, Tuple[float, float], Tuple[float, float, float]],
    ) -> bool:
        """Check if robot pose satisfies the active waypoint's arrival condition.
        
        Strict Invariant:
        Only evaluates the currently active waypoint. Never checks future waypoints prematurely.
        """
        curr = self.active_waypoint
        if curr is None:
            return False
        return curr.is_reached(pose_or_state)

    def mark_active_reached(
        self,
        pose_or_state: Optional[Union[RobotState, Tuple[float, float], Tuple[float, float, float]]] = None,
        force: bool = False,
        tolerance_m: Optional[float] = None,
    ) -> bool:
        """Mark active waypoint reached if and only if the arrival condition is verified.
        
        Returns:
            True if arrival was verified and recorded.
            False if robot is outside tolerance.
        """
        curr = self.active_waypoint
        if curr is None:
            return False

        if force:
            if curr.id not in self._reached_ids:
                self._reached_ids.append(curr.id)
            return True

        if pose_or_state is not None:
            if tolerance_m is not None:
                if curr.distance_to(pose_or_state) <= tolerance_m:
                    if curr.id not in self._reached_ids:
                        self._reached_ids.append(curr.id)
                    return True
                return False
            if curr.is_reached(pose_or_state):
                if curr.id not in self._reached_ids:
                    self._reached_ids.append(curr.id)
                return True

        return False

    def advance(self) -> Optional[Waypoint]:
        """Advance to the next sequential waypoint in order.
        
        Strict Invariants:
        1. Cannot advance if current active waypoint has NOT been marked reached.
        2. Cannot skip indexes (advances strictly +1).
        3. Returns None if already on or beyond the final waypoint.
        """
        curr = self.active_waypoint
        if curr is None:
            return None

        # Verify that current waypoint is in reached_ids
        if curr.id not in self._reached_ids:
            raise RuntimeError(
                f"Cannot advance past waypoint {curr.id} (index {self._active_index}) "
                f"because it has not been marked reached."
            )

        if self._active_index + 1 < len(self._waypoints):
            self._active_index += 1
            return self._waypoints[self._active_index]

        # Reached final waypoint
        self._active_index = len(self._waypoints)
        return None

    def reset(self) -> None:
        """Reset progression state to start from the first waypoint."""
        self._active_index = 0
        self._reached_ids.clear()

    def clear(self) -> None:
        """Completely clear loaded waypoints and mission state."""
        self._waypoints.clear()
        self._active_index = 0
        self._reached_ids.clear()
        self._is_validated = False
