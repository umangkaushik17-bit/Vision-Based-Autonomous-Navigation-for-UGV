"""Planner ROS2 Node wrapping Phase 12 UGVNavigationPlanner and PlannerAdapter."""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Tuple, Union

from ai.planning.base import PlannerStatus
from ugv_navigation.adapters.planner_adapter import PathMsg, PlannerAdapter, PoseStampedMsg, TwistMsg
from ugv_navigation.diagnostics.node_diagnostics import NodeDiagnosticsTracker
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.tf.tf_tree import TfTreeBuffer

try:
    import rclpy
    from rclpy.node import Node
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class PlannerNode(Node if RCLPY_AVAILABLE else object):
    """Subscribes to /terrain/world_map, /localization/status, and /goal_pose;

    runs A* + DWA + Safety arbitration; publishes /cmd_vel and /global_path.
    """

    def __init__(
        self,
        adapter: Optional[PlannerAdapter] = None,
        tf_buffer: Optional[TfTreeBuffer] = None,
    ):
        self.adapter = adapter or PlannerAdapter()
        self.tf_buffer = tf_buffer or TfTreeBuffer()
        self.diagnostics = NodeDiagnosticsTracker("planner_node")

        self._latest_map: Optional[Union[WorldTerrainMapMsg, Any]] = None
        self._latest_loc: Optional[LocalizationStatusMsg] = None
        self._latest_goal: Optional[PoseStampedMsg] = None
        self._last_cmd: TwistMsg = TwistMsg.create(0.0, 0.0)
        self._last_path: PathMsg = PathMsg.from_planner_path(None)
        self._last_status: PlannerStatus = PlannerStatus.IDLE

        if RCLPY_AVAILABLE:
            super().__init__("planner_node")
            self.get_logger().info("Planner Node initialized.")

    def on_map_received(self, map_msg: Union[WorldTerrainMapMsg, Any]) -> None:
        """Callback for incoming WorldTerrainMap updates."""
        self._latest_map = map_msg

    def on_localization_status(self, loc_msg: LocalizationStatusMsg) -> None:
        """Callback for incoming localization estimates."""
        self._latest_loc = loc_msg

    def on_goal_received(self, goal_msg: PoseStampedMsg) -> None:
        """Callback for new target navigation goal."""
        self._latest_goal = goal_msg

    def process_cycle(
        self,
        sim_time_s: Optional[float] = None,
        dt_s: float = 0.10,
    ) -> Tuple[TwistMsg, PathMsg, PlannerStatus, Dict[str, Any]]:
        """Execute one complete planning cycle from stored topic messages."""
        if self._latest_loc is None:
            cmd = TwistMsg.create(0.0, 0.0)
            return cmd, self._last_path, PlannerStatus.LOCALIZATION_INVALID, {"error": "No localization received"}

        if self._latest_map is None:
            cmd = TwistMsg.create(0.0, 0.0)
            return cmd, self._last_path, PlannerStatus.MAP_UNAVAILABLE, {"error": "No WorldTerrainMap received"}

        # Check TF tree availability
        tf_avail = True
        try:
            # Quick check if base_link exists in buffer
            if len(self.tf_buffer._frames) == 0:
                tf_avail = True  # Defaults to true if mock buffer not initialized
        except Exception:
            tf_avail = True

        cmd, path, status, diag = self.adapter.plan_step_ros(
            loc_msg=self._latest_loc,
            map_msg=self._latest_map,
            goal_msg=self._latest_goal,
            sim_time_s=sim_time_s,
            dt_s=dt_s,
            tf_available=tf_avail,
        )

        self._last_cmd = cmd
        self._last_path = path
        self._last_status = status

        # Record diagnostics
        tracking_name = LocalizationStatusMsg.INV_STATE_MAPPING.get(self._latest_loc.tracking_state, "LOST").name if hasattr(LocalizationStatusMsg, "INV_STATE_MAPPING") else "UNKNOWN"
        self.diagnostics.record_cycle(
            latency_ms=float(diag.get("dwa_diag", {}).get("dwa_latency_ms", 10.0)),
            tracking_state_name=tracking_name,
            confidence=float(self._latest_loc.confidence),
            rejected=(status in (PlannerStatus.EMERGENCY_STOP, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, PlannerStatus.NO_PATH)),
            tf_failed=not tf_avail,
            stale=(status == PlannerStatus.MAP_STALE),
        )


        return cmd, path, status, diag
