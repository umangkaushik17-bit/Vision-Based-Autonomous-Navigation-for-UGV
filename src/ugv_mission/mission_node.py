"""ROS 2 Node wrapper for the Phase 14.1 Mission Executive."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ai.planning.base import PlannerStatus
from ugv_mission.mission_executive import MissionExecutive, MissionExecutiveConfig
from ugv_mission.mission_state import MissionState, MissionStatus
from ugv_mission.waypoint import Waypoint

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from geometry_msgs.msg import Point, PoseStamped, Quaternion
    from nav_msgs.msg import Odometry
    from std_msgs.msg import ColorRGBA, String
    from visualization_msgs.msg import Marker, MarkerArray
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class MissionExecutiveNode(Node if RCLPY_AVAILABLE else object):
    """ROS 2 Node orchestrating multi-waypoint navigation missions."""

    def __init__(self, config: Optional[MissionExecutiveConfig] = None) -> None:
        self.config = config or MissionExecutiveConfig()
        self.executive = MissionExecutive(config=self.config)

        self._ugv_x: float = 0.0
        self._ugv_y: float = 0.0
        self._ugv_yaw: float = 0.0
        self._odom_received: bool = False
        self._last_published_wp_id: Optional[str] = None

        if RCLPY_AVAILABLE:
            super().__init__("mission_executive_node")

            self.declare_parameter("mission_file", "")
            self.declare_parameter("control_rate_hz", 10.0)

            mission_file = str(self.get_parameter("mission_file").value)
            rate_hz = float(self.get_parameter("control_rate_hz").value)

            state_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            )
            sensor_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            )

            # Subscriptions
            self.create_subscription(Odometry, "/odom", self._on_odom, sensor_qos)
            self.create_subscription(String, "/mission/command", self._on_command, state_qos)
            self.create_subscription(String, "/mission/load", self._on_load_mission, state_qos)

            # Publishers
            self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", state_qos)
            self.status_pub = self.create_publisher(String, "/mission/status", state_qos)
            self.marker_pub = self.create_publisher(MarkerArray, "/mission/markers", state_qos)

            # Periodic update timer
            self.timer = self.create_timer(1.0 / max(rate_hz, 1.0), self._tick)

            # Auto-load mission file if parameter specified
            if mission_file:
                ok, reason = self.executive.load_from_yaml(mission_file)
                if ok:
                    self.get_logger().info(f"Loaded mission file: {mission_file}")
                else:
                    self.get_logger().error(f"Failed loading mission file '{mission_file}': {reason}")

            self.get_logger().info("MissionExecutiveNode initialized.")

    def _on_odom(self, msg: Odometry) -> None:
        """Process UGV odometry position."""
        self._ugv_x = msg.pose.pose.position.x
        self._ugv_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        import math
        self._ugv_yaw = math.atan2(siny_cosp, cosy_cosp)
        self._odom_received = True

    def _on_command(self, msg: String) -> None:
        """Handle mission control commands: start, cancel, reset."""
        cmd = msg.data.strip().lower()
        if cmd == "start":
            current_pose = (self._ugv_x, self._ugv_y, self._ugv_yaw)
            ok, reason = self.executive.start_mission(current_pose)
            if RCLPY_AVAILABLE:
                self.get_logger().info(f"Command 'start': {ok} ({reason})")
        elif cmd == "cancel":
            self.executive.cancel_mission("Commanded via /mission/command")
            if RCLPY_AVAILABLE:
                self.get_logger().info("Mission cancelled via command topic.")
        elif cmd == "reset":
            self.executive.reset()
            self._last_published_wp_id = None
            if RCLPY_AVAILABLE:
                self.get_logger().info("Mission reset to IDLE.")

    def _on_load_mission(self, msg: String) -> None:
        """Load mission from file path or JSON payload."""
        payload = msg.data.strip()
        if payload.endswith(".yaml") or payload.endswith(".yml"):
            ok, reason = self.executive.load_from_yaml(payload)
        else:
            try:
                data = json.loads(payload)
                ok, reason = self.executive.waypoint_manager.load_from_dict(data)
            except Exception as e:
                ok, reason = False, f"JSON parse error: {e}"

        if RCLPY_AVAILABLE:
            if ok:
                self.get_logger().info(f"Loaded mission ({self.executive.total_waypoints} waypoints).")
            else:
                self.get_logger().error(f"Failed loading mission: {reason}")

    def _tick(self) -> None:
        """Periodic executive cycle."""
        if not self._odom_received:
            return

        current_pose = (self._ugv_x, self._ugv_y, self._ugv_yaw)
        now_s = time.time()

        # Step executive
        state, _, _, reason = self.executive.step(
            current_time_s=now_s,
            robot_pose=current_pose,
        )

        # Publish target goal pose whenever active waypoint changes
        active_wp = self.executive.active_waypoint
        if active_wp is not None and active_wp.id != self._last_published_wp_id:
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = active_wp.frame_id
            goal_msg.pose.position.x = float(active_wp.x)
            goal_msg.pose.position.y = float(active_wp.y)
            goal_msg.pose.position.z = 0.05
            if active_wp.yaw is not None:
                import math
                half_yaw = active_wp.yaw / 2.0
                goal_msg.pose.orientation.z = math.sin(half_yaw)
                goal_msg.pose.orientation.w = math.cos(half_yaw)
            else:
                goal_msg.pose.orientation.w = 1.0

            self.goal_pub.publish(goal_msg)
            self._last_published_wp_id = active_wp.id
            if RCLPY_AVAILABLE:
                self.get_logger().info(
                    f"Dispatched new target goal to /goal_pose: {active_wp.id} ({active_wp.x:.2f}, {active_wp.y:.2f})"
                )

        # Publish mission status
        status = self.executive.get_status(current_pose)
        status_msg = String()
        status_msg.data = json.dumps(status.to_dict())
        self.status_pub.publish(status_msg)

        # Publish RViz marker array
        self._publish_markers()

    def _publish_markers(self) -> None:
        """Publish RViz markers representing all waypoints and the route."""
        if not RCLPY_AVAILABLE:
            return

        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        wps = self.executive.waypoint_manager.waypoints
        active_id = self.executive.active_waypoint.id if self.executive.active_waypoint else None
        reached_ids = set(self.executive.waypoint_manager.reached_ids)

        for i, wp in enumerate(wps):
            # Sphere marker
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = wp.frame_id
            m.ns = "mission_waypoints"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(wp.x)
            m.pose.position.y = float(wp.y)
            m.pose.position.z = 0.20
            m.pose.orientation.w = 1.0
            m.scale.x = 0.35
            m.scale.y = 0.35
            m.scale.z = 0.35

            if wp.id == active_id:
                # Active: Bright Gold
                m.color = ColorRGBA(r=1.0, g=0.84, b=0.0, a=0.95)
                m.scale.x = 0.45
                m.scale.y = 0.45
                m.scale.z = 0.45
            elif wp.id in reached_ids:
                # Reached: Green
                m.color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.7)
            else:
                # Pending: Cyan
                m.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.7)

            markers.markers.append(m)

            # Text label
            t = Marker()
            t.header.stamp = now
            t.header.frame_id = wp.frame_id
            t.ns = "mission_labels"
            t.id = 100 + i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(wp.x)
            t.pose.position.y = float(wp.y)
            t.pose.position.z = 0.55
            t.pose.orientation.w = 1.0
            t.scale.z = 0.25
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            t.text = f"{wp.id}"
            markers.markers.append(t)

        self.marker_pub.publish(markers)


def main(args=None) -> None:
    """Entry point for ROS 2 mission executive node."""
    if not RCLPY_AVAILABLE:
        print("[ERROR] rclpy is required to run MissionExecutiveNode.", file=sys.stderr)
        sys.exit(1)

    rclpy.init(args=args)
    node = MissionExecutiveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
