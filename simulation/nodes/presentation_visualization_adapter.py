"""Presentation Visualization and Telemetry Adapters for Phase 14.7.2.

Provides isolated, non-blocking, read-only visualization bridges connecting:
- WorldTerrainMap -> nav_msgs/msg/OccupancyGrid on /costmap/occupancy
- Canonical Mission Waypoints (W1-W5) -> visualization_msgs/msg/MarkerArray
- RobotModel URDF -> std_msgs/msg/String on /robot_description (latched)

Core Design Invariants:
1. Visualization failures are strictly NON-FATAL and cannot interrupt navigation.
2. OccupancyGrid conversion uses vectorized NumPy operations (< 2.0 ms latency).
3. ROS OccupancyGrid semantics: -1 = unknown, 0 = free, 100 = lethal/occupied.
4. Waypoint marker styling reflects live MissionExecutive/WaypointManager state.
5. No fabricated or hardcoded telemetry.
"""

from __future__ import annotations

import array
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

# ROS2 interfaces
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import ColorRGBA, Header, String
from visualization_msgs.msg import Marker, MarkerArray

# Domain types
from ai.spatial_fusion.world_map import WorldTerrainMap


class CostmapOccupancyAdapter:
    """Vectorized conversion and throttled publishing of WorldTerrainMap as nav_msgs/msg/OccupancyGrid."""

    def __init__(
        self,
        node: Optional[Node] = None,
        topic_name: str = "/costmap/occupancy",
        publish_rate_hz: float = 5.0,
    ) -> None:
        self.node = node
        self.topic_name = topic_name
        self.publish_rate_hz = float(publish_rate_hz)
        self.min_interval_s = 1.0 / max(0.1, self.publish_rate_hz)
        self.last_publish_time_s = 0.0

        self.publisher = None
        if self.node is not None:
            grid_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            self.publisher = self.node.create_publisher(OccupancyGrid, self.topic_name, grid_qos)

    @staticmethod
    def world_terrain_map_to_occupancy_grid(
        world_map: WorldTerrainMap,
        frame_id: Optional[str] = None,
        stamp: Optional[Any] = None,
    ) -> OccupancyGrid:
        """Converts WorldTerrainMap to ROS OccupancyGrid.

        Coordinate conventions:
          WorldTerrainMap:
            x_w in [x_bounds_m[0], x_bounds_m[1]]
            y_w in [y_bounds_m[0], y_bounds_m[1]]
            row = (x_max - x_w) / res (X axis down)
            col = (y_max - y_w) / res (Y axis down)

          ROS OccupancyGrid:
            origin at (x_min, y_min)
            width = cells along X (n_rows of WorldTerrainMap)
            height = cells along Y (n_cols of WorldTerrainMap)
            flat index k = i_y * width + i_x

        Value conventions:
          -1 = unknown
          0 = completely free
          100 = lethal obstacle
          1..99 = proportional intermediate cost
        """
        if world_map is None:
            raise ValueError("world_map cannot be None")

        # Vectorized flip & transpose to align with ROS standard grid ordering
        F_cost = world_map.cost[::-1, ::-1].T
        F_unk = world_map.unknown[::-1, ::-1].T
        F_obs = world_map.obstacle[::-1, ::-1].T

        # Proportional mapping [0.0, 0.95) -> [0, 99]
        scaled = np.clip(F_cost / 0.95, 0.0, 1.0) * 99.0
        occ = scaled.astype(np.int8)

        # Lethal obstacles and critical cost cells
        occ[F_obs | (F_cost >= 0.95)] = 100

        # Unknown space strictly preserved as -1
        occ[F_unk] = -1

        msg = OccupancyGrid()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.header.frame_id = frame_id or world_map.frame_id or "odom"

        msg.info.resolution = float(world_map.resolution_m)
        msg.info.width = int(world_map.n_rows)
        msg.info.height = int(world_map.n_cols)
        msg.info.origin.position.x = float(world_map.x_bounds_m[0])
        msg.info.origin.position.y = float(world_map.y_bounds_m[0])
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # Fast 1D conversion using C array.array directly (fast-path in rclpy IDL)
        msg.data = array.array("b", occ.tobytes())
        return msg

    def maybe_publish(
        self,
        world_map: WorldTerrainMap,
        current_time_s: float,
        stamp: Optional[Any] = None,
        frame_id: Optional[str] = None,
    ) -> bool:
        """Publishes the occupancy grid if the rate interval has elapsed."""
        if self.publisher is None or world_map is None:
            return False

        if (current_time_s - self.last_publish_time_s) < self.min_interval_s:
            return False

        try:
            msg = self.world_terrain_map_to_occupancy_grid(world_map, frame_id=frame_id, stamp=stamp)
            self.publisher.publish(msg)
            self.last_publish_time_s = current_time_s
            return True
        except Exception as e:
            if self.node is not None:
                self.node.get_logger().warn(f"[OCCUPANCY] Publish failure (non-fatal): {e}", throttle_duration_sec=5.0)
            return False


class WaypointMarkerAdapter:
    """Generates rich, state-aware visual markers for canonical mission waypoints."""

    DEFAULT_MISSION_PATH = "configs/mission_phase14_5_100m.yaml"

    def __init__(
        self,
        node: Optional[Node] = None,
        mission_file: Optional[str] = None,
        topic_name: str = "/mission/waypoint_markers",
    ) -> None:
        self.node = node
        self.topic_name = topic_name
        self.mission_file = mission_file or self._resolve_default_mission()
        self.waypoints: List[Dict[str, Any]] = []
        self._load_mission_waypoints()

        self.publisher = None
        if self.node is not None:
            marker_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            )
            self.publisher = self.node.create_publisher(MarkerArray, self.topic_name, marker_qos)

    def _resolve_default_mission(self) -> str:
        candidates = [
            self.DEFAULT_MISSION_PATH,
            os.path.join(os.path.dirname(__file__), "..", "..", self.DEFAULT_MISSION_PATH),
            os.path.join(os.getcwd(), self.DEFAULT_MISSION_PATH),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return self.DEFAULT_MISSION_PATH

    def _load_mission_waypoints(self) -> None:
        """Loads canonical waypoints from mission YAML file."""
        if not os.path.exists(self.mission_file):
            if self.node:
                self.node.get_logger().warn(f"[WAYPOINTS] Mission file not found at '{self.mission_file}'. Using fallback coordinates.")
            self.waypoints = [
                {"id": "W1", "x": 20.0, "y": 0.0, "tolerance": 0.5},
                {"id": "W2", "x": 40.0, "y": 6.0, "tolerance": 0.5},
                {"id": "W3", "x": 60.0, "y": -4.0, "tolerance": 0.5},
                {"id": "W4", "x": 80.0, "y": 3.0, "tolerance": 0.5},
                {"id": "W5", "x": 100.0, "y": 0.0, "tolerance": 0.5},
            ]
            return

        try:
            with open(self.mission_file, "r") as f:
                data = yaml.safe_load(f)
            raw_wps = data.get("mission", {}).get("waypoints", [])
            self.waypoints = []
            for wp in raw_wps:
                self.waypoints.append({
                    "id": str(wp.get("id", "")),
                    "x": float(wp.get("x", 0.0)),
                    "y": float(wp.get("y", 0.0)),
                    "tolerance": float(wp.get("xy_tolerance_m", 0.50)),
                    "description": str(wp.get("metadata", {}).get("description", "")),
                })
            if self.node:
                self.node.get_logger().info(f"[WAYPOINTS] Loaded {len(self.waypoints)} canonical waypoints from '{self.mission_file}'.")
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"[WAYPOINTS] Failed loading mission YAML: {e}")

    def generate_markers(
        self,
        active_wp_id: Optional[str] = None,
        reached_wp_ids: Optional[Sequence[str]] = None,
        robot_pose: Optional[Tuple[float, float, float]] = None,
        stamp: Optional[Any] = None,
        frame_id: str = "odom",
    ) -> MarkerArray:
        """Generates ground rings, beacon spheres, and status text labels."""
        m_array = MarkerArray()
        reached_set = set(reached_wp_ids or [])
        marker_id = 500

        # 0. START Marker
        start_ring = Marker()
        if stamp:
            start_ring.header.stamp = stamp
        start_ring.header.frame_id = frame_id
        start_ring.ns = "mission_start"
        start_ring.id = marker_id
        marker_id += 1
        start_ring.type = Marker.CYLINDER
        start_ring.action = Marker.ADD
        start_ring.pose.position.x = 0.0
        start_ring.pose.position.y = 0.0
        start_ring.pose.position.z = 0.02
        start_ring.scale = Vector3(x=1.2, y=1.2, z=0.04)
        start_ring.color = ColorRGBA(r=0.2, g=0.7, b=0.9, a=0.6)
        m_array.markers.append(start_ring)

        start_text = Marker()
        start_text.header = start_ring.header
        start_text.ns = "mission_start_label"
        start_text.id = marker_id
        marker_id += 1
        start_text.type = Marker.TEXT_VIEW_FACING
        start_text.action = Marker.ADD
        start_text.pose.position.x = 0.0
        start_text.pose.position.y = 0.0
        start_text.pose.position.z = 0.80
        start_text.scale.z = 0.30
        start_text.color = ColorRGBA(r=0.9, g=0.9, b=0.9, a=1.0)
        start_text.text = "MISSION START (0.0, 0.0)"
        m_array.markers.append(start_text)

        # 1. Waypoint Markers (W1 through W5)
        for i, wp in enumerate(self.waypoints):
            wid = wp["id"]
            wx = wp["x"]
            wy = wp["y"]
            is_final = (i == len(self.waypoints) - 1)

            # Determine state
            if wid in reached_set:
                state_str = "REACHED"
                ring_color = ColorRGBA(r=0.1, g=0.85, b=0.25, a=0.55)  # Soft green
                beacon_color = ColorRGBA(r=0.2, g=0.9, b=0.3, a=0.7)
                label_color = ColorRGBA(r=0.4, g=1.0, b=0.5, a=1.0)
                ring_scale = 1.2
            elif wid == active_wp_id:
                state_str = "ACTIVE"
                ring_color = ColorRGBA(r=1.0, g=0.80, b=0.05, a=0.90)  # Luminous gold
                beacon_color = ColorRGBA(r=1.0, g=0.90, b=0.10, a=1.0)
                label_color = ColorRGBA(r=1.0, g=0.95, b=0.2, a=1.0)
                ring_scale = 1.45
            else:
                state_str = "NOT_REACHED"
                ring_color = ColorRGBA(r=0.25, g=0.55, b=0.85, a=0.45)  # Muted slate blue
                beacon_color = ColorRGBA(r=0.3, g=0.6, b=0.9, a=0.6)
                label_color = ColorRGBA(r=0.8, g=0.85, b=0.9, a=0.85)
                ring_scale = 1.2

            # Compute distance if robot pose provided
            dist_str = ""
            if robot_pose is not None:
                d = math.hypot(robot_pose[0] - wx, robot_pose[1] - wy)
                dist_str = f" | {d:.1f}m"

            # Marker A: Ground Ring
            ring_m = Marker()
            if stamp:
                ring_m.header.stamp = stamp
            ring_m.header.frame_id = frame_id
            ring_m.ns = "waypoint_rings"
            ring_m.id = marker_id
            marker_id += 1
            ring_m.type = Marker.CYLINDER
            ring_m.action = Marker.ADD
            ring_m.pose.position.x = wx
            ring_m.pose.position.y = wy
            ring_m.pose.position.z = 0.02
            ring_m.scale = Vector3(x=ring_scale, y=ring_scale, z=0.04)
            ring_m.color = ring_color
            m_array.markers.append(ring_m)

            # Marker B: Beacon Sphere
            beacon_m = Marker()
            beacon_m.header = ring_m.header
            beacon_m.ns = "waypoint_beacons"
            beacon_m.id = marker_id
            marker_id += 1
            beacon_m.type = Marker.SPHERE
            beacon_m.action = Marker.ADD
            beacon_m.pose.position.x = wx
            beacon_m.pose.position.y = wy
            beacon_m.pose.position.z = 0.60
            b_size = 0.45 if state_str == "ACTIVE" else 0.35
            beacon_m.scale = Vector3(x=b_size, y=b_size, z=b_size)
            beacon_m.color = beacon_color
            m_array.markers.append(beacon_m)

            # Marker C: Status Text Label
            text_m = Marker()
            text_m.header = ring_m.header
            text_m.ns = "waypoint_labels"
            text_m.id = marker_id
            marker_id += 1
            text_m.type = Marker.TEXT_VIEW_FACING
            text_m.action = Marker.ADD
            text_m.pose.position.x = wx
            text_m.pose.position.y = wy
            text_m.pose.position.z = 1.30
            text_m.scale.z = 0.35 if state_str == "ACTIVE" else 0.28
            text_m.color = label_color

            if state_str == "ACTIVE":
                text_m.text = f">>> ACTIVE: {wid} <<<{dist_str}"
            elif state_str == "REACHED":
                text_m.text = f"[REACHED] {wid}"
            elif is_final:
                text_m.text = f"[GOAL] {wid}{dist_str}"
            else:
                text_m.text = f"[{wid}]{dist_str}"

            m_array.markers.append(text_m)

        return m_array

    def maybe_publish(
        self,
        active_wp_id: Optional[str] = None,
        reached_wp_ids: Optional[Sequence[str]] = None,
        robot_pose: Optional[Tuple[float, float, float]] = None,
        stamp: Optional[Any] = None,
        frame_id: str = "odom",
    ) -> bool:
        """Publishes waypoint markers safely."""
        if self.publisher is None:
            return False
        try:
            m_array = self.generate_markers(
                active_wp_id=active_wp_id,
                reached_wp_ids=reached_wp_ids,
                robot_pose=robot_pose,
                stamp=stamp,
                frame_id=frame_id,
            )
            self.publisher.publish(m_array)
            return True
        except Exception as e:
            if self.node:
                self.node.get_logger().warn(f"[WAYPOINTS] Publish failure (non-fatal): {e}", throttle_duration_sec=5.0)
            return False


class RobotDescriptionAdapter:
    """Publishes latched /robot_description from URDF for RViz RobotModel display."""

    DEFAULT_URDF_PATH = "simulation/models/ugv/ugv.urdf"

    def __init__(self, node: Node, urdf_path: Optional[str] = None) -> None:
        self.node = node
        self.urdf_path = urdf_path or self._resolve_default_urdf()
        self.publisher = None

        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.node.create_publisher(String, "/robot_description", latched_qos)

    def _resolve_default_urdf(self) -> str:
        candidates = [
            self.DEFAULT_URDF_PATH,
            os.path.join(os.path.dirname(__file__), "..", "..", self.DEFAULT_URDF_PATH),
            os.path.join(os.getcwd(), self.DEFAULT_URDF_PATH),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return self.DEFAULT_URDF_PATH

    def publish_once(self) -> bool:
        """Publishes URDF as a latched string message."""
        if not os.path.exists(self.urdf_path):
            self.node.get_logger().warn(f"[ROBOT_DESC] URDF not found at '{self.urdf_path}'")
            return False
        try:
            with open(self.urdf_path, "r") as f:
                content = f.read()
            msg = String()
            msg.data = content
            self.publisher.publish(msg)
            self.node.get_logger().info(f"[ROBOT_DESC] Published latched /robot_description ({len(content)} bytes).")
            return True
        except Exception as e:
            self.node.get_logger().warn(f"[ROBOT_DESC] Could not publish URDF (non-fatal): {e}")
            return False


class PresentationTelemetryCoordinator:
    """Unified telemetry coordinator managing OccupancyGrid, Waypoint Markers, and RobotModel."""

    def __init__(
        self,
        node: Node,
        mission_file: Optional[str] = None,
        costmap_rate_hz: float = 5.0,
        publish_robot_desc: bool = True,
    ) -> None:
        self.node = node
        self.costmap_adapter = CostmapOccupancyAdapter(
            node=self.node,
            topic_name="/costmap/occupancy",
            publish_rate_hz=costmap_rate_hz,
        )
        self.waypoint_adapter = WaypointMarkerAdapter(
            node=self.node,
            mission_file=mission_file,
            topic_name="/mission/waypoint_markers",
        )
        self.robot_desc_adapter = None
        if publish_robot_desc:
            self.robot_desc_adapter = RobotDescriptionAdapter(node=self.node)
            self.robot_desc_adapter.publish_once()

    def publish_step(
        self,
        world_map: Optional[WorldTerrainMap],
        mission_executive: Optional[Any],
        current_pose: Tuple[float, float, float],
        now_stamp: Any,
        current_time_s: Optional[float] = None,
        frame_id: str = "odom",
    ) -> None:
        """Periodic visualization publish step called from control loop.

        Guaranteed non-blocking and exception-isolated.
        """
        t_s = current_time_s if current_time_s is not None else time.time()

        # 1. Publish Costmap OccupancyGrid
        if world_map is not None:
            try:
                self.costmap_adapter.maybe_publish(
                    world_map=world_map,
                    current_time_s=t_s,
                    stamp=now_stamp,
                    frame_id=frame_id,
                )
            except Exception as e:
                self.node.get_logger().warn(f"[VIZ] Costmap publishing exception: {e}", throttle_duration_sec=5.0)

        # 2. Publish Waypoint Markers
        try:
            active_id = None
            reached_ids = []
            if mission_executive is not None and hasattr(mission_executive, "waypoint_manager"):
                mgr = mission_executive.waypoint_manager
                active_wp = getattr(mgr, "active_waypoint", None)
                active_id = getattr(active_wp, "id", None) if active_wp else None
                reached_ids = list(getattr(mgr, "reached_waypoint_ids", []))

            self.waypoint_adapter.maybe_publish(
                active_wp_id=active_id,
                reached_wp_ids=reached_ids,
                robot_pose=current_pose,
                stamp=now_stamp,
                frame_id=frame_id,
            )
        except Exception as e:
            self.node.get_logger().warn(f"[VIZ] Waypoint publishing exception: {e}", throttle_duration_sec=5.0)

    def get_waypoint_markers(
        self,
        active_wp_id: Optional[str] = None,
        reached_wp_ids: Optional[Sequence[str]] = None,
        robot_pose: Optional[Tuple[float, float, float]] = None,
        stamp: Optional[Any] = None,
        frame_id: str = "odom",
    ) -> MarkerArray:
        """Returns the marker array for direct merging into /visualization_marker_array if desired."""
        try:
            return self.waypoint_adapter.generate_markers(
                active_wp_id=active_wp_id,
                reached_wp_ids=reached_wp_ids,
                robot_pose=robot_pose,
                stamp=stamp,
                frame_id=frame_id,
            )
        except Exception:
            return MarkerArray()
