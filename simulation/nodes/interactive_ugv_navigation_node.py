#!/usr/bin/env python3
"""Interactive Judge-Facing UGV Navigation Node for DEMO-01.

Coordinates the complete Phase 13 closed-loop navigation pipeline:
- Subscribes to /odom and broadcasts TF (odom -> base_link)
- Generates global path via A* routing around static rock boulder
- Simulates perception observations from dynamic obstacle telemetry (isolated sensor adapter)
- Executes MultiFrameTracker, MotionPredictor, DynamicOccupancyLayer, DWAPlanner, and BehavioralStateMachine
- Publishes /cmd_vel to drive the UGV in Gazebo
- Publishes full RViz visual telemetry (candidates, predictions, HUD overlay, dynamic slices)
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TransformStamped, Twist, Vector3
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import ColorRGBA, Empty, Header
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

try:
    from ai.traversability.dense_predictor import DenseTraversabilityPredictor
    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    DenseTraversabilityPredictor = None
    TORCH_AVAILABLE = False

# Custom ROS2 Messages
from ugv_navigation_msgs.msg import (
    BehavioralDecision as RosBehavioralDecision,
    DynamicObstacleDetection as RosDynamicObstacleDetection,
    DynamicObstacleDetectionArray as RosDynamicObstacleDetectionArray,
    DynamicOccupancyGridSlice as RosDynamicOccupancyGridSlice,
    DynamicOccupancySliceArray as RosDynamicOccupancySliceArray,
    LocalizationStatus as RosLocalizationStatus,
    PredictedObstacleTrajectory as RosPredictedObstacleTrajectory,
    PredictedTrajectoryArray as RosPredictedTrajectoryArray,
    TrackedObstacle as RosTrackedObstacle,
    TrackedObstacleArray as RosTrackedObstacleArray,
    TrajectoryWaypoint as RosTrajectoryWaypoint,
)

# AI Planning Components (Frozen Reference Core)
from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
)
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionConfig,
)
from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    TrackedObstacleState,
)
from ai.planning.dynamic.global_replanner import (
    DynamicReplanTrigger,
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
)
from ai.planning.dynamic.occupancy_layer import (
    DynamicOccupancyConfig,
    DynamicOccupancyLayer,
)
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig
from ai.planning.dynamic.tracker import (
    MultiFrameTracker,
    TrackerConfig,
    TrackStatus,
)
from ai.control.command_buffer import SafeCommandBuffer, SafeCommandBufferConfig, DispatchStatus
from ai.planning.backends.backend_interface import DWABackendType
from ai.planning.backends.unified_planner import UnifiedDWAPlanner
from ai.planning.base import PlanningConfig
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path as AIPath, PathPoint
from ai.planning.recovery_controller import (
    RecoveryConfig,
    RecoveryController,
    RecoveryState,
)
from ai.planning.safety_governor import (
    LatchedHazard,
    RecentHazardMemory,
    RecentHazardMemoryConfig,
    SafetyGovernor,
    SafetyGovernorConfig,
)
from ai.spatial_fusion.world_map import WorldTerrainMap


class InteractiveUGVNavigationNode(Node):
    """Integrated ROS2 Node orchestrating the judge-facing UGV demonstration."""

    def __init__(self) -> None:
        super().__init__("interactive_ugv_navigation_node")

        # Parameters
        self.declare_parameter("goal_x", 7.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("rock_x", 3.5)
        self.declare_parameter("rock_y", 0.15)
        self.declare_parameter("rock_radius", 0.65)
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("cmd_publish_rate_hz", 20.0)
        self.declare_parameter("mission_file", "")

        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.rock_x = float(self.get_parameter("rock_x").value)
        self.rock_y = float(self.get_parameter("rock_y").value)
        self.rock_radius = float(self.get_parameter("rock_radius").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.cmd_publish_rate_hz = float(self.get_parameter("cmd_publish_rate_hz").value)
        self.mission_file = str(self.get_parameter("mission_file").value)

        # High-frequency continuous safe command buffer
        self.command_buffer = SafeCommandBuffer(SafeCommandBufferConfig(
            publish_rate_hz=self.cmd_publish_rate_hz,
            freshness_timeout_s=0.35,
            stale_timeout_s=0.50,
            max_deceleration_mps2=1.50,
        ))

        # 1. State Storage
        self.ugv_x: float = 0.0
        self.ugv_y: float = 0.0
        self.ugv_yaw: float = 0.0
        self.ugv_v: float = 0.0
        self.ugv_w: float = 0.0
        self.last_odom_time: float = 0.0
        self.odom_received: bool = False

        # Dynamic obstacle raw state (for observation adapter)
        self.obs_x: float = 2.5
        self.obs_y: float = -3.0
        self.obs_yaw: float = 1.57
        self.obs_vx: float = 0.0
        self.obs_vy: float = 0.0
        self.obs_odom_received: bool = False

        # 2. Build World Terrain Map with Static Rock Boulder
        self.world_map = WorldTerrainMap(
            resolution_m=0.10,
            x_bounds_m=[-5.0, 12.0],
            y_bounds_m=[-8.0, 8.0],
            created_timestamp=0.0,
        )
        self.world_map.cost.fill(0.10)
        self.world_map.obstacle.fill(False)
        self.world_map.hazard.fill(0.0)
        self.world_map.unknown.fill(False)

        # Imprint static rock boulder
        r_c, c_c = self.world_map.world_to_grid(self.rock_x, self.rock_y)
        r_rad = max(1, int(round(self.rock_radius / self.world_map.resolution_m)))
        for dr in range(-r_rad, r_rad + 1):
            for dc in range(-r_rad, r_rad + 1):
                if dr * dr + dc * dc <= r_rad * r_rad:
                    rr, cc = r_c + dr, c_c + dc
                    if 0 <= rr < self.world_map.shape[0] and 0 <= cc < self.world_map.shape[1]:
                        self.world_map.obstacle[rr, cc] = True
                        self.world_map.cost[rr, cc] = 1.0

        # Imprint soft safety buffer around rock
        buf_rad = r_rad + 3
        for dr in range(-buf_rad, buf_rad + 1):
            for dc in range(-buf_rad, buf_rad + 1):
                d_sq = dr * dr + dc * dc
                if r_rad * r_rad < d_sq <= buf_rad * buf_rad:
                    rr, cc = r_c + dr, c_c + dc
                    if 0 <= rr < self.world_map.shape[0] and 0 <= cc < self.world_map.shape[1]:
                        if not self.world_map.obstacle[rr, cc]:
                            self.world_map.cost[rr, cc] = max(self.world_map.cost[rr, cc], 0.65)

        # 3. Core Phase 13 Frozen Navigation Stack
        self.tracker = MultiFrameTracker(TrackerConfig(target_frame="odom", use_kalman_filter=True))
        self.predictor = MotionPredictor(PredictorConfig(default_horizon_s=2.0, default_dt_s=0.25))
        self.occupancy_layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))
        self.collision_checker = PredictiveCollisionChecker(
            PredictiveCollisionConfig(vehicle_length_m=0.65, vehicle_width_m=0.45, collision_margin_m=0.08)
        )
        self.global_planner = AStarPlanner()
        # Hardware-independent DWA (defaults to native C++ AVX2 backend in AUTO mode)
        dwa_cfg = PlanningConfig.load()
        # Restrict nominal DWA local velocity sampling to forward-only (v >= 0.0).
        # Reverse maneuvers are deterministically governed by RecoveryController (Phase 13.24)
        dwa_cfg.vehicle.min_linear_velocity_mps = 0.0
        self.dwa_planner = UnifiedDWAPlanner(config=dwa_cfg, backend=DWABackendType.AUTO)
        self.dwa_planner.enable_behavioral_arbitration = True
        self.behavior_machine = BehavioralStateMachine(BehavioralConfig())
        self.replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())

        # Phase 13.23: Lost-Obstacle Hazard Latching & Authoritative Safety Governor
        self.hazard_memory = RecentHazardMemory(
            RecentHazardMemoryConfig(
                proximity_threshold_m=2.50,
                critical_proximity_m=1.20,
                base_ttl_s=3.50,
                close_ttl_s=5.00,
            )
        )
        self.safety_governor = SafetyGovernor(
            SafetyGovernorConfig(
                stop_distance_m=0.40,
                cautious_distance_m=0.95,
                cautious_max_v_mps=0.15,
                corridor_half_width_m=0.35,
            )
        )

        # Phase 13.24: Safe Reverse Recovery & Deadlock Replanner
        self.recovery_controller = RecoveryController(
            RecoveryConfig(
                reverse_velocity_mps=-0.15,
                target_reverse_distance_m=0.45,
                deadlock_stop_cycles=4,
            )
        )

        self.active_global_path: Optional[AIPath] = None
        self.last_behavior: Optional[BehavioralDecision] = None
        self.mission_completed: bool = False
        # Rolling pose history for spatial progress monitoring (detecting chattering/limit-cycles)
        self._pose_history: List[Tuple[float, float, float]] = []

        # Optional Phase 14.1 Multi-Waypoint Mission Executive
        self.mission_executive = None
        if self.mission_file:
            try:
                from ugv_mission import MissionExecutive, MissionExecutiveConfig
                self.mission_executive = MissionExecutive(
                    config=MissionExecutiveConfig(mission_name="demo_01_multi_waypoint"),
                    global_planner=self.replanner,
                    safety_governor=self.safety_governor,
                    command_buffer=self.command_buffer,
                )
                ok, reason = self.mission_executive.load_from_yaml(self.mission_file)
                if ok and self.mission_executive.active_waypoint is not None:
                    self.goal_x = self.mission_executive.active_waypoint.x
                    self.goal_y = self.mission_executive.active_waypoint.y
                    self.get_logger().info(
                        f"[MISSION] Loaded multi-waypoint mission '{self.mission_file}' "
                        f"({self.mission_executive.total_waypoints} waypoints). "
                        f"First waypoint: ({self.goal_x:.2f}, {self.goal_y:.2f})"
                    )
                else:
                    self.get_logger().error(f"[MISSION] Failed to load mission file: {reason}")
            except Exception as e:
                self.get_logger().warn(f"[MISSION] Could not initialize MissionExecutive: {e}")

        # 4. TF Broadcasters
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        # 5. QoS Profiles
        cmd_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        sensor_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE, depth=5)
        state_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE, depth=5)

        # 6. Subscriptions
        self.create_subscription(Odometry, "/odom", self._on_ugv_odom, sensor_qos)
        self.create_subscription(Odometry, "/dynamic_obstacle/odom", self._on_obs_odom, sensor_qos)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal_pose, state_qos)
        self.create_subscription(Empty, "/demo/reset", self._on_reset_requested, state_qos)
        self.create_subscription(RosImage, "/camera/image_raw", self._on_camera_image, sensor_qos)

        # 6b. Live Traversability Model (RELLIS-3D Dense Net)
        self.traversability_predictor = None
        self.last_trav_fps = 0.0
        self.last_trav_ms = 0.0
        if DenseTraversabilityPredictor is not None:
            try:
                self.traversability_predictor = DenseTraversabilityPredictor(
                    checkpoint="outputs/rellis3d/checkpoints/best.pt"
                )
                self.get_logger().info(
                    f"Loaded Dense Traversability Model: {self.traversability_predictor.model_name} | "
                    f"PERCEPTION DEVICE: {self.traversability_predictor.device.upper()}"
                )
            except Exception as e:
                self.get_logger().warn(f"Could not initialize dense traversability predictor: {e}")
        else:
            self.get_logger().info("Dense Traversability: Running in core geometric/obstacle mode (PyTorch optional).")

        # 6c. Vision-in-the-Loop Metric Projection & Costmap Fusion (Phase 14.3)
        self.vision_costmap_integrator = None
        try:
            from production.perception.vision_costmap_integrator import VisionCostmapIntegrator
            self.vision_costmap_integrator = VisionCostmapIntegrator.create_simulation_default()
            self.get_logger().info("Loaded VisionCostmapIntegrator (calibrated for Gazebo demo_outdoor camera).")
        except Exception as e:
            self.get_logger().warn(f"Could not initialize VisionCostmapIntegrator: {e}")

        # 7. Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", cmd_qos)
        self.path_pub = self.create_publisher(Path, "/planning/global_path", state_qos)
        self.traj_pub = self.create_publisher(Path, "/planning/dynamic_trajectory", state_qos)
        self.decision_pub = self.create_publisher(RosBehavioralDecision, "/behavior/tactical_decision", state_qos)
        self.loc_pub = self.create_publisher(RosLocalizationStatus, "/localization/status", state_qos)
        self.detection_pub = self.create_publisher(RosDynamicObstacleDetectionArray, "/perception/dynamic_obstacles", sensor_qos)
        self.track_pub = self.create_publisher(RosTrackedObstacleArray, "/tracking/dynamic_obstacles", sensor_qos)
        self.pred_pub = self.create_publisher(RosPredictedTrajectoryArray, "/prediction/obstacle_trajectories", sensor_qos)
        self.marker_pub = self.create_publisher(MarkerArray, "/visualization_marker_array", state_qos)
        self.overlay_pub = self.create_publisher(RosImage, "/perception/traversability_overlay", sensor_qos)

        # 8. Decoupled Planning & Actuation Timers
        # 8a. Planning loop runs at control_rate_hz (default 10 Hz)
        dt_plan = 1.0 / self.control_rate_hz
        self.control_timer = self.create_timer(dt_plan, self._control_loop)

        # 8b. High-frequency command dispatcher runs at cmd_publish_rate_hz (default 20 Hz)
        dt_cmd = 1.0 / self.cmd_publish_rate_hz
        self.cmd_timer = self.create_timer(dt_cmd, self._dispatch_cmd_vel_loop)

        self.get_logger().info(
            f"InteractiveUGVNavigationNode initialized (Planner: {self.control_rate_hz}Hz, "
            f"Actuator: {self.cmd_publish_rate_hz}Hz). Waiting for odometry..."
        )

    def _publish_static_transforms(self) -> None:
        """Broadcasts static camera transforms base_link -> camera_link -> camera_optical."""
        now = self.get_clock().now().to_msg()
        t1 = TransformStamped()
        t1.header.stamp = now
        t1.header.frame_id = "base_link"
        t1.child_frame_id = "camera_link"
        t1.transform.translation.x = 0.15
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = 0.40
        # 15 deg pitch down
        pitch = 15.0 * math.pi / 180.0
        t1.transform.rotation.y = math.sin(pitch / 2.0)
        t1.transform.rotation.w = math.cos(pitch / 2.0)

        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = "camera_link"
        t2.child_frame_id = "camera_optical"
        # Standard optical frame rotation (-90 deg yaw, 90 deg pitch)
        t2.transform.rotation.x = -0.5
        t2.transform.rotation.y = 0.5
        t2.transform.rotation.z = -0.5
        t2.transform.rotation.w = 0.5

        self.static_tf_broadcaster.sendTransform([t1, t2])

    def _on_ugv_odom(self, msg: Odometry) -> None:
        """Processes UGV odometry, tracks state, and broadcasts dynamic TF."""
        self.ugv_x = msg.pose.pose.position.x
        self.ugv_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.ugv_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.ugv_v = msg.twist.twist.linear.x
        self.ugv_w = msg.twist.twist.angular.z
        self.last_odom_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.odom_received = True

        # Broadcast odom -> base_link
        tf_msg = TransformStamped()
        tf_msg.header = msg.header
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "base_link"
        tf_msg.transform.translation.x = self.ugv_x
        tf_msg.transform.translation.y = self.ugv_y
        tf_msg.transform.translation.z = msg.pose.pose.position.z
        tf_msg.transform.rotation = q
        self.tf_broadcaster.sendTransform(tf_msg)

        # Publish localization status
        loc_msg = RosLocalizationStatus()
        loc_msg.header = msg.header
        loc_msg.header.frame_id = "odom"
        loc_msg.child_frame_id = "base_link"
        loc_msg.tracking_state = 2  # TRACKING_OK
        loc_msg.confidence = 1.0
        loc_msg.is_pose_valid = True
        loc_msg.allows_spatial_fusion = True
        loc_msg.pose = msg.pose
        loc_msg.twist = msg.twist
        self.loc_pub.publish(loc_msg)

    def _on_obs_odom(self, msg: Odometry) -> None:
        """Processes dynamic obstacle telemetry and simulates noisy sensor observations."""
        self.obs_x = msg.pose.pose.position.x
        self.obs_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.obs_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.obs_vx = msg.twist.twist.linear.x
        self.obs_vy = msg.twist.twist.linear.y
        self.obs_odom_received = True

    def _on_camera_image(self, msg: RosImage) -> None:
        """Processes camera feed through the RELLIS-3D Dense Traversability Net."""
        if self.traversability_predictor is None:
            return
        try:
            h, w = msg.height, msg.width
            if msg.encoding in ("rgb8", "bgr8"):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
                rgb = arr if msg.encoding == "rgb8" else arr[:, :, ::-1]
            elif msg.encoding == "mono8":
                mono = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w))
                rgb = np.stack([mono, mono, mono], axis=-1)
            else:
                return

            t0 = time.perf_counter()
            res = self.traversability_predictor.predict(rgb)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.last_trav_ms = dt_ms
            self.last_trav_fps = 1000.0 / max(dt_ms, 1e-3)

            # Phase 14.3 Vision-in-the-Loop: Metric Ground Projection & Costmap Fusion
            if self.vision_costmap_integrator is not None and self.odom_received:
                t_img_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                bev_costmap = self.vision_costmap_integrator.project_traversability(res, current_time_s=t_img_s)
                self.vision_costmap_integrator.fuse_into_world_map(
                    bev_costmap,
                    (self.ugv_x, self.ugv_y, self.ugv_yaw),
                    self.world_map,
                )

            if res.overlay is not None:
                out_msg = RosImage()
                out_msg.header = msg.header
                out_msg.height, out_msg.width = res.overlay.shape[:2]
                out_msg.encoding = "rgb8"
                out_msg.step = out_msg.width * 3
                out_msg.data = res.overlay.tobytes()
                self.overlay_pub.publish(out_msg)
        except Exception:
            pass

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        """Allows dynamic goal updates."""
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.active_global_path = None  # Trigger replan
        self.mission_completed = False
        self._pose_history.clear()
        self.get_logger().info(f"New goal set: ({self.goal_x:.2f}, {self.goal_y:.2f})")

    def _on_reset_requested(self, _: Empty) -> None:
        """Reset internal navigation state."""
        self.get_logger().info("Resetting navigation and tracker states.")
        self.command_buffer.reset()
        self.hazard_memory.clear()
        self.recovery_controller.reset()
        self.cmd_pub.publish(Twist())
        self.ugv_v = 0.0
        self.ugv_w = 0.0
        self.active_global_path = None
        self._pose_history.clear()
        self.tracker = MultiFrameTracker(TrackerConfig(target_frame="odom", use_kalman_filter=True))
        self.predictor = MotionPredictor(PredictorConfig(default_horizon_s=2.0, default_dt_s=0.25))
        self.occupancy_layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))
        self.behavior_machine = BehavioralStateMachine(BehavioralConfig())
        self.replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
        self.mission_completed = False

    def _generate_sensor_observation(self, t_now: float) -> List[DynamicObstacleDetection]:
        """Simulated Sensor Observation Adapter (clearly documented per M5)."""
        if not self.obs_odom_received:
            return []

        # Only observe obstacle if within front camera sensing range and FOV
        dx = self.obs_x - self.ugv_x
        dy = self.obs_y - self.ugv_y
        dist_ugv_obs = math.hypot(dx, dy)
        if dist_ugv_obs > 6.0:
            return []

        # Relative bearing in UGV forward heading frame
        angle_to_obs = math.atan2(dy, dx)
        rel_bearing = abs((angle_to_obs - self.ugv_yaw + math.pi) % (2.0 * math.pi) - math.pi)
        if rel_bearing > math.radians(45.0):
            return []

        # Add realistic sensor observation noise (std = 0.02m)
        meas_x = self.obs_x + float(np.random.normal(0.0, 0.02))
        meas_y = self.obs_y + float(np.random.normal(0.0, 0.02))

        det = DynamicObstacleDetection(
            detection_id="dynamic_obs_1",
            timestamp=float(t_now),
            frame_id="odom",
            bbox_center=(float(meas_x), float(meas_y)),
            bbox_size=(0.8, 0.8),
            bbox_yaw=float(self.obs_yaw),
            confidence=0.95,
        )

        # Publish ROS observation message
        ros_det_array = RosDynamicObstacleDetectionArray()
        ros_det_array.header.stamp = self.get_clock().now().to_msg()
        ros_det_array.header.frame_id = "odom"
        ros_det = RosDynamicObstacleDetection()
        ros_det.header = ros_det_array.header
        ros_det.detection_id = 1
        ros_det.class_name = "dynamic_obstacle"
        ros_det.confidence = float(det.confidence)
        ros_det.position.x = float(det.bbox_center[0])
        ros_det.position.y = float(det.bbox_center[1])
        ros_det.position.z = 0.3
        ros_det.dimensions.x = 0.8
        ros_det.dimensions.y = 0.8
        ros_det.dimensions.z = 0.6
        ros_det.yaw_rad = float(det.bbox_yaw)
        ros_det.is_valid = True
        ros_det_array.detections.append(ros_det)
        self.detection_pub.publish(ros_det_array)

        return [det]

    def _dispatch_cmd_vel_loop(self) -> None:
        """High-frequency continuous command dispatcher (default 20 Hz) to Gazebo diff-drive."""
        if not self.odom_received:
            return
        now_s = self.last_odom_time if self.last_odom_time > 0 else time.time()
        dispatched_v, dispatched_w, status, diag = self.command_buffer.get_dispatch_command(now_s)

        cmd_twist = Twist()
        cmd_twist.linear.x = float(dispatched_v)
        cmd_twist.angular.z = float(dispatched_w)
        self.cmd_pub.publish(cmd_twist)

    def _control_loop(self) -> None:
        """Main periodic navigation execution step."""
        if not self.odom_received:
            return

        t_loop_start = time.perf_counter()
        t_now = self.last_odom_time if self.last_odom_time > 0 else time.time()
        robot_state = RobotState(
            x=self.ugv_x, y=self.ugv_y, yaw=self.ugv_yaw, v=self.ugv_v, omega=self.ugv_w, timestamp=t_now
        )

        # 1. Check Goal Arrival
        dist_to_goal = math.hypot(self.ugv_x - self.goal_x, self.ugv_y - self.goal_y)
        if dist_to_goal <= 0.35:
            if self.mission_executive is not None and not self.mission_executive.waypoint_manager.is_final_waypoint:
                curr_wp = self.mission_executive.active_waypoint
                self.get_logger().info(f"[MISSION] Waypoint {curr_wp.id if curr_wp else ''} reached! Advancing to next waypoint.")
                self.mission_executive.waypoint_manager.mark_active_reached((self.ugv_x, self.ugv_y, self.ugv_yaw))
                next_wp = self.mission_executive.waypoint_manager.advance()
                if next_wp is not None:
                    self.goal_x = next_wp.x
                    self.goal_y = next_wp.y
                    self.active_global_path = None
                    self._pose_history.clear()
                    self.recovery_controller.reset()
                    self.get_logger().info(f"[MISSION] Now targeting waypoint {next_wp.id} at ({self.goal_x:.2f}, {self.goal_y:.2f})")
                    return

            if not self.mission_completed:
                self.get_logger().info("GOAL REACHED! Mission successful.")
                self.mission_completed = True
            # Command stop immediately via command buffer
            self.command_buffer.emergency_stop(t_now, reason="GOAL_REACHED")
            stop_twist = Twist()
            self.cmd_pub.publish(stop_twist)
            self._publish_visualizations(None, [], [], [], "GOAL REACHED", dist_to_goal)
            return

        # 2. Perception & Tracking (Phase 13 Pipeline)
        t0 = time.perf_counter()
        detections = self._generate_sensor_observation(t_now)
        tracked_array = self.tracker.update(detections, t_now)
        confirmed_tracks = [
            t for t in tracked_array.tracks if t.status in (TrackStatus.TRACKING, TrackStatus.DEGRADED, TrackStatus.LOST)
        ]
        t_track = (time.perf_counter() - t0) * 1000.0

        # Publish Tracked Obstacles
        ros_tracks = RosTrackedObstacleArray()
        ros_tracks.header.stamp = self.get_clock().now().to_msg()
        ros_tracks.header.frame_id = "odom"
        for tr in confirmed_tracks:
            r_tr = RosTrackedObstacle()
            r_tr.header = ros_tracks.header
            try:
                r_tr.track_id = int(tr.track_id)
            except Exception:
                r_tr.track_id = 1
            r_tr.status = str(getattr(tr.status, "value", getattr(tr.status, "name", tr.status)))
            r_tr.is_confirmed = True
            r_tr.confidence = 0.95
            r_tr.position.x = float(tr.x)
            r_tr.position.y = float(tr.y)
            r_tr.position.z = 0.3
            r_tr.velocity.x = float(tr.vx)
            r_tr.velocity.y = float(tr.vy)
            r_tr.bbox_size.x = float(tr.bbox_size[0])
            r_tr.bbox_size.y = float(tr.bbox_size[1])
            r_tr.bbox_size.z = 0.6
            r_tr.bbox_yaw = float(tr.bbox_yaw)
            ros_tracks.tracks.append(r_tr)
        self.track_pub.publish(ros_tracks)

        # 3. Prediction & Recent Hazard Latching (Phase 13.23 "Not Detected != Safe")
        t0 = time.perf_counter()
        all_hazards = self.hazard_memory.update(confirmed_tracks, t_now, (self.ugv_x, self.ugv_y, self.ugv_yaw))
        pred_array = self.predictor.predict_tracks(confirmed_tracks)
        active_predictions = list(pred_array.predictions)

        # Augment with dropped latched hazards so DWA evaluates them
        dropped_hazards = self.hazard_memory.get_latched_dropped_hazards([t.track_id for t in confirmed_tracks])
        for dh in dropped_hazards:
            active_predictions.append(dh.to_predicted_trajectory(horizon_s=2.0, dt_s=0.25))

        t_pred = (time.perf_counter() - t0) * 1000.0

        # Publish Predictions
        ros_preds = RosPredictedTrajectoryArray()
        ros_preds.header = ros_tracks.header
        for pr in active_predictions:
            r_pr = RosPredictedObstacleTrajectory()
            r_pr.header = ros_preds.header
            try:
                r_pr.track_id = int(pr.track_id)
            except Exception:
                r_pr.track_id = 1
            r_pr.status = "TRACKING"
            r_pr.confidence = 0.95
            r_pr.prediction_horizon_s = 2.0
            r_pr.bbox_size.x = float(pr.bbox_size[0]) if hasattr(pr, "bbox_size") else 0.8
            r_pr.bbox_size.y = float(pr.bbox_size[1]) if hasattr(pr, "bbox_size") else 0.8
            r_pr.bbox_size.z = 0.6
            for wp in pr.waypoints:
                r_wp = RosTrajectoryWaypoint()
                r_wp.relative_time_s = float(getattr(wp, "time_offset", getattr(wp, "timestamp", 0.0) - t_now))
                r_wp.position.x = float(wp.x)
                r_wp.position.y = float(wp.y)
                r_wp.position.z = 0.3
                r_wp.velocity.x = float(wp.vx) if hasattr(wp, "vx") else 0.0
                r_wp.velocity.y = float(wp.vy) if hasattr(wp, "vy") else 0.0
                r_wp.confidence = 0.95
                r_pr.waypoints.append(r_wp)
            ros_preds.trajectories.append(r_pr)
        self.pred_pub.publish(ros_preds)

        # 4. Dynamic Occupancy Update
        t0 = time.perf_counter()
        self.occupancy_layer.update_predictions(active_predictions, header_timestamp=t_now)
        t_occ = (time.perf_counter() - t0) * 1000.0

        # 5. Global Path Generation (Routing around static rock)
        t0 = time.perf_counter()
        if self.active_global_path is None or len(self.active_global_path.points) == 0:
            init_res = self.replanner.execute_replan(
                current_pose=(robot_state.x, robot_state.y, robot_state.yaw),
                goal=(self.goal_x, self.goal_y),
                base_map=self.world_map,
                dynamic_obstacles=confirmed_tracks,
            )
            if init_res.success and init_res.path is not None:
                self.active_global_path = init_res.path
                self.replanner.set_global_path(self.active_global_path)
                self.get_logger().info(f"Global path computed with {len(self.active_global_path.points)} waypoints.")
        t_plan = (time.perf_counter() - t0) * 1000.0

        # Publish Global Path
        if self.active_global_path is not None:
            ros_path = Path()
            ros_path.header.stamp = self.get_clock().now().to_msg()
            ros_path.header.frame_id = "odom"
            for pt in self.active_global_path.points:
                ps = PoseStamped()
                ps.header = ros_path.header
                ps.pose.position.x = float(pt.x)
                ps.pose.position.y = float(pt.y)
                ps.pose.position.z = 0.05
                ros_path.poses.append(ps)
            self.path_pub.publish(ros_path)

        # 6. DWA Local Trajectory Evaluation
        t0 = time.perf_counter()
        cand_traj, cand_v, cand_w, planner_status, dwa_diag = self.dwa_planner.compute_trajectory(
            current_state=robot_state,
            global_path=self.active_global_path,
            costmap=self.world_map,
            dynamic_layer=self.occupancy_layer,
            predicted_tracks=active_predictions,
        )
        t_dwa = (time.perf_counter() - t0) * 1000.0

        # Extract latest candidate trajectories for visualization
        candidate_list = getattr(self.dwa_planner, "latest_candidates", [])

        # 7. Behavioral Decision
        decision_str = dwa_diag.get("behavioral_state", "PROCEED")

        # Phase 13.24: Multi-signal Deadlock & Spatial Progress Monitoring
        dist_to_goal = math.hypot(self.ugv_x - self.goal_x, self.ugv_y - self.goal_y)

        # Rolling pose history for spatial progress monitoring (detecting chattering/limit-cycles)
        self._pose_history.append((t_now, self.ugv_x, self.ugv_y))
        progress_window_s = 2.50
        cutoff_t = t_now - progress_window_s
        self._pose_history = [p for p in self._pose_history if p[0] >= cutoff_t]

        lack_of_progress = False
        if len(self._pose_history) >= 8 and (t_now - self._pose_history[0][0]) >= (progress_window_s * 0.75):
            start_x, start_y = self._pose_history[0][1], self._pose_history[0][2]
            net_disp = math.hypot(self.ugv_x - start_x, self.ugv_y - start_y)
            if net_disp < 0.15 and dist_to_goal > 0.50:
                lack_of_progress = True

        is_dynamic_deadlock = (dwa_diag.get("deadlock_event") == "DYNAMIC_DEADLOCK" and dist_to_goal > 0.50)

        is_deadlocked = (
            decision_str in ("STOP", "EMERGENCY_STOP")
            or planner_status in (PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, PlannerStatus.NO_SAFE_TRAJECTORY)
            or (cand_v <= 0.01 and dist_to_goal > 0.5 and abs(robot_state.v) <= 0.05)
            or is_dynamic_deadlock
            or lack_of_progress
        )
        rec_v, rec_w, rec_state, rec_reason = self.recovery_controller.step(
            current_time_s=t_now,
            robot_pose=(self.ugv_x, self.ugv_y, self.ugv_yaw),
            is_deadlocked=is_deadlocked,
            hazards=all_hazards,
            costmap=self.world_map,
        )

        if rec_state == RecoveryState.REPLAN_READY:
            self.get_logger().info("Recovery complete: Triggering global A* replan around obstacle...")
            replan_res = self.replanner.execute_replan(
                current_pose=(self.ugv_x, self.ugv_y, self.ugv_yaw),
                goal=(self.goal_x, self.goal_y),
                base_map=self.world_map,
                dynamic_obstacles=confirmed_tracks,
            )
            if replan_res.success and replan_res.path is not None:
                self.active_global_path = replan_res.path
                self.replanner.set_global_path(self.active_global_path)
                self.get_logger().info(f"Replan succeeded! New path has {len(self.active_global_path.points)} waypoints.")
                self.recovery_controller.mark_replan_completed()
                self._pose_history.clear()
            else:
                self.get_logger().warn(f"Replan failed: {replan_res.reason}. Holding safe stop.")

        if rec_state == RecoveryState.REVERSING and rec_v is not None:
            cand_v = rec_v
            cand_w = rec_w or 0.0
            decision_str = f"RECOVERY_REVERSE"

        behavior_msg = RosBehavioralDecision()
        behavior_msg.header.stamp = self.get_clock().now().to_msg()
        behavior_msg.header.frame_id = "odom"
        behavior_msg.state = decision_str
        behavior_msg.command_v = float(cand_v)
        behavior_msg.command_omega = float(cand_w)
        self.decision_pub.publish(behavior_msg)

        # 8. Authoritative Safety Governor & Command Buffering
        governed_v, governed_w, gov_decision, gov_diag = self.safety_governor.evaluate(
            requested_v=cand_v,
            requested_w=cand_w,
            robot_pose=(self.ugv_x, self.ugv_y, self.ugv_yaw),
            hazards=all_hazards,
            current_time_s=t_now,
        )

        is_safe_cmd = (
            (decision_str not in ("STOP", "EMERGENCY_STOP"))
            and ("STOP" not in gov_decision)
            and (dist_to_goal > 0.35)
        )
        self.command_buffer.set_command(
            linear_v=governed_v,
            angular_w=governed_w,
            timestamp_s=t_now,
            is_safe=is_safe_cmd,
            source="SAFETY_GOVERNOR" if governed_v != cand_v else "DWA",
            reason=f"{decision_str} | {gov_decision}",
        )

        # 9. Publish Selected Trajectory
        if cand_traj is not None and len(cand_traj.points) > 0:
            traj_msg = Path()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.header.frame_id = "odom"
            for tp in cand_traj.points:
                ps = PoseStamped()
                ps.header = traj_msg.header
                ps.pose.position.x = float(tp.x)
                ps.pose.position.y = float(tp.y)
                ps.pose.position.z = 0.08
                traj_msg.poses.append(ps)
            self.traj_pub.publish(traj_msg)

        t0 = time.perf_counter()
        self._publish_visualizations(
            cand_traj, candidate_list, confirmed_tracks, active_predictions, decision_str, dist_to_goal
        )
        t_viz = (time.perf_counter() - t0) * 1000.0

        dt_ms = (time.perf_counter() - t_loop_start) * 1000.0
        self.get_logger().info(
            f"Step: pose=({self.ugv_x:.2f}, {self.ugv_y:.2f}, yaw={self.ugv_yaw:.2f}) "
            f"cmd=(v={cand_v:.2f}, w={cand_w:.2f}) dec={decision_str} | "
            f"tr={t_track:.0f}ms pr={t_pred:.0f}ms occ={t_occ:.0f}ms dwa={t_dwa:.0f}ms viz={t_viz:.0f}ms tot={dt_ms:.0f}ms"
        )

    def _publish_visualizations(
        self,
        best_traj: Any,
        candidates: List[Any],
        tracks: List[TrackedObstacleState],
        predictions: List[Any],
        decision_str: str,
        dist_to_goal: float,
    ) -> None:
        """Publishes all RViz judge-facing visual elements (HUD, candidate bundle, markers)."""
        m_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Marker 1: Static Rock Boulder Marker
        rock_marker = Marker()
        rock_marker.header.stamp = now
        rock_marker.header.frame_id = "odom"
        rock_marker.ns = "static_rock"
        rock_marker.id = 1
        rock_marker.type = Marker.CUBE
        rock_marker.action = Marker.ADD
        rock_marker.pose.position.x = self.rock_x
        rock_marker.pose.position.y = self.rock_y
        rock_marker.pose.position.z = 0.25
        rock_marker.scale.x = 1.1
        rock_marker.scale.y = 1.1
        rock_marker.scale.z = 0.5
        rock_marker.color = ColorRGBA(r=0.45, g=0.42, b=0.40, a=1.0)
        m_array.markers.append(rock_marker)

        # Marker 2: Static Rock Text Label
        rock_text = Marker()
        rock_text.header = rock_marker.header
        rock_text.ns = "static_rock_label"
        rock_text.id = 2
        rock_text.type = Marker.TEXT_VIEW_FACING
        rock_text.action = Marker.ADD
        rock_text.pose.position.x = self.rock_x
        rock_text.pose.position.y = self.rock_y
        rock_text.pose.position.z = 0.75
        rock_text.scale.z = 0.35
        rock_text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        rock_text.text = "STATIC OBSTACLE (BOULDER)"
        m_array.markers.append(rock_text)

        # Marker 3: Goal Beacon Marker
        goal_marker = Marker()
        goal_marker.header.stamp = now
        goal_marker.header.frame_id = "odom"
        goal_marker.ns = "goal_beacon"
        goal_marker.id = 3
        goal_marker.type = Marker.CYLINDER
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = self.goal_x
        goal_marker.pose.position.y = self.goal_y
        goal_marker.pose.position.z = 0.5
        goal_marker.scale.x = 0.7
        goal_marker.scale.y = 0.7
        goal_marker.scale.z = 1.0
        goal_marker.color = ColorRGBA(r=1.0, g=0.85, b=0.0, a=0.8)
        m_array.markers.append(goal_marker)

        # Marker 4: Goal Text
        goal_text = Marker()
        goal_text.header = goal_marker.header
        goal_text.ns = "goal_text"
        goal_text.id = 4
        goal_text.type = Marker.TEXT_VIEW_FACING
        goal_text.action = Marker.ADD
        goal_text.pose.position.x = self.goal_x
        goal_text.pose.position.y = self.goal_y
        goal_text.pose.position.z = 1.3
        goal_text.scale.z = 0.4
        goal_text.color = ColorRGBA(r=0.1, g=1.0, b=0.2, a=1.0)
        goal_text.text = f"GOAL (Dist: {dist_to_goal:.2f}m)"
        m_array.markers.append(goal_text)

        # Marker 5: DWA Candidate Rollout Trajectories (Semantic color-coding)
        marker_id = 100
        for item in candidates[:25]:  # Display top 25 representative candidates
            traj = item[1]
            dyn_res = item[4]
            line_m = Marker()
            line_m.header.stamp = now
            line_m.header.frame_id = "odom"
            line_m.ns = "dwa_candidates"
            line_m.id = marker_id
            marker_id += 1
            line_m.type = Marker.LINE_STRIP
            line_m.action = Marker.ADD
            line_m.scale.x = 0.03

            # Determine color: Safe (Green), Risk (Yellow), Collision (Red)
            if dyn_res is not None and dyn_res.collision:
                line_m.color = ColorRGBA(r=0.9, g=0.1, b=0.1, a=0.6)  # Red
            elif dyn_res is not None and str(getattr(dyn_res.risk_classification, "name", dyn_res.risk_classification)) in ("HIGH_RISK", "MEDIUM_RISK", "DynamicRiskLevel.HIGH_RISK", "DynamicRiskLevel.MEDIUM_RISK"):
                line_m.color = ColorRGBA(r=0.9, g=0.8, b=0.1, a=0.6)  # Yellow
            else:
                line_m.color = ColorRGBA(r=0.1, g=0.8, b=0.2, a=0.4)  # Green

            for tp in traj.points:
                line_m.points.append(Point(x=float(tp.x), y=float(tp.y), z=0.04))
            m_array.markers.append(line_m)

        # Marker 6: Dynamic Obstacle Tracks & Predicted Trajectory
        for tr in tracks:
            # Track Bounding Box
            track_m = Marker()
            track_m.header.stamp = now
            track_m.header.frame_id = "odom"
            track_m.ns = "dynamic_track"
            track_m.id = marker_id
            marker_id += 1
            track_m.type = Marker.CUBE
            track_m.action = Marker.ADD
            track_m.pose.position.x = float(tr.x)
            track_m.pose.position.y = float(tr.y)
            track_m.pose.position.z = 0.35
            track_m.scale.x = float(tr.bbox_size[0])
            track_m.scale.y = float(tr.bbox_size[1])
            track_m.scale.z = 0.6
            track_m.color = ColorRGBA(r=1.0, g=0.4, b=0.0, a=0.85)  # Orange
            m_array.markers.append(track_m)

            # Track Label
            speed = math.hypot(tr.vx, tr.vy)
            label_m = Marker()
            label_m.header = track_m.header
            label_m.ns = "dynamic_track_label"
            label_m.id = marker_id
            marker_id += 1
            label_m.type = Marker.TEXT_VIEW_FACING
            label_m.action = Marker.ADD
            label_m.pose.position.x = float(tr.x)
            label_m.pose.position.y = float(tr.y)
            label_m.pose.position.z = 0.9
            label_m.scale.z = 0.3
            label_m.color = ColorRGBA(r=1.0, g=0.9, b=0.2, a=1.0)
            label_m.text = f"TRACK #{tr.track_id} | V={speed:.2f} m/s"
            m_array.markers.append(label_m)

        # Marker 7: Prediction Horizons (+0.5s, +1.0s, +1.5s, +2.0s)
        for pr in predictions:
            pred_line = Marker()
            pred_line.header.stamp = now
            pred_line.header.frame_id = "odom"
            pred_line.ns = "pred_trajectory"
            pred_line.id = marker_id
            marker_id += 1
            pred_line.type = Marker.LINE_STRIP
            pred_line.action = Marker.ADD
            pred_line.scale.x = 0.05
            pred_line.color = ColorRGBA(r=1.0, g=0.2, b=0.8, a=0.8)  # Magenta

            for wp in pr.waypoints:
                pred_line.points.append(Point(x=float(wp.x), y=float(wp.y), z=0.08))
            m_array.markers.append(pred_line)

            # Waypoint timestamp markers
            for i, wp in enumerate(pr.waypoints):
                if i in (1, 3, 5, 7):  # Select representative time horizons
                    w_text = Marker()
                    w_text.header = pred_line.header
                    w_text.ns = "pred_time_label"
                    w_text.id = marker_id
                    marker_id += 1
                    w_text.type = Marker.TEXT_VIEW_FACING
                    w_text.action = Marker.ADD
                    w_text.pose.position.x = float(wp.x)
                    w_text.pose.position.y = float(wp.y)
                    w_text.pose.position.z = 0.3
                    w_text.scale.z = 0.2
                    w_text.color = ColorRGBA(r=1.0, g=0.4, b=1.0, a=1.0)
                    dt_s = (i + 1) * 0.25
                    w_text.text = f"+{dt_s:.1f}s"
                    m_array.markers.append(w_text)

        # Marker 8: Judge-Facing Heads-Up Display (HUD) Text Overlay
        hud_m = Marker()
        hud_m.header.stamp = now
        hud_m.header.frame_id = "base_link"
        hud_m.ns = "judge_hud"
        hud_m.id = 999
        hud_m.type = Marker.TEXT_VIEW_FACING
        hud_m.action = Marker.ADD
        hud_m.pose.position.x = 1.0
        hud_m.pose.position.y = 0.0
        hud_m.pose.position.z = 1.5
        hud_m.scale.z = 0.28

        # Color decision: PROCEED (Green), AVOID (Yellow), WAIT (Orange), STOP (Red)
        if decision_str == "PROCEED":
            hud_m.color = ColorRGBA(r=0.2, g=1.0, b=0.2, a=1.0)
        elif decision_str == "AVOID":
            hud_m.color = ColorRGBA(r=1.0, g=0.9, b=0.1, a=1.0)
        elif decision_str == "WAIT":
            hud_m.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
        else:
            hud_m.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)

        model_tag = "LightweightTraversabilityNet (RELLIS-3D)" if self.traversability_predictor else "Default Core"
        hud_m.text = (
            f"=== DEMO-01 JUDGE MONITOR ===\n"
            f"MODEL STATUS   : {model_tag}\n"
            f"INFERENCE      : {self.last_trav_fps:.1f} FPS ({self.last_trav_ms:.1f} ms)\n"
            f"BEHAVIOR STATE : [ {decision_str} ]\n"
            f"UGV SPEED      : {self.ugv_v:.2f} m/s | ROT : {self.ugv_w:.2f} rad/s\n"
            f"GOAL DISTANCE  : {dist_to_goal:.2f} m\n"
            f"ACTIVE TRACKS  : {len(tracks)} | CANDIDATES: {len(candidates)}"
        )
        m_array.markers.append(hud_m)

        self.marker_pub.publish(m_array)


def main(args=None):
    rclpy.init(args=args)
    node = InteractiveUGVNavigationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception as e:
        if rclpy.ok():
            node.get_logger().error(f"Node exception: {e}")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
