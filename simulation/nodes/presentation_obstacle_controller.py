#!/usr/bin/env python3
"""Multi-Obstacle Presentation Controller for Phase 15/16 100m Corridor.

Manages autonomous patrol patterns for all 5 dynamic corridor obstacles:
- dynamic_obs_w1: Head-on oncoming patrol along x in [11.5, 17.5] (speed ~0.35 m/s)
- dynamic_obs_w2: Left lateral cross patrol along y in [-2.0, 7.5] (speed ~0.40 m/s)
- dynamic_obs_w3: Right lateral cross patrol along y in [-7.5, 2.0] (speed ~0.40 m/s)
- dynamic_obs_w4: Diagonal oncoming cross patrol along (-0.707, -0.707) (speed ~0.35 m/s)
- dynamic_obs_w5: Gateway sentinel cross patrol along y in [-2.5, 2.0] (speed ~0.30 m/s)

Coordinates live odometry heading for clean world-to-body Cartesian transformation.
Gracefully halts all entities on reset or shutdown.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

PATROL_CONFIGS = {
    "dynamic_obs_w1": {
        "type": "axial_x",
        "min_x": 11.5,
        "max_x": 17.5,
        "speed": 0.35,
        "init_dir": -1.0,  # Moving in -x towards oncoming UGV
    },
    "dynamic_obs_w2": {
        "type": "axial_y",
        "min_y": -2.0,
        "max_y": 7.5,
        "speed": 0.40,
        "init_dir": -1.0,  # Moving in -y from left across trail
    },
    "dynamic_obs_w3": {
        "type": "axial_y",
        "min_y": -7.5,
        "max_y": 2.0,
        "speed": 0.40,
        "init_dir": 1.0,   # Moving in +y from right across trail
    },
    "dynamic_obs_w4": {
        "type": "diagonal",
        "min_x": 68.0,
        "max_x": 74.0,
        "min_y": 0.5,
        "max_y": 6.0,
        "speed": 0.35,
        "init_dir": -1.0,  # Moving diagonally into chicane
    },
    "dynamic_obs_w5": {
        "type": "axial_y",
        "min_y": -2.5,
        "max_y": 2.0,
        "speed": 0.30,
        "init_dir": 1.0,   # Sentinel sweeping across gateway
    },
}

OBSTACLE_KEYS = list(PATROL_CONFIGS.keys())


class PresentationObstacleController(Node):
    def __init__(self) -> None:
        super().__init__("presentation_obstacle_controller")

        self.declare_parameter("enable_patrol", True)
        self.declare_parameter("control_rate_hz", 10.0)

        self.enable_patrol: bool = bool(self.get_parameter("enable_patrol").value)
        self.control_rate_hz: float = float(self.get_parameter("control_rate_hz").value)

        # State tracking for each obstacle
        self.obs_x: Dict[str, float] = {k: 0.0 for k in OBSTACLE_KEYS}
        self.obs_y: Dict[str, float] = {k: 0.0 for k in OBSTACLE_KEYS}
        self.obs_yaw: Dict[str, float] = {k: 0.0 for k in OBSTACLE_KEYS}
        self.odom_received: Dict[str, bool] = {k: False for k in OBSTACLE_KEYS}
        self.direction: Dict[str, float] = {k: PATROL_CONFIGS[k]["init_dir"] for k in OBSTACLE_KEYS}

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        state_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )

        # Publishers & Subscribers
        self.cmd_pubs: Dict[str, Any] = {}
        for name in OBSTACLE_KEYS:
            self.cmd_pubs[name] = self.create_publisher(Twist, f"/{name}/cmd_vel", state_qos)
            self._create_obs_sub(name, sensor_qos)

        # Legacy publisher alias for dynamic_obstacle (mapped to W3)
        self.legacy_pub = self.create_publisher(Twist, "/dynamic_obstacle/cmd_vel", state_qos)
        self.create_subscription(Empty, "/demo/reset", self._on_reset, state_qos)

        dt = 1.0 / max(1.0, self.control_rate_hz)
        self.timer = self.create_timer(dt, self._control_loop)

        self.get_logger().info(
            f"Multi-Obstacle Presentation Controller active (patrol={self.enable_patrol}, "
            f"managing {len(OBSTACLE_KEYS)} dynamic obstacles)."
        )

    def _create_obs_sub(self, name: str, qos: QoSProfile) -> None:
        def odom_callback(msg: Odometry) -> None:
            self.obs_x[name] = msg.pose.pose.position.x
            self.obs_y[name] = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.obs_yaw[name] = math.atan2(siny_cosp, cosy_cosp)
            self.odom_received[name] = True

        self.create_subscription(Odometry, f"/{name}/odom", odom_callback, qos)

    def _on_reset(self, _: Empty) -> None:
        self.get_logger().info("Reset signal received. Resetting patrol directions and halting velocities.")
        for name in OBSTACLE_KEYS:
            self.direction[name] = PATROL_CONFIGS[name]["init_dir"]
        self._publish_stop()

    def _publish_stop(self) -> None:
        tw = Twist()
        for pub in self.cmd_pubs.values():
            try:
                pub.publish(tw)
            except Exception:
                pass
        try:
            self.legacy_pub.publish(tw)
        except Exception:
            pass

    def _control_loop(self) -> None:
        if not self.enable_patrol:
            return

        for name, cfg in PATROL_CONFIGS.items():
            if not self.odom_received.get(name, False):
                continue

            ptype = cfg["type"]
            speed = cfg["speed"]
            cur_dir = self.direction[name]
            px = self.obs_x[name]
            py = self.obs_y[name]
            yaw = self.obs_yaw[name]

            vx_w = 0.0
            vy_w = 0.0

            if ptype == "axial_x":
                min_x = cfg["min_x"]
                max_x = cfg["max_x"]
                if px <= min_x and cur_dir < 0:
                    self.direction[name] = 1.0
                elif px >= max_x and cur_dir > 0:
                    self.direction[name] = -1.0
                vx_w = self.direction[name] * speed
                vy_w = 0.0

            elif ptype == "axial_y":
                min_y = cfg["min_y"]
                max_y = cfg["max_y"]
                if py <= min_y and cur_dir < 0:
                    self.direction[name] = 1.0
                elif py >= max_y and cur_dir > 0:
                    self.direction[name] = -1.0
                vx_w = 0.0
                vy_w = self.direction[name] * speed

            elif ptype == "diagonal":
                min_x, max_x = cfg["min_x"], cfg["max_x"]
                min_y, max_y = cfg["min_y"], cfg["max_y"]
                if (px <= min_x or py <= min_y) and cur_dir < 0:
                    self.direction[name] = 1.0
                elif (px >= max_x or py >= max_y) and cur_dir > 0:
                    self.direction[name] = -1.0
                diag_norm = math.sqrt(2.0)
                vx_w = self.direction[name] * (speed / diag_norm)
                vy_w = self.direction[name] * (speed / diag_norm)

            # Transform world velocities to body velocities
            c = math.cos(yaw)
            s = math.sin(yaw)
            tw = Twist()
            tw.linear.x = vx_w * c + vy_w * s
            tw.linear.y = -vx_w * s + vy_w * c

            pub = self.cmd_pubs.get(name)
            if pub:
                pub.publish(tw)
            if name == "dynamic_obs_w3":
                self.legacy_pub.publish(tw)


def main(args=None):
    rclpy.init(args=args)
    node = PresentationObstacleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
