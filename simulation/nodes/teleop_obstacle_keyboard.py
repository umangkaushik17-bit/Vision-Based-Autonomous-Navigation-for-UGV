#!/usr/bin/env python3
"""Interactive Dynamic Obstacle Keyboard Teleoperation Node for DEMO-01.

Provides responsive real-time keyboard control of the dynamic obstacle:
- 'W' / 'w': Drive forward in world +X
- 'S' / 's': Drive backward in world -X
- 'A' / 'a': Drive towards UGV path (world +Y)
- 'D' / 'd': Drive away from UGV path (world -Y)
- 'SPACE'  : Emergency Stop
- 'R' / 'r': Reset Demo (Resets UGV, dynamic obstacle, tracker, and planner)
- 'Q' / 'q': Quit

Maps desired Cartesian velocities into the dynamic obstacle's body coordinates
using live odometry orientation feedback.
"""

from __future__ import annotations

import math
import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

INSTRUCTIONS = """
=============================================================
  DEMO-01 DYNAMIC OBSTACLE KEYBOARD CONTROLLER
=============================================================
  W / S : Drive Forward (+X) / Backward (-X)
  A / D : Drive Into UGV Path (+Y) / Away (-Y)
  SPACE : Stop Dynamic Obstacle
  R / r : RESET DEMO (UGV, Obstacle, Planner, & Tracker)
  Q / q : Quit Controller
=============================================================
"""


class TeleopObstacleKeyboard(Node):
    def __init__(self) -> None:
        super().__init__("teleop_obstacle_keyboard")

        self.speed: float = 0.8  # m/s
        self.obs_yaw: float = 1.570796  # Body yaw in odom frame
        self.cmd_vx: float = 0.0
        self.cmd_vy: float = 0.0

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

        self.cmd_pub = self.create_publisher(Twist, "/dynamic_obstacle/cmd_vel", state_qos)
        self.reset_pub = self.create_publisher(Empty, "/demo/reset", state_qos)
        self.create_subscription(Odometry, "/dynamic_obstacle/odom", self._on_odom, sensor_qos)

        # Periodic command publishing timer (10 Hz)
        self.timer = self.create_timer(0.10, self._publish_cmd)

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.obs_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _publish_cmd(self) -> None:
        """Transforms desired world Cartesian velocity into obstacle body frame."""
        twist = Twist()
        if abs(self.cmd_vx) > 1e-3 or abs(self.cmd_vy) > 1e-3:
            # Body coordinates transformation
            # v_body_x = vx_w * cos(yaw) + vy_w * sin(yaw)
            # v_body_y = -vx_w * sin(yaw) + vy_w * cos(yaw)
            c = math.cos(self.obs_yaw)
            s = math.sin(self.obs_yaw)
            twist.linear.x = self.cmd_vx * c + self.cmd_vy * s
            twist.linear.y = -self.cmd_vx * s + self.cmd_vy * c
        self.cmd_pub.publish(twist)

    def set_motion(self, vx: float, vy: float) -> None:
        self.cmd_vx = float(vx)
        self.cmd_vy = float(vy)

    def stop(self) -> None:
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self._publish_cmd()

    def reset_demo(self) -> None:
        """Triggers complete simulation and node reset."""
        self.get_logger().info("RESET DEMO TRIGGERED ('R')")
        # 1. Stop dynamic obstacle
        self.stop()

        # 2. Reset Gazebo models via set_pose service
        try:
            # Reset UGV to (0, 0, 0.15)
            subprocess.run([
                "gz", "service", "-s", "/world/demo_outdoor/set_pose",
                "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                "--timeout", "1000",
                "--req", 'name: "ugv_4wd", position: {x: 0.0, y: 0.0, z: 0.15}, orientation: {x: 0, y: 0, z: 0, w: 1}'
            ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Reset Dynamic Obstacle to (2.5, -3.0, 0.25) yaw=1.5708
            subprocess.run([
                "gz", "service", "-s", "/world/demo_outdoor/set_pose",
                "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                "--timeout", "1000",
                "--req", 'name: "dynamic_obstacle", position: {x: 2.5, y: -3.0, z: 0.25}, orientation: {x: 0, y: 0, z: 0.7071, w: 0.7071}'
            ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as ex:
            self.get_logger().warn(f"Failed to reset Gazebo pose: {ex}")

        # 3. Publish ROS Reset signal
        empty_msg = Empty()
        self.reset_pub.publish(empty_msg)
        print("\n>>> DEMO RESET COMPLETE (UGV @ origin, Obstacle @ standby, Planner reset) <<<\n")


def get_key(settings) -> str:
    """Reads single character from terminal without Enter."""
    if not sys.stdin.isatty():
        time.sleep(0.1)
        return ""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main(args=None):
    rclpy.init(args=args)
    teleop = TeleopObstacleKeyboard()

    # Spin ROS node in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(teleop,), daemon=True)
    spin_thread.start()

    print(INSTRUCTIONS)

    if not sys.stdin.isatty():
        print("Non-interactive terminal detected. Running teleop node in background.")
        try:
            while rclpy.ok():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        return

    settings = termios.tcgetattr(sys.stdin)

    try:
        while rclpy.ok():
            key = get_key(settings)
            if not key:
                continue

            if key in ("w", "W"):
                teleop.set_motion(teleop.speed, 0.0)
                print(f"[MANEUVER] Move Forward (+X) -> V=({teleop.cmd_vx:.1f}, {teleop.cmd_vy:.1f})")
            elif key in ("s", "S"):
                teleop.set_motion(-teleop.speed, 0.0)
                print(f"[MANEUVER] Move Backward (-X) -> V=({teleop.cmd_vx:.1f}, {teleop.cmd_vy:.1f})")
            elif key in ("a", "A"):
                teleop.set_motion(0.0, teleop.speed)
                print(f"[MANEUVER] Intercept UGV Path (+Y) -> V=({teleop.cmd_vx:.1f}, {teleop.cmd_vy:.1f})")
            elif key in ("d", "D"):
                teleop.set_motion(0.0, -teleop.speed)
                print(f"[MANEUVER] Clear UGV Path (-Y) -> V=({teleop.cmd_vx:.1f}, {teleop.cmd_vy:.1f})")
            elif key == " ":
                teleop.stop()
                print("[MANEUVER] Dynamic Obstacle Stopped (SPACE)")
            elif key in ("r", "R"):
                teleop.reset_demo()
            elif key in ("q", "Q", "\x03"):  # 'q' or Ctrl+C
                print("Exiting teleoperation...")
                break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        teleop.stop()
        teleop.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
