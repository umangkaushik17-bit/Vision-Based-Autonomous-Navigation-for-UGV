#!/usr/bin/env python3
"""Interactive Multi-Zone Dynamic Obstacle Keyboard Teleoperation Node.

Provides responsive real-time keyboard control of dynamic obstacles across
the 100m corridor with automatic UGV-zone-based handoff:
- W1 Zone (x < 20m)     : Controls dynamic_obs_w1 (Oncoming straight towards UGV)
- W2 Zone (20m <= x < 40m): Controls dynamic_obs_w2 (Crossing from left)
- W3 Zone (40m <= x < 60m): Controls dynamic_obs_w3 (Crossing from right)
- W4 Zone (60m <= x < 80m): Controls dynamic_obs_w4 (Diagonal oncoming intercept)
- W5 Zone (x >= 80m)    : Controls dynamic_obs_w5 (Sentinel blocker near gateway)

Automatic Zone Handoff:
- Automatically transfers WASD control as the UGV crosses waypoint boundaries.
- Halts the previously controlled obstacle with zero velocity during handoff.

Controls:
- 'W' / 'w': Drive forward in world +X
- 'S' / 's': Drive backward in world -X
- 'A' / 'a': Drive towards UGV corridor (world +Y)
- 'D' / 'd': Drive away from UGV corridor (world -Y)
- 'SPACE'  : Stop active obstacle
- '1' - '5': Manually select obstacle W1 - W5
- '0'      : Re-enable AUTO zone tracking based on UGV progress
- 'R' / 'r': Reset Demo (Resets UGV, all 5 obstacles, tracker, and planner)
- 'Q' / 'q': Quit Controller
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
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

OBSTACLE_CONFIGS = {
    "dynamic_obs_w1": {
        "label": "W1 Corridor (Oncoming / Lateral Crossing)",
        "zone_x": (0.0, 20.0),
        "standby_pose": {"x": 16.0, "y": 3.2, "z": 0.35, "yaw": -math.pi / 2.0},
        "color": "Crimson Red",
    },
    "dynamic_obs_w2": {
        "label": "W2 Corridor (Left Lateral Crossing)",
        "zone_x": (20.0, 40.0),
        "standby_pose": {"x": 32.0, "y": 7.5, "z": 0.35, "yaw": -math.pi / 2.0},
        "color": "Cyber Cyan",
    },
    "dynamic_obs_w3": {
        "label": "W3 Corridor (Right Lateral Crossing)",
        "zone_x": (40.0, 60.0),
        "standby_pose": {"x": 55.0, "y": -7.5, "z": 0.35, "yaw": math.pi / 2.0},
        "color": "Safety Gold",
    },
    "dynamic_obs_w4": {
        "label": "W4 Corridor (Diagonal Intercept)",
        "zone_x": (60.0, 80.0),
        "standby_pose": {"x": 73.0, "y": 5.5, "z": 0.35, "yaw": -2.356},
        "color": "Electric Magenta",
    },
    "dynamic_obs_w5": {
        "label": "W5 Corridor (Gateway Sentinel)",
        "zone_x": (80.0, 115.0),
        "standby_pose": {"x": 94.0, "y": 2.5, "z": 0.35, "yaw": -math.pi / 2.0},
        "color": "Neon Lime",
    },
}

OBSTACLE_KEYS = list(OBSTACLE_CONFIGS.keys())

INSTRUCTIONS = """
=============================================================================
        DEMO-01 MULTI-ZONE DYNAMIC OBSTACLE KEYBOARD CONTROLLER
=============================================================================
  W / S   : Drive Forward (+X) / Backward (-X) in World Frame
  A / D   : Drive Left (+Y) / Right (-Y) in World Frame
  SPACE   : Immediate Stop Active Obstacle
  1 - 5   : Force Manual Control of Obstacle W1 - W5
  0       : Restore AUTO Zone Tracking (Follows UGV Progress)
  R / r   : Full Simulation Reset (UGV + All 5 Dynamic Obstacles)
  Q / q   : Quit Teleoperation
-----------------------------------------------------------------------------
  AUTO HANDOFF ACTIVE:
    UGV x < 20m    -> W1 (Head-on)
    UGV 20m..40m   -> W2 (Left Cross)
    UGV 40m..60m   -> W3 (Right Cross)
    UGV 60m..80m   -> W4 (Diagonal Intercept)
    UGV x >= 80m   -> W5 (Sentinel Gateway)
=============================================================================
"""


class TeleopObstacleKeyboard(Node):
    def __init__(self) -> None:
        super().__init__("teleop_obstacle_keyboard")

        self.speed: float = 0.8  # m/s
        self.cmd_vx: float = 0.0
        self.cmd_vy: float = 0.0

        # Mode: "AUTO" or manual target name
        self.mode: str = "AUTO"
        self.active_target: str = "dynamic_obs_w1"

        # UGV Odometry State
        self.ugv_x: float = 0.0
        self.ugv_y: float = 0.0
        self.ugv_odom_received: bool = False

        # Obstacle Orientation & Position Cache
        self.obs_yaw: Dict[str, float] = {k: 0.0 for k in OBSTACLE_KEYS}
        self.obs_pos: Dict[str, Tuple[float, float]] = {k: (0.0, 0.0) for k in OBSTACLE_KEYS}

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

        # Publishers for all 5 obstacles
        self.cmd_pubs: Dict[str, Any] = {}
        for name in OBSTACLE_KEYS:
            self.cmd_pubs[name] = self.create_publisher(Twist, f"/{name}/cmd_vel", state_qos)

        # Legacy publisher alias for dynamic_obstacle
        self.legacy_pub = self.create_publisher(Twist, "/dynamic_obstacle/cmd_vel", state_qos)
        self.reset_pub = self.create_publisher(Empty, "/demo/reset", state_qos)

        # UGV odometry subscriber for zone handoff
        self.create_subscription(Odometry, "/odom", self._on_ugv_odom, sensor_qos)

        # Subscribe to all 5 obstacle odometry topics
        for name in OBSTACLE_KEYS:
            self._create_obs_sub(name, sensor_qos)

        # Periodic command publishing timer (10 Hz)
        self.timer = self.create_timer(0.10, self._publish_cmd)

        self.get_logger().info("Multi-Zone Dynamic Obstacle Teleop initialized in AUTO mode.")

    def _create_obs_sub(self, name: str, qos: QoSProfile) -> None:
        def odom_callback(msg: Odometry) -> None:
            self.obs_pos[name] = (msg.pose.pose.position.x, msg.pose.pose.position.y)
            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.obs_yaw[name] = math.atan2(siny_cosp, cosy_cosp)

        self.create_subscription(Odometry, f"/{name}/odom", odom_callback, qos)

    def _determine_zone_target(self, x: float) -> str:
        """Determines active obstacle based on UGV corridor x-coordinate."""
        if x < 20.0:
            return "dynamic_obs_w1"
        elif x < 40.0:
            return "dynamic_obs_w2"
        elif x < 60.0:
            return "dynamic_obs_w3"
        elif x < 80.0:
            return "dynamic_obs_w4"
        else:
            return "dynamic_obs_w5"

    def _on_ugv_odom(self, msg: Odometry) -> None:
        self.ugv_x = msg.pose.pose.position.x
        self.ugv_y = msg.pose.pose.position.y
        self.ugv_odom_received = True

        if self.mode == "AUTO":
            expected_target = self._determine_zone_target(self.ugv_x)
            if expected_target != self.active_target:
                prev_target = self.active_target
                # Halt previous obstacle immediately
                self._stop_target(prev_target)
                self.active_target = expected_target
                self.cmd_vx = 0.0
                self.cmd_vy = 0.0

                cfg = OBSTACLE_CONFIGS[expected_target]
                print(
                    f"\n"
                    f"=====================================================================\n"
                    f"  >>> AUTO ZONE HANDOFF TRIGGERED! <<<\n"
                    f"  UGV Position: x={self.ugv_x:.2f}m, y={self.ugv_y:.2f}m\n"
                    f"  Now Controlling: [{expected_target}] ({cfg['label']})\n"
                    f"  Active Color: {cfg['color']} | Prior [{prev_target}] halted.\n"
                    f"=====================================================================\n"
                )

    def _publish_cmd(self) -> None:
        """Transforms world velocities into active obstacle's body frame."""
        twist = Twist()
        if abs(self.cmd_vx) > 1e-3 or abs(self.cmd_vy) > 1e-3:
            yaw = self.obs_yaw.get(self.active_target, 0.0)
            c = math.cos(yaw)
            s = math.sin(yaw)
            # Body coordinates transformation
            twist.linear.x = self.cmd_vx * c + self.cmd_vy * s
            twist.linear.y = -self.cmd_vx * s + self.cmd_vy * c

        pub = self.cmd_pubs.get(self.active_target)
        if pub:
            pub.publish(twist)

        # Maintain legacy dynamic_obstacle topic compatibility if W3
        if self.active_target == "dynamic_obs_w3":
            self.legacy_pub.publish(twist)

    def _stop_target(self, target_name: str) -> None:
        """Publishes explicit zero twist to halt target."""
        tw = Twist()
        pub = self.cmd_pubs.get(target_name)
        if pub:
            pub.publish(tw)
        if target_name == "dynamic_obs_w3":
            self.legacy_pub.publish(tw)

    def set_motion(self, vx: float, vy: float) -> None:
        self.cmd_vx = float(vx)
        self.cmd_vy = float(vy)

    def stop(self) -> None:
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self._stop_target(self.active_target)

    def set_manual_target(self, idx: int) -> None:
        if 1 <= idx <= len(OBSTACLE_KEYS):
            target = OBSTACLE_KEYS[idx - 1]
            self._stop_target(self.active_target)
            self.active_target = target
            self.mode = "MANUAL"
            self.cmd_vx = 0.0
            self.cmd_vy = 0.0
            cfg = OBSTACLE_CONFIGS[target]
            print(f"\n[MANUAL OVERRIDE] Target forced to: {target} ({cfg['label']})\n")

    def restore_auto_mode(self) -> None:
        self.mode = "AUTO"
        expected = self._determine_zone_target(self.ugv_x)
        if expected != self.active_target:
            self._stop_target(self.active_target)
            self.active_target = expected
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        cfg = OBSTACLE_CONFIGS[self.active_target]
        print(f"\n[AUTO MODE RESTORED] Tracking UGV (x={self.ugv_x:.1f}m) -> Target: {self.active_target} ({cfg['label']})\n")

    def reset_demo(self) -> None:
        """Triggers complete simulation and node reset for all entities."""
        self.get_logger().info("RESET DEMO TRIGGERED ('R')")
        # 1. Stop all dynamic obstacles
        for name in OBSTACLE_KEYS:
            self._stop_target(name)
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0

        # 2. Reset Gazebo models via set_pose service
        try:
            # Reset UGV to (0, 0, 0.15)
            subprocess.run([
                "gz", "service", "-s", "/world/outdoor_presentation_100m/set_pose",
                "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                "--timeout", "1000",
                "--req", 'name: "ugv_4wd", position: {x: 0.0, y: 0.0, z: 0.15}, orientation: {x: 0, y: 0, z: 0, w: 1}'
            ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Reset each dynamic obstacle to its designated standby pose
            for name, cfg in OBSTACLE_CONFIGS.items():
                p = cfg["standby_pose"]
                cy = math.cos(p["yaw"] * 0.5)
                sy = math.sin(p["yaw"] * 0.5)
                req_str = (
                    f'name: "{name}", position: {{x: {p["x"]}, y: {p["y"]}, z: {p["z"]}}}, '
                    f'orientation: {{x: 0, y: 0, z: {sy:.4f}, w: {cy:.4f}}}'
                )
                subprocess.run([
                    "gz", "service", "-s", "/world/outdoor_presentation_100m/set_pose",
                    "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                    "--timeout", "1000",
                    "--req", req_str
                ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        except Exception as ex:
            self.get_logger().warn(f"Failed to reset Gazebo pose: {ex}")

        # 3. Publish ROS Reset signal
        empty_msg = Empty()
        self.reset_pub.publish(empty_msg)
        print("\n>>> DEMO RESET COMPLETE (UGV @ origin, All 5 Obstacles @ standby, Planner reset) <<<\n")


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

            target_label = OBSTACLE_CONFIGS[teleop.active_target]["label"]

            if key in ("w", "W"):
                teleop.set_motion(teleop.speed, 0.0)
                print(f"[{teleop.active_target}] Move Forward (+X) -> V=({teleop.cmd_vx:.1f}, {teleop.cmd_vy:.1f})")
            elif key in ("s", "S"):
                teleop.set_motion(-teleop.speed, 0.0)
                print(f"[{teleop.active_target}] Move Backward (-X) -> V=({teleop.cmd_vx:.1f}, {teleop.cmd_vy:.1f})")
            elif key in ("a", "A"):
                teleop.set_motion(0.0, teleop.speed)
                print(f"[{teleop.active_target}] Intercept Left (+Y) -> V=({teleop.cmd_vx:.1f}, {teleop.cmd_vy:.1f})")
            elif key in ("d", "D"):
                teleop.set_motion(0.0, -teleop.speed)
                print(f"[{teleop.active_target}] Clear Right (-Y) -> V=({teleop.cmd_vx:.1f}, {teleop.cmd_vy:.1f})")
            elif key == " ":
                teleop.stop()
                print(f"[{teleop.active_target}] Stopped (SPACE)")
            elif key in ("1", "2", "3", "4", "5"):
                teleop.set_manual_target(int(key))
            elif key == "0":
                teleop.restore_auto_mode()
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
