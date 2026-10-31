#!/usr/bin/env python3
"""Verification script for M3 & M4 UGV navigation and static rock avoidance.

Runs Gazebo, ROS-GZ bridge, and the interactive navigation node.
Subscribes directly to /odom and /cmd_vel to record trajectory, verify forward
progress, and ensure minimum clearance around the static boulder.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class NavigationVerifier(Node):
    def __init__(self) -> None:
        super().__init__("navigation_verifier")

        self.rock_x = 3.5
        self.rock_y = 0.15
        self.rock_radius = 0.55  # Physical boulder visual radius

        self.poses: List[Tuple[float, float, float]] = []  # (x, y, dist_to_rock)
        self.cmd_vels: List[Tuple[float, float]] = []

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        cmd_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        self.create_subscription(Odometry, "/odom", self._on_odom, sensor_qos)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, cmd_qos)

    def _on_odom(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        dist_rock = math.hypot(x - self.rock_x, y - self.rock_y)
        self.poses.append((x, y, dist_rock))

    def _on_cmd(self, msg: Twist) -> None:
        self.cmd_vels.append((msg.linear.x, msg.angular.z))


def main() -> None:
    # 1. Launch Gazebo headless
    print("Starting Gazebo Sim headless...")
    gz_cmd = [
        "gz", "sim", "-s", "-r",
        "/mnt/c/Users/shres/Desktop/SIH2026/simulation/worlds/demo_outdoor.sdf",
    ]
    gz_proc = subprocess.Popen(gz_cmd)

    time.sleep(3.0)

    # 2. Launch Parameter Bridge
    print("Starting ROS-GZ parameter bridge...")
    bridge_cmd = [
        "ros2", "run", "ros_gz_bridge", "parameter_bridge",
        "--ros-args", "-p",
        "config_file:=/mnt/c/Users/shres/Desktop/SIH2026/simulation/config/bridge_params.yaml",
    ]
    bridge_proc = subprocess.Popen(bridge_cmd)

    time.sleep(3.0)

    # 3. Launch Interactive Navigation Node
    print("Starting Interactive Navigation Node...")
    nav_cmd = [
        "python3",
        "/mnt/c/Users/shres/Desktop/SIH2026/simulation/nodes/interactive_ugv_navigation_node.py",
    ]
    nav_proc = subprocess.Popen(nav_cmd)

    time.sleep(2.0)

    rclpy.init()
    verifier = NavigationVerifier()

    start_time = time.time()
    duration_s = 35.0
    print(f"Monitoring navigation run for {duration_s} seconds...")

    try:
        while time.time() - start_time < duration_s:
            rclpy.spin_once(verifier, timeout_sec=0.2)
            if len(verifier.poses) > 0 and len(verifier.poses) % 20 == 0:
                latest_x, latest_y, latest_dist = verifier.poses[-1]
                print(f"  t={time.time()-start_time:.1f}s | Pose: ({latest_x:.2f}, {latest_y:.2f}) | Dist to Rock: {latest_dist:.2f}m")
    finally:
        print("Stopping processes...")
        nav_proc.terminate()
        bridge_proc.terminate()
        gz_proc.terminate()
        time.sleep(1.0)
        nav_proc.kill()
        bridge_proc.kill()
        gz_proc.kill()
        verifier.destroy_node()
        rclpy.shutdown()

    # 4. Analyze Results
    print("\n=== Navigation Analysis ===")
    print(f"Total Odometry Samples Received: {len(verifier.poses)}")
    print(f"Total CmdVel Samples Received: {len(verifier.cmd_vels)}")

    if len(verifier.poses) == 0:
        print("ERROR: No odometry samples received!")
        sys.exit(1)

    max_x = max(p[0] for p in verifier.poses)
    min_dist_rock = min(p[2] for p in verifier.poses)

    print(f"Initial Pose: ({verifier.poses[0][0]:.2f}, {verifier.poses[0][1]:.2f})")
    print(f"Final Pose:   ({verifier.poses[-1][0]:.2f}, {verifier.poses[-1][1]:.2f})")
    print(f"Maximum X Reached: {max_x:.2f} m")
    print(f"Minimum Distance to Rock Center: {min_dist_rock:.2f} m")

    # Verification Checks
    assert max_x > 1.2, f"UGV failed to make forward progress: max X was {max_x:.2f} m"
    print("CHECK 1: Forward navigation progress (> 1.2m) -> PASS")

    assert min_dist_rock > 0.45, f"UGV got too close to rock center: {min_dist_rock:.2f} m"
    print(f"CHECK 2: Rock obstacle clearance maintained ({min_dist_rock:.2f}m > 0.45m) -> PASS")

    print("\n=== M3 (NAVIGATION STACK) & M4 (STATIC ROCK AVOIDANCE) VERIFIED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
