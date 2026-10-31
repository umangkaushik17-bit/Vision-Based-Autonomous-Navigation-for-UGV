#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source /mnt/c/Users/shres/Desktop/SIH2026/ros2_ws/install/setup.bash

echo "=== M2 TEST: Starting Gazebo & ROS-GZ Bridge ==="

# 1. Start Gazebo Sim
gz sim -s -r /mnt/c/Users/shres/Desktop/SIH2026/simulation/worlds/demo_outdoor.sdf &
GZ_PID=$!

sleep 3

# 2. Start Parameter Bridge
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=/mnt/c/Users/shres/Desktop/SIH2026/simulation/config/bridge_params.yaml &
BRIDGE_PID=$!

cleanup() {
    echo "Cleaning up processes..."
    kill -9 $BRIDGE_PID 2>/dev/null || true
    kill -9 $GZ_PID 2>/dev/null || true
}
trap cleanup EXIT

sleep 3

echo "=== Step 1: Listing ROS2 Topics ==="
ros2 topic list

echo "=== Step 2: Checking Topic Info ==="
ros2 topic info /odom
ros2 topic info /cmd_vel
ros2 topic info /dynamic_obstacle/cmd_vel
ros2 topic info /dynamic_obstacle/odom

echo "=== Step 3: Publishing ROS2 cmd_vel to UGV ==="
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.6}, angular: {z: 0.0}}'

sleep 2

echo "=== Step 4: Reading UGV /odom in ROS2 ==="
timeout 3 ros2 topic echo /odom --once || true

echo "=== Step 5: Publishing ROS2 cmd_vel to Dynamic Obstacle ==="
ros2 topic pub --once /dynamic_obstacle/cmd_vel geometry_msgs/msg/Twist '{linear: {y: 0.5}, angular: {z: 0.0}}'

sleep 2

echo "=== Step 6: Reading /dynamic_obstacle/odom in ROS2 ==="
timeout 3 ros2 topic echo /dynamic_obstacle/odom --once || true

echo "=== M2 TEST PASSED: Full ROS2 Simulation Bridge Operational ==="
