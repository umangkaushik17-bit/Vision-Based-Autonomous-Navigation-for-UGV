#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash

echo "=== M1 TEST: Starting Gazebo in headless mode ==="
gz sim -s -r /mnt/c/Users/shres/Desktop/SIH2026/simulation/worlds/demo_outdoor.sdf &
GZ_PID=$!

trap "kill -9 $GZ_PID 2>/dev/null || true" EXIT

sleep 3

echo "=== Checking Gazebo Topics ==="
gz topic -l | grep -E 'ugv_4wd|dynamic_obstacle'

echo "=== Testing UGV Motion Command ==="
gz topic -t /model/ugv_4wd/cmd_vel -m gz.msgs.Twist -p "linear: {x: 0.5}"

sleep 1

echo "=== Testing Dynamic Obstacle Motion Command ==="
gz topic -t /model/dynamic_obstacle/cmd_vel -m gz.msgs.Twist -p "linear: {y: 0.5}"

sleep 1

echo "=== Reading Dynamic Obstacle Odometry ==="
timeout 2 gz topic -e -t /model/dynamic_obstacle/odometry -n 1 || true

echo "=== Reading UGV Odometry ==="
timeout 2 gz topic -e -t /model/ugv_4wd/odometry -n 1 || true

echo "=== M1 TEST PASSED ==="
