#!/usr/bin/env bash
set -e

# Clean up any lingering simulation processes
killall -9 gz sim ruby parameter_bridge ros2 python3 2>/dev/null || true

source /opt/ros/jazzy/setup.bash
source /mnt/c/Users/shres/Desktop/SIH2026/ros2_ws/install/setup.bash
export PYTHONPATH=/mnt/c/Users/shres/Desktop/SIH2026:$PYTHONPATH

ros2 launch /mnt/c/Users/shres/Desktop/SIH2026/simulation/launch/interactive_judge_demo.launch.py "$@"
