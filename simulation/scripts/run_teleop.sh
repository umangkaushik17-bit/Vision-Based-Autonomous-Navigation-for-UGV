#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source /mnt/c/Users/shres/Desktop/SIH2026/ros2_ws/install/setup.bash
export PYTHONPATH=/mnt/c/Users/shres/Desktop/SIH2026:$PYTHONPATH

python3 /mnt/c/Users/shres/Desktop/SIH2026/simulation/nodes/teleop_obstacle_keyboard.py "$@"
