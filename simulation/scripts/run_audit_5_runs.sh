#!/usr/bin/env bash
set -e

killall -9 gz sim ruby parameter_bridge ros2 python3 2>/dev/null || true

source /opt/ros/jazzy/setup.bash
source /mnt/c/Users/shres/Desktop/SIH2026/ros2_ws/install/setup.bash
export PYTHONPATH=/mnt/c/Users/shres/Desktop/SIH2026:$PYTHONPATH

python3 /mnt/c/Users/shres/Desktop/SIH2026/simulation/scripts/run_5_demo_runs.py
