#!/usr/bin/env bash
set -e

# Cleanup function to kill all simulation, GUI, and bridge processes
cleanup() {
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "gz-sim" 2>/dev/null || true
    pkill -9 -f "rviz2" 2>/dev/null || true
    pkill -9 -f "parameter_bridge" 2>/dev/null || true
    pkill -9 -f "interactive_ugv_navigation" 2>/dev/null || true
    pkill -9 -f "teleop_obstacle" 2>/dev/null || true
    killall -9 gz-sim-server gz-sim-gui rviz2 parameter_bridge ruby 2>/dev/null || true
}

# Trap signals so stopping this script cleans up all background child processes
trap cleanup EXIT INT TERM

# Run cleanup before launch to guarantee fresh start
cleanup
sleep 0.5

# Ensure GUI display environment for WSLg / X11
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi
if [ -z "$WAYLAND_DISPLAY" ] && [ -e "/run/user/1000/wayland-0" ]; then
    export WAYLAND_DISPLAY=wayland-0
fi
if [ -d "/run/user/1000" ]; then
    export XDG_RUNTIME_DIR=/run/user/1000
elif [ -d "/mnt/wslg/runtime-dir" ]; then
    export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
fi
export QT_QPA_PLATFORM=xcb

source /opt/ros/jazzy/setup.bash
source /mnt/c/Users/shres/Desktop/SIH2026/ros2_ws/install/setup.bash
export PYTHONPATH=/mnt/c/Users/shres/Desktop/SIH2026:$PYTHONPATH

echo "=============================================================="
echo "  LAUNCHING DEMO-01 INTERACTIVE UGV SIMULATION & RVIZ2 HUD"
echo "  DISPLAY: $DISPLAY | QT_PLATFORM: $QT_QPA_PLATFORM"
echo "=============================================================="

ros2 launch /mnt/c/Users/shres/Desktop/SIH2026/simulation/launch/interactive_judge_demo.launch.py "$@"
