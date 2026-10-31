#!/usr/bin/env bash
set -e

# Cleanup function to kill any dangling simulation, GUI, and bridge processes
cleanup() {
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "gz-sim" 2>/dev/null || true
    pkill -9 -f "rviz2" 2>/dev/null || true
    pkill -9 -f "parameter_bridge" 2>/dev/null || true
    pkill -9 -f "interactive_ugv_navigation" 2>/dev/null || true
    pkill -9 -f "presentation_obstacle_controller" 2>/dev/null || true
    pkill -9 -f "teleop_obstacle" 2>/dev/null || true
    killall -9 gz-sim-server gz-sim-gui rviz2 parameter_bridge ruby 2>/dev/null || true
}

# Trap signals so stopping this script cleans up all background child processes
trap cleanup EXIT INT TERM

# Run cleanup before launch to guarantee fresh start
cleanup
sleep 0.5

# Ensure GUI display environment for WSLg / Wayland / X11
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

# Source ROS 2 environment
source /opt/ros/jazzy/setup.bash
if [ -f "/mnt/c/Users/shres/Desktop/SIH2026/ros2_ws/install/setup.bash" ]; then
    source /mnt/c/Users/shres/Desktop/SIH2026/ros2_ws/install/setup.bash
fi
export PYTHONPATH=/mnt/c/Users/shres/Desktop/SIH2026:$PYTHONPATH
export GZ_SIM_RESOURCE_PATH=/mnt/c/Users/shres/Desktop/SIH2026/simulation/worlds:/mnt/c/Users/shres/Desktop/SIH2026/simulation/models:/mnt/c/Users/shres/Desktop/SIH2026/simulation/models/outdoor_assets:$GZ_SIM_RESOURCE_PATH

echo "=============================================================="
echo "  LAUNCHING PHASE 14.7 PROTOTYPE PRESENTATION SIMULATION"
echo "  World: 100m Outdoor Presentation Course"
echo "  RViz:  Hero Presentation Profile"
echo "  DISPLAY: $DISPLAY | WAYLAND: $WAYLAND_DISPLAY"
echo "=============================================================="

ros2 launch /mnt/c/Users/shres/Desktop/SIH2026/simulation/launch/prototype_presentation.launch.py "$@"
