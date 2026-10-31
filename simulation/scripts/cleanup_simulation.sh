#!/usr/bin/env bash
# Thoroughly kill all simulation, GUI, bridge, and navigation processes
echo "Cleaning up lingering simulation processes..."
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "gz-sim" 2>/dev/null || true
pkill -9 -f "rviz2" 2>/dev/null || true
pkill -9 -f "parameter_bridge" 2>/dev/null || true
pkill -9 -f "interactive_ugv_navigation" 2>/dev/null || true
pkill -9 -f "presentation_obstacle_controller" 2>/dev/null || true
pkill -9 -f "teleop_obstacle" 2>/dev/null || true
killall -9 gz-sim-server gz-sim-gui rviz2 parameter_bridge ruby 2>/dev/null || true
sleep 0.5
echo "Simulation environment cleaned."
