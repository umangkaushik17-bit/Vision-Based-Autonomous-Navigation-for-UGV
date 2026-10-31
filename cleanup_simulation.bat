@echo off
title Cleanup Simulation Processes
echo Cleaning up all Gazebo Sim, RViz2, and ROS2 simulation processes in WSL...
wsl.exe -d Ubuntu bash -c "bash /mnt/c/Users/shres/Desktop/SIH2026/simulation/scripts/cleanup_simulation.sh"
echo Cleanup complete.
timeout /t 3
