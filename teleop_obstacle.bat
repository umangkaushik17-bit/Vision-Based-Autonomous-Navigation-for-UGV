@echo off
title Dynamic Obstacle Keyboard Teleop (WASD / SPACE / R)
echo ==============================================================
echo   DYNAMIC OBSTACLE KEYBOARD CONTROLLER
echo   Controls: W/S (+X/-X), A/D (+Y/-Y into/out of path)
echo             SPACE (Stop), R (Reset), Q (Quit)
echo ==============================================================
wsl.exe -d Ubuntu bash -c "source /opt/ros/jazzy/setup.bash; source /mnt/c/Users/shres/Desktop/SIH2026/ros2_ws/install/setup.bash; export PYTHONPATH=/mnt/c/Users/shres/Desktop/SIH2026:$PYTHONPATH; python3 /mnt/c/Users/shres/Desktop/SIH2026/simulation/nodes/teleop_obstacle_keyboard.py"
pause
