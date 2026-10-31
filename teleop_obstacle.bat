@echo off
title Dynamic Obstacle Keyboard Teleop (WASD / SPACE / R)
echo ==============================================================
echo   DYNAMIC OBSTACLE KEYBOARD CONTROLLER
echo   Controls: W/S (+X/-X), A/D (+Y/-Y into/out of path)
echo             SPACE (Stop), R (Reset), Q (Quit)
echo ==============================================================
wsl.exe -d Ubuntu bash -l /mnt/c/Users/shres/Desktop/SIH2026/simulation/scripts/run_teleop.sh
pause
