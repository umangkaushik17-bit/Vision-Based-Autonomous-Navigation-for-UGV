@echo off
title Phase 14.7 UGV Simulation (Interactive WASD Dynamic Obstacle)
echo ==============================================================
echo   PHASE 14.7: INTERACTIVE UGV SIMULATION (WASD TELEOP MODE)
echo   Starting Dynamic Obstacle WASD Keyboard Window...
echo   Starting Gazebo Sim, Hero RViz2, and Autonomous Navigation...
echo ==============================================================
timeout /t 1 >nul
start "Dynamic Obstacle WASD Controller" cmd.exe /c "teleop_obstacle.bat"
wsl.exe -d Ubuntu bash -l /mnt/c/Users/shres/Desktop/SIH2026/simulation/scripts/launch_presentation.sh enable_dynamic_obstacle:=false
pause
