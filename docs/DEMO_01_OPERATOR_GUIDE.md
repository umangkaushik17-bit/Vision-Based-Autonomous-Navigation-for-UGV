# DEMO-01 OPERATOR QUICKSTART & JUDGE WALKTHROUGH

This guide provides the exact 10-step instructions for the operator during the local, interactive judge demonstration of the autonomous UGV navigation system.

---

## Prerequisites
- **OS**: Windows 11 with WSL2 (Ubuntu 24.04 LTS).
- **Display**: WSLg enabled (default on Windows 11).
- **Environment**: ROS 2 Jazzy & Gazebo Sim Harmonic (CPU-only, no GPU required).

---

## 10-Step Interactive Judge Demonstration

### Step 1: Open WSL Terminal
Open PowerShell or Windows Terminal and launch your Ubuntu shell:
```bash
wsl -d Ubuntu
```

### Step 2: Source ROS 2 and Workspace
Navigate to the project root and source environments:
```bash
cd /mnt/c/Users/shres/Desktop/SIH2026
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
export PYTHONPATH=/mnt/c/Users/shres/Desktop/SIH2026:$PYTHONPATH
```

### Step 3: Launch the Unified Simulation and Navigation Demo
Run the one-click launcher script in Terminal 1:
```bash
bash simulation/scripts/launch_demo.sh
```
*Gazebo Sim Harmonic and RViz2 will open side-by-side with full 3D models, paths, and real-time HUD telemetry.*

### Step 4: Open Dynamic Obstacle Teleop Terminal
In a second terminal tab (Terminal 2), launch the keyboard teleoperation controller:
```bash
wsl -d Ubuntu
cd /mnt/c/Users/shres/Desktop/SIH2026
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
export PYTHONPATH=/mnt/c/Users/shres/Desktop/SIH2026:$PYTHONPATH

python3 simulation/nodes/teleop_obstacle_keyboard.py
```

### Step 5: Observe UGV Autonomous Navigation & Static Rock Avoidance
- Watch the UGV autonomously drive forward from $(0.0, 0.0)$ along the green global A* path.
- As the UGV approaches the large stationary boulder at $(3.5, 0.15)$, point out to the judges that the UGV computes a smooth detour trajectory (cyan DWA trajectory) keeping $> 1.0$m safe clearance without stopping or colliding.

### Step 6: Steer Dynamic Obstacle into the Path Corridor
- In Terminal 2, press and hold the **`A`** key to drive the orange dynamic obstacle from standby $(2.5, -3.0)$ into the active navigation corridor ($Y \approx -1.0$ to $0.0$).
- Press **`SPACE`** to halt the obstacle directly in front of the advancing UGV.

### Step 7: Highlight Real-Time Perception, Prediction & Tactical Decision
Direct the judges' attention to the RViz2 display:
1. **Dynamic Tracking**: An orange 3D bounding box encloses the moving obstacle with velocity vector arrows and Track ID.
2. **Trajectory Prediction**: Amber forward trajectory with spheres marks future predicted positions at **+0.5s, +1.0s, +1.5s, and +2.0s**.
3. **Dynamic Occupancy**: Purple spatial-temporal slices show reserved dynamic risk zones.
4. **DWA Candidate Trajectories**: Candidate paths color-coded in real-time (Green = Safe, Yellow = Risky, Red = Collision).
5. **HUD State Transition**: The top HUD banner immediately switches from **`PROCEED`** (Green) to **`AVOID`** / **`WAIT`** (Yellow/Red), adjusting speed and steering away from predicted impact.

### Step 8: Steer Dynamic Obstacle Away
- In Terminal 2, press **`D`** to drive the obstacle out of the corridor back into neutral space ($Y < -2.0$).
- Press **`SPACE`** to stop.

### Step 9: Observe Tactical Recovery and Goal Arrival
- Observe the HUD status automatically recover to **`PROCEED`**.
- The UGV resumes its forward trajectory and reaches the emerald **GOAL BEACON** at $(7.0, 0.0)$, coming to a controlled stop with `MISSION COMPLETED`.

### Step 10: Reset Demonstration for Repeatability
- In Terminal 2, press **`R`**.
- The UGV and dynamic obstacle teleport back to their initial starting poses, odometry and trackers reset instantly, and the UGV is immediately ready for another run.

---

## Keyboard Controls Summary (Terminal 2)

| Key | Action | Description |
|---|---|---|
| **W** | Move Forward | Moves dynamic obstacle +X in world |
| **S** | Move Backward | Moves dynamic obstacle -X in world |
| **A** | Move Left | Steers obstacle into UGV path corridor (+Y) |
| **D** | Move Right | Steers obstacle away from UGV path (-Y) |
| **SPACE** | Stop | Halts dynamic obstacle immediately |
| **R** | Reset | Resets UGV, dynamic obstacle, and navigation stack |
| **Q / Ctrl+C** | Quit | Exits teleop node |
