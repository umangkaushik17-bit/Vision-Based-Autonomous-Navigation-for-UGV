# DEMO-01 TROUBLESHOOTING & RECOVERY GUIDE

This document provides quick diagnosis and resolutions for any common issues encountered while setting up or presenting the DEMO-01 interactive UGV simulation.

---

## 1. Leftover Background Processes
**Symptoms**:
- Gazebo reports `Address already in use` or hangs at `Waiting for master`.
- ROS 2 Bridge reports `Transport initialization failure`.

**Immediate Fix**:
Run the clean shutdown command in your WSL terminal before launching:
```bash
killall -9 gz sim ruby parameter_bridge ros2 python3 2>/dev/null || true
```

---

## 2. WSLg Graphical Display Issues
**Symptoms**:
- RViz2 or Gazebo exits immediately with `cannot open display` or `WAYLAND_DISPLAY not set`.

**Diagnosis**:
Check your display variables in WSL:
```bash
echo $DISPLAY
echo $WAYLAND_DISPLAY
```

**Immediate Fix**:
If using WSLg on Windows 11:
```bash
export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
```
If using an external X-Server (VcXsrv / Xming):
```bash
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
export LIBGL_ALWAYS_INDIRECT=0
```

---

## 3. Keyboard Teleoperation Not Responding
**Symptoms**:
- Pressing `W`, `A`, `S`, `D` does not move the dynamic orange obstacle.

**Diagnosis**:
- Make sure the terminal running `teleop_obstacle_keyboard.py` has active window focus.
- The teleop captures raw terminal keypresses using standard `termios`. If running in an IDE integrated terminal that intercepts keys, run the teleop script directly in a native Windows Terminal or bash shell.

**Alternative Velocity Injection**:
If running without keyboard teleop, you can command the obstacle directly via ROS 2 topic:
```bash
# Drive obstacle into UGV corridor (+Y direction)
ros2 topic pub --once /dynamic_obstacle/cmd_vel geometry_msgs/msg/Twist "{linear: {y: 0.6}}"

# Stop obstacle
ros2 topic pub --once /dynamic_obstacle/cmd_vel geometry_msgs/msg/Twist "{}"
```

---

## 4. High CPU Latency on Older Laptops
**Symptoms**:
- DWA computation takes $> 1.0$s per step; UGV moves with stutter.

**Immediate Fix**:
1. Run Gazebo headless (`-s`) while keeping RViz2 open for full 3D visual fidelity:
   ```bash
   # Launch Gazebo headless server
   gz sim -s -r /mnt/c/Users/shres/Desktop/SIH2026/simulation/worlds/demo_outdoor.sdf &
   ```
2. The DWA candidate grid is already optimized to $5 \times 9 = 45$ candidate trajectories, taking only $\sim 450$ms on CPU.

---

## 5. Dynamic Obstacle Not Detected in RViz
**Symptoms**:
- Orange obstacle is in Gazebo, but no bounding box or prediction trajectory appears in RViz2.

**Diagnosis**:
- The simulated sensor adapter strictly adheres to real vehicle optics with a **$6.0\text{m}$ range** and **$\pm 45^\circ$ forward cone**.
- When the obstacle is parked in its initial standby position at $(2.5, -3.0)$, it sits at $-50.2^\circ$ relative to the UGV heading and is intentionally outside the forward sensor cone.

**Resolution**:
- Press **`A`** in the teleop terminal to drive the obstacle into the forward corridor ($Y \approx -1.2\text{m}$ to $0.0\text{m}$). Once it enters the sensor cone, the tracker will immediately lock on.

---

## 6. Manual Reset Fallback
**Symptoms**:
- Pressing `R` in teleop did not teleport the vehicle or reset odometry.

**Immediate Fix**:
Publish the reset trigger directly via ROS 2:
```bash
ros2 topic pub --once /demo/reset std_msgs/msg/Empty "{}"
```
Or reset Gazebo entity poses directly via the Gazebo transport service:
```bash
gz service -s /world/demo_outdoor/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 1000 --req 'name: "ugv_4wd", position: {x: 0.0, y: 0.0, z: 0.15}, orientation: {x: 0, y: 0, z: 0, w: 1}'
gz service -s /world/demo_outdoor/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 1000 --req 'name: "dynamic_obstacle", position: {x: 2.5, y: -3.0, z: 0.25}, orientation: {x: 0, y: 0, z: 0.7071, w: 0.7071}'
```
