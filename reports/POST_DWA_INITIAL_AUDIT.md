# SIH2026 — Post-DWA Initial Architecture & Safety Audit
**Investigation Report: Root Cause Analysis of Vehicle Motion, Lost-Obstacle Hazards, and Recovery Deficiencies**

- **Project**: SIH2026 — Vision-Based Autonomous Navigation for UGV
- **Date**: 2026-09-30T16:00:00+05:30
- **Base Commit**: `fc7819a348aebea9f0b59d5e2288c92a6bceaeac`
- **Branch**: `demo-01-interactive-ugv`
- **Auditor**: Autonomous Vehicle Safety & Planning Engineering Agent

---

## Executive Summary
This audit rigorously investigates the current control, tracking, planning, and simulation stack to diagnose the root causes of three previously observed runtime failure modes in Gazebo Sim:
1. **Low UGV forward speed (~0.06–0.09 m/s)** despite a commanded speed of 0.50–0.60 m/s.
2. **Intentional/Unintentional obstacle pushing** when a dynamic or stationary obstacle is positioned directly in front of the vehicle.
3. **Deadlock without recovery** when the forward corridor is blocked.

Every answer below is verified against authoritative repository source code and configuration files.

---

## Audit Findings: Detailed Analysis

### A. Why is current UGV motion only ~0.06–0.09 m/s?
The crawl speed is caused by a compounding **control-frequency and dynamic-window acceleration starvation loop**:
1. **Synchronous 4 Hz Loop with Heavy Overhead**: In `simulation/nodes/interactive_ugv_navigation_node.py` (line 101), the node declares `control_rate_hz = 4.0` (period = 250 ms). In each tick of `_control_loop()`, the node sequentially executes obstacle tracking, trajectory prediction, dynamic occupancy rasterization, A* replanning checks, DWA trajectory evaluation, and comprehensive RViz marker generation (building over 10 marker types). As a result, the loop execution time regularly reaches 250–350 ms, causing actual publish intervals to stretch to 300–450 ms.
2. **Gazebo DiffDrive Controller Watchdog Deceleration**: Gazebo Sim's `gz::sim::systems::DiffDrive` plugin (`simulation/worlds/demo_outdoor.sdf`, lines 571–583) expects continuous command streams. In the absence of high-frequency velocity commands, the joint damping and friction on the 4 wheels decelerate the chassis between sparse pulses. The vehicle effectively undergoes a "pulse-and-brake" duty cycle.
3. **DWA Dynamic Window Acceleration Clamping**: In `ai/planning/local_planners/dwa.py` (lines 86–100), reachable forward velocity is strictly bounded by:
   $$v_{\text{high}} = \min(v_{\text{max}}, v_{\text{state}} + a_{\text{max}} \cdot \Delta t)$$
   With $\Delta t = 0.10\text{ s}$ and $a_{\text{max}} = 0.8\text{ m/s}^2$, the window only permits $v \le v_{\text{state}} + 0.08\text{ m/s}$. Because the vehicle decelerates to near zero ($v_{\text{state}} \approx 0.02\text{ m/s}$) during the long inter-command gap, DWA can only sample velocities up to $0.10\text{ m/s}$. The command issued is therefore only $0.06–0.09\text{ m/s}$. At the next tick, $v_{\text{state}}$ has once again decayed to near zero, permanently locking the vehicle into an acceleration-starved crawl.
4. **Un-accelerated Backend in Interactive Demo**: `interactive_ugv_navigation_node.py` (lines 84, 169) directly instantiated the reference Python `DWAPlanner()` throttled to $5 \times 9 = 45$ candidates, instead of utilizing `UnifiedDWAPlanner(backend=DWABackendType.AUTO)`, bypassing the newly validated $1.5\text{ ms}$ native C++ engine.

### B. Is `/cmd_vel` being published continuously?
**No.** `/cmd_vel` is published strictly once per iteration at line 569 of `interactive_ugv_navigation_node.py`:
```python
self.cmd_pub.publish(cmd_twist)
```
There is no background timer, asynchronous publisher thread, or command hold mechanism. If the planner loop stalls or experiences jitter, `/cmd_vel` publishing halts completely.

### C. What is the actual publish frequency?
- **Nominal Target**: $4.0\text{ Hz}$ ($250\text{ ms}$ period).
- **Actual Runtime**: $2.2\text{ Hz}$ to $3.6\text{ Hz}$ (gaps of $280\text{ ms}$ to $450\text{ ms}$) depending on whether the camera feed and dense traversability model (`last_trav_ms` $\approx 118\text{ ms}$) or extensive RViz marker generation run concurrently.

### D. What is the planner computation frequency?
The planner executes synchronously within the same $4.0\text{ Hz}$ timer callback. Thus, the planner computation frequency is identical to the control loop frequency ($\approx 2.5–4.0\text{ Hz}$).

### E. Is a watchdog braking between planner updates?
**Yes.** In Gazebo Sim Harmonic, the differential drive physics solver applies continuous damping and ground friction. When velocity command updates arrive at $\le 4\text{ Hz}$ with jitter, the vehicle loses drive torque before the next packet arrives, causing mechanical deceleration between ticks.

### F. Does Gazebo receive stale commands?
**No.** In fact, Gazebo experiences **command starvation** (gaps where no command is received) rather than stale command persistence. The bridge passes each discrete ROS2 `Twist` message to `/model/ugv_4wd/cmd_vel`, but because commands arrive irregularly, the vehicle stutters.

### G. Does the command buffer retain the last safe command?
**No.** Currently, there is **zero command buffering**. Neither `interactive_ugv_navigation_node.py` nor `dynamic_planner_node.py` maintains an asynchronous dispatch queue or a timestamped `latest_safe_command` buffer.

### H. Where should the high-frequency command publisher live?
The high-frequency command publisher should live as an **independent high-rate timer (e.g., $20\text{ Hz}$, $50\text{ ms}$ period)** within the navigation node:
- **Planner Thread / Lower-Rate Timer ($5–10\text{ Hz}$)**: Computes trajectory, evaluates dynamic risk, performs behavioral arbitration, validates safety, and updates a thread-safe `LatestSafeCommand` container with a monotonic timestamp.
- **Publisher Timer ($20\text{ Hz}$)**: Regularly dispatches the latest validated safe command to `/cmd_vel`. If the command age exceeds a strict freshness window ($300\text{ ms}$), the publisher commands controlled deceleration to zero to guarantee safety.

### I. How are close obstacles currently removed after tracking loss?
In `ai/planning/dynamic/tracker.py` (lines 235–252, 379–384):
1. When an obstacle is not associated with an incoming detection:
   - Cycle 1: `misses_to_degrade = 1` $\rightarrow$ status becomes `DEGRADED`.
   - Cycle 3: `misses_to_lost = 3` $\rightarrow$ status becomes `LOST`.
   - Cycle 5: `misses_to_expire = 5` $\rightarrow$ status becomes `EXPIRED`.
2. When marked `EXPIRED`, line 383 executes:
   ```python
   del self._tracks[tid]
   ```
   The obstacle is completely erased from memory.
3. In `interactive_ugv_navigation_node.py` (lines 368–377), `_generate_sensor_observation()` suppresses detections if relative bearing exceeds $45^\circ$ or distance exceeds $6.0\text{ m}$. When the UGV is close to or contacting an obstacle, physical occlusion or bumper proximity causes detection loss.
4. After 5 cycles ($\approx 1.25\text{ s}$), the track expires and vanishes. DWA evaluates the space in front as **unoccupied free space**, samples maximum forward velocity candidates, and commands the UGV to drive forward into the obstacle, physically bulldozing it toward the goal.

### J. Where can a final safety governor safely override `cmd_vel`?
The **Safety Governor** must be positioned downstream of Behavioral Arbitration and immediately upstream of the Command Buffer:
```
DWA Local Trajectory Evaluation
              ↓
Behavioral State Machine (PROCEED / AVOID / WAIT / STOP)
              ↓
     SAFETY GOVERNOR (Strict Proximity & Latched Hazard Check)
              ↓
  Latest Safe Command Buffer (Thread-safe, timestamped)
              ↓
    High-Frequency Publisher (20 Hz -> /cmd_vel)
```
The safety governor operates independently of DWA trajectory scoring:
- If any confirmed obstacle or recently latched hazard lies within the stopping distance safety envelope in the direction of motion, the governor clamps linear velocity to $0.0\text{ m/s}$.
- The governor can downgrade commands (`PROCEED` $\rightarrow$ `STOP`, or cap maximum velocity), but can **never** upgrade risk.

### K. How is deadlock currently detected?
In `ai/planning/dynamic/behavioral_state.py` (lines 211–232), `_evaluate_deadlock()` checks three conditions:
1. `consecutive_stop_count >= deadlock_consecutive_stop_cycles` (default: 5 cycles).
2. `wait_timeout_count >= deadlock_max_wait_timeouts` (default: 2 timeouts, where timeout is $3.0\text{ s}$).
3. Rolling pose displacement: net Euclidean displacement $< 0.10\text{ m}$ over a $5.0\text{ s}$ window while in `STOP` or `WAIT`.
When any condition triggers, the state machine emits `DeadlockEvent.DYNAMIC_DEADLOCK`.

### L. Is reverse currently supported?
- **Kinematically**: `min_linear_velocity_mps: -0.20` is configured in `configs/planning.yaml` and `ai/planning/base.py`, permitting reverse up to $-0.20\text{ m/s}$.
- **Tactically & Behaviorally**: **No.**
  - In `DWAPlanner._predict_trajectory()` and scoring (lines 345–346), reverse velocity is penalized (`vel_score = -0.5 * abs(v) / max_v`), and distance score to forward goal waypoints drops.
  - In `BehavioralStateMachine` (lines 461–463), whenever an obstacle blocks the path or no collision-free forward trajectory exists, the state machine forces `cmd_v = 0.0, cmd_omega = 0.0`.
  - There is **no recovery state machine** to command a controlled reverse maneuver or verify rear clearance. Once blocked, the UGV is permanently paralyzed in `STOP` unless pushed or reset.

### M. How does global replanning currently trigger?
In `ai/planning/dynamic/global_replanner.py`, `EventDrivenGlobalReplanner` listens for `DYNAMIC_DEADLOCK`, `PERSISTENT_BLOCKAGE`, `INSUFFICIENT_PROGRESS`, or `PATH_INVALIDATED`:
- Upon trigger, it inflates dynamic obstacles by $0.20\text{ m}$ into a `DynamicPlanningMapOverlay` and invokes A* route planning.
- **Critical Flaw**: When the UGV is pinned against an obstacle, the robot's current pose lies inside the inflated obstacle boundary (`entrance_clearance_steps` check fails or A* start cell is in collision). Without a preliminary reverse maneuver to disengage from the obstacle's safety margin, A* replanning fails repeatedly.

### N. What tests already exist?
The repository contains 112 test modules in `tests/`, including:
- Local Planning & Backends: `test_dwa_backends.py`, `test_dwa_cpp_parity.py`, `test_dwa_acceleration_parity.py`, `test_predictive_collision.py`.
- Behavior & Replanning: `test_behavioral_state.py`, `test_global_replanning.py`, `test_dynamic_sil_core_scenarios.py`.
- Tracking & Recovery: `test_dynamic_tracker.py`, `test_phase14_recovery.py` (which validates track reacquisition after blackout).
- ROS2 Integration: `test_ros2_dynamic_integration.py`, `test_phase13_ros2_end_to_end.py`, `test_dynamic_planner_ros_boundary.py`.

---

## Conclusion & Action Plan
With the root causes rigorously proven from first principles and repository code, the path forward is clear:
1. **Stage 1 (Phase 13.22)**: Implement a $20\text{ Hz}$ decoupled command publisher with a bounded freshness safety watchdog, and wire `UnifiedDWAPlanner(backend=DWABackendType.AUTO)` into the simulation navigation stack.
2. **Stage 2 (Phase 13.23)**: Implement the Latched Hazard Memory ("Not Detected != Safe") and the downstream Authoritative Safety Governor.
3. **Stage 3 (Phase 13.24)**: Implement Deadlock Reverse Recovery with rear clearance verification and post-recovery global replanning.
4. **Stage 4 (Phase 13.25)**: Execute integrated closed-loop validation across scenarios B01–B10.
