# SIH2026 — Post-DWA Vehicle Safety & Recovery Workstream Master Plan
**Stages 1–4: Continuous Motion → Lost-Obstacle Safety → Recovery/Replanning → Integrated Validation**

- **Project**: SIH2026 — Vision-Based Autonomous Navigation for UGV
- **Branch**: `demo-01-interactive-ugv`
- **Initial Commit**: `fc7819a348aebea9f0b59d5e2288c92a6bceaeac`
- **Date & Timestamp**: 2026-09-30T16:00:00+05:30
- **Status**: ACTIVE EXECUTION PLAN (Persistent Master Checkpoint)

---

## 1. Primary Objective
Having verified and benchmarked the Native C++ DWA local planning backend in Phase 13.21, the objective is to resolve real-world closed-loop vehicle behavior in Gazebo Sim:
1. **Restore Continuous Vehicle Motion**: Eliminate pulsed/jerky actuation and low crawl speed (0.06–0.09 m/s) by separating planning rate from high-frequency command publication.
2. **Enforce Lost-Obstacle Safety ("Not Detected != Safe")**: Prevent recently observed close-range hazards from turning into free space upon temporary perception/tracking loss.
3. **Add Controlled Reverse Recovery, Deadlock Detection & Replanning**: Prevent the UGV from getting pinned against obstacles or attempting to push them forward; reverse safely with rear-clearance gating and trigger global replanning.
4. **Validate Integrated Autonomous Navigation**: Validate end-to-end A -> B navigation in Gazebo with static and dynamic obstacles while preserving the multi-backend DWA architecture.

---

## 2. Current Repository State & Context
- **DWA Status**: Fully optimized and verified in Phase 13.21. Five backends are architecturally preserved: Python Reference, Vectorized CPU, Native C++ CPU (AVX2/OpenMP), NVIDIA CUDA, and Intel GPU/XPU. Native C++ achieves 1.1–1.5 ms per 375 candidates. DWA algorithm and mathematics are FROZEN.
- **Perception Status**: `LightweightTraversabilityNet` and dense traversability models are FROZEN.
- **Actuation & Simulation**: Gazebo Sim Harmonic running diff-drive skid-steer model (`gz-sim-diff-drive-system`) connected via `ros_gz_bridge`.

---

## 3. Known Historical Failure Modes
1. **Slow Crawl & Pulsed Actuation**: In `interactive_ugv_navigation_node.py`, a single synchronous 4 Hz timer runs tracking, prediction, DWA, and RViz marker generation. The resulting 250–400 ms gaps between `/cmd_vel` messages cause the Gazebo diff-drive plugin to brake between pulses. Odometry velocity drops to near-zero, constraining DWA's dynamic window acceleration bounds and locking the UGV at 0.06–0.09 m/s.
2. **Obstacle Pushing & Lost-Obstacle Acceleration**: When an obstacle is directly in front of the UGV, if the obstacle leaves the camera's 45° FOV or detection fails for 5 cycles, the tracker marks it `EXPIRED` and prunes it. The occupied zone becomes free space, DWA generates full forward throttle, and the UGV bulldozes the obstacle.
3. **Deadlock Without Recovery**: When blocked by an obstacle, `BehavioralStateMachine` commands `STOP` (`v=0, w=0`). No reverse maneuver is executed because DWA penalizes reverse and behavioral arbitration lacks an active recovery state.

---

## 4. Stage-by-Stage Execution Plan

### Stage 1: Phase 13.22 — Continuous Command Delivery & Vehicle Motion
- **Goal**: Decouple planner frequency from command publication rate.
- **Architecture**:
  - High-frequency command publisher running at 20 Hz (50 ms period).
  - Thread-safe, timestamped `latest_safe_command` buffer.
  - Command freshness watchdog: if command age exceeds safety window (e.g. 300 ms), safely decelerate to zero.
  - Plug `UnifiedDWAPlanner(backend=DWABackendType.AUTO)` into `interactive_ugv_navigation_node.py` to leverage the 1.5 ms C++ engine.
- **Safety Invariant**: The publisher must only dispatch validated commands; stale commands (> 300 ms) automatically trigger safe deceleration.

### Stage 2: Phase 13.23 — Lost-Obstacle Safety Governor
- **Goal**: Prevent close-range obstacles from disappearing into free space upon detection loss.
- **Architecture**:
  - **Recent-Hazard Memory / Latch**: Any confirmed obstacle within close proximity (< 2.0 m) that is lost by the detector/tracker is latched into a conservative safety envelope with a physics-justified Time-To-Live (TTL).
  - **Independent Safety Governor**: Located downstream of behavioral arbitration, immediately before `/cmd_vel` buffering. Evaluates instantaneous forward clearance against latched hazards and clamps linear/angular velocity to zero if forward clearance is compromised.
- **Safety Invariant**: Strict 0 collision false negatives and 0 unsafe `STOP` -> `PROCEED` transitions caused by tracking dropouts.

### Stage 3: Phase 13.24 — Safe Reverse Recovery & Deadlock Replanning
- **Goal**: Controlled reverse escape and route regeneration when forward progress is blocked.
- **Architecture**:
  - **Deadlock Recognition**: Multi-signal detection (consecutive stop cycles, zero progress over rolling window, forward corridor blocked).
  - **Rear Clearance Verification**: Rigorous rear safety envelope check (static costmap and rear sensor bounds). If rear clearance is uncertain or blocked, UGV remains stopped.
  - **Controlled Reverse Maneuver**: Bounded negative velocity (e.g. -0.15 m/s) for a bounded distance/duration (e.g. 0.4–0.6 m).
  - **Post-Reverse Global Replan**: Invalidate blocked path, update costmap overlay with obstacle inflation, execute A* replan, and resume nominal navigation.
- **Safety Invariant**: Zero blind reverse. Active rear-clearance validation required prior to and during reverse. Zero obstacle pushing.

### Stage 4: Phase 13.25 — Integrated Dynamic-Obstacle Navigation Validation
- **Goal**: End-to-end validation of complete chain in Gazebo Sim and SITL test harness.
- **Evaluation Matrix (B01–B10)**:
  - B01: Open nominal path.
  - B02: Static obstacle avoidance.
  - B03: Dynamic crossing obstacle.
  - B04: Temporary perception dropout (0.5s).
  - B05: Close obstacle tracker loss.
  - B06: Narrow passage traversal.
  - B07: Complete forward corridor blockage.
  - B08: Deadlock detection and safe reverse recovery.
  - B09: Rear blocked recovery rejection (safe hold).
  - B10: Dynamic obstacle crossing during recovery.
- **Metrics Collected**: Collision count, command gap statistics, vehicle velocity profile, replan events, DWA/cycle latencies.

---

## 5. Absolute Safety Invariants
1. **Preserve All Backends**: Python Reference, Vectorized CPU, Native C++, CUDA, Intel GPU must remain fully operational.
2. **DWA Frozen**: No algorithmic or mathematical alterations to DWA.
3. **Perception Models Frozen**: No retraining or semantic alterations to `LightweightTraversabilityNet`.
4. **Safety Prioritized Over Velocity**: Never suppress safety checks to achieve higher speeds.
5. **No Blind Reverse**: Every reverse maneuver requires verified rear clearance.
6. **"Not Detected != Safe"**: Latched hazard persistence for close-range obstacles.
7. **No Unsafe Stale Command Persistence**: Hard timeout (< 350 ms) on unrefreshed commands.

---

## 6. Test & Checkpoint Strategy
- After each stage (13.22, 13.23, 13.24, 13.25):
  1. Execute unit and regression tests in Python.
  2. Verify safety metrics and hardware invariants.
  3. Update `reports/POST_DWA_VEHICLE_SAFETY_WORKSTREAM_STATE.md`.
  4. Write stage report in `reports/`.
  5. Commit with semantic git commit message.
- If any safety invariant fails, stop immediately.

---

## 7. Tool & MCP Usage Notes
- Use `sequentialthinking` MCP tool for dynamic problem analysis, state transition logic, and risk assessment.
- Use `run_command` for test executions and Git operations.
- Avoid modifying code outside the specified control, safety, and recovery modules.
