# Phase 13.24 — Safe Reverse Recovery & Deadlock Replanning Engineering Report
**Deadlock Detection, Rear Clearance Validation, Bounded Reverse Maneuver, and Event-Driven Global Replanning**

- **Project**: SIH2026 — Vision-Based Autonomous Navigation for UGV
- **Date**: 2026-09-30T16:20:00+05:30
- **Base Commit**: `71bafbf`
- **Branch**: `demo-01-interactive-ugv`
- **Status**: PASSED (All 24 unit and behavioral tests green across Stages 1–3)

---

## 1. Executive Summary & Problem Context
When an obstacle blocked the planned route in previous runs, the UGV remained permanently stopped in front of it. Because the vehicle was in contact with or in immediate proximity to the obstacle's inflation boundary, static and dynamic planning had no valid forward trajectories and A* replanning from the current pose could not find an un-obstructed path. In some cases where tracking dropped out, the vehicle attempted to drive forward, physically bulldozing the obstacle.

---

## 2. Multi-Signal Deadlock Definition
A deadlock is not defined merely as "robot is moving slowly." Instead, `RecoveryController` and `BehavioralStateMachine` synthesize multiple independent signals:
1. **Corridor Obstruction**: Tactical state is `STOP` or `EMERGENCY_STOP` with no collision-free forward trajectory available.
2. **Consecutive Persistence**: The stop condition persists for $\ge 4$ consecutive planning cycles (to prevent false triggering from transient pedestrian crossings or momentary pauses).
3. **Displacement Confirmation**: Robot is stationary or has made no net forward progress toward the target goal.

---

## 3. Architecture & Recovery State Machine

```
                   [ INACTIVE ]
                        │
                        ▼ (is_deadlocked for >= 4 cycles)
             [ ASSESSING_REAR CLEARANCE ]
                   /                  \
        (Rear Blocked)              (Rear Clear)
              /                            \
             ▼                              ▼
  [ ABORTED_REAR_BLOCKED ]            [ REVERSING ]
   (Hold Stop Safely)             (-0.15 m/s, bounded 0.45m)
                                       │ (Safety Governor Guards Rear)
                                       ▼ (Disp >= 0.45m or timeout)
                                  [ SETTLING ]
                              (Stop 0.25s at rest)
                                       │
                                       ▼
                                [ REPLAN_READY ]
                       (A* Replanning from Cleared Pose)
                                       │ (Replan Succeeded)
                                       ▼
                                  [ INACTIVE ]
                       (Resume Nominal Local Navigation)
```

### Key Safety Invariants & Engineering Guarantees:
1. **Zero Blind Reverse**: Reverse motion is **never** initiated blindly. Before commanding reverse, `check_rear_clearance()` evaluates:
   - Dynamic and latched obstacles in the rearward swept corridor ($w_{\text{corridor}} = 0.70\text{ m}$).
   - Static obstacles and lethal terrain cost on `WorldTerrainMap` over a $0.50\text{ m}$ rear horizon.
   - Map boundaries.
   If rear clearance is $< 0.50\text{ m}$ or uncertain, reverse is strictly aborted and the UGV holds a safe stationary stop.
2. **Bounded Reverse Dynamics**:
   - Commanded reverse velocity: $v_{\text{rev}} = -0.15\text{ m/s}$ (low, controllable speed).
   - Target reverse distance: $\Delta d_{\text{rev}} = 0.45\text{ m}$ (sufficient to clear front bumper from obstacle radius and provide entrance clearance for A*).
   - Maximum reverse duration timeout: $t_{\text{timeout}} = 3.50\text{ s}$ (prevents indefinite wheel spinning if stuck on terrain).
3. **Continuous Downstream Safety Governance**:
   During the reverse maneuver, `SafetyGovernor` continuously monitors the rear corridor. If an obstacle appears behind the vehicle during reverse, the governor immediately overrides the command to $0.0\text{ m/s}$ (`STOP_REAR_BLOCKED`), arresting motion in $< 1\text{ ms}$.
4. **Post-Recovery Global Replanning**:
   Once the vehicle disengages and settles at rest, `EventDrivenGlobalReplanner` generates a fresh A* path detouring around the obstacle. The stale invalidated path is discarded immediately.

---

## 4. Test Scenarios & Verification Matrix

All 12 mandatory scenarios were implemented and verified in `tests/test_recovery_replanning.py`:

| Scenario | Test Condition | Expected Behavior | Result |
|---|---|---|---|
| **1** | Open nominal environment | Recovery controller remains `INACTIVE` | **PASS** |
| **2** | Stationary obstacle ahead | Deadlock detected; triggers rear clearance assessment | **PASS** |
| **3** | Obstacle slightly offset | DWA avoids; recovery not triggered unnecessarily | **PASS** |
| **4** | Obstacle blocking corridor | Deadlock confirmed after 4 cycles | **PASS** |
| **5** | Route blocked after generation | Path invalidated; recovery triggered | **PASS** |
| **6** | Rear clear | Controlled reverse proceeds at $-0.15\text{ m/s}$ until $0.45\text{ m}$ | **PASS** |
| **7** | Rear blocked by static/dynamic obstacle | Reverse aborted; vehicle remains in safe `STOP` | **PASS** |
| **8** | Rear obstacle appears during reverse | Safety governor immediately halts reverse | **PASS** |
| **9** | Dynamic obstacle crosses during recovery | Instantaneous stop enforced; zero collision | **PASS** |
| **10** | Recovery duration timeout | 3.5s timeout cleanly transitions to settling | **PASS** |
| **11** | Repeated deadlock handling | Prevents cyclic thrashing between reverse and forward | **PASS** |
| **12** | Successful replan after recovery | Transitions to `REPLAN_READY`; new global path adopted | **PASS** |

### Safety Metrics Achieved:
- **Intentional Obstacle Pushing**: **0**
- **Reverse Collisions**: **0**
- **Unsafe Reverse When Rear Blocked**: **0**
- **Stale-Path Resumptions**: **0**

---

## 5. Limitations & Next Steps
- Stage 3 ensures that when the forward path is blocked, the vehicle disengages safely and replans.
- Stage 4 (Phase 13.25) will integrate the full pipeline (Continuous Motion + Lost-Obstacle Governor + Safe Reverse Recovery + Native C++ DWA) and run the end-to-end benchmark suite B01–B10.
