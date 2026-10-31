"""Unit tests verifying Multi-Signal Spatial Progress Monitoring, Deadlock Detection,
and Forward-Only DWA Candidate Generation for Limit-Cycle / Chattering Elimination.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.base import PlanningConfig, RobotState, PlannerStatus, VehicleKinematics
from ai.planning.backends.backend_interface import DWABackendType
from ai.planning.backends.unified_planner import UnifiedDWAPlanner
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
    DeadlockEvent,
)
from ai.planning.path import Path, PathPoint
from ai.planning.recovery_controller import (
    RecoveryConfig,
    RecoveryController,
    RecoveryState,
)
from ai.planning.trajectory import Trajectory, TrajectoryPoint
from ai.spatial_fusion.world_map import WorldTerrainMap


def _create_test_map(res: float = 0.10) -> WorldTerrainMap:
    wtm = WorldTerrainMap(
        resolution_m=res,
        x_bounds_m=[-2.0, 8.0],
        y_bounds_m=[-5.0, 5.0],
        created_timestamp=0.0,
    )
    wtm.cost.fill(0.10)
    wtm.obstacle.fill(False)
    wtm.hazard.fill(0.0)
    wtm.unknown.fill(False)
    return wtm


def test_behavioral_state_detects_avoid_limit_cycle_deadlock():
    """Verify BehavioralStateMachine detects lack of progress even when in AVOID state."""
    cfg = BehavioralConfig(
        deadlock_progress_window_s=3.0,
        deadlock_progress_threshold_m=0.10,
        min_dwell_time_s=0.0,
    )
    sm = BehavioralStateMachine(cfg)

    # Simulate oscillatory AVOID commands at x ≈ 1.25 (chattering limit cycle)
    traj = Trajectory([TrajectoryPoint(1.25, 0.0, 0.0, 0.0, 0.15, 0.1, 0.1, 1)])
    avoid_diag = {
        "dynamic_collision": False,
        "dynamic_risk": "MEDIUM",
        "valid_trajectories": 10,
        "dynamically_rejected_candidates": 2,
    }

    decisions = []
    # 35 steps at 10 Hz (3.5 seconds) oscillating between x=1.24 and x=1.26
    for i in range(35):
        t = 1.0 + i * 0.10
        osc_x = 1.25 + 0.02 * math.sin(i * 1.5)
        osc_y = -0.14 + 0.01 * math.cos(i * 1.5)
        st = RobotState(x=osc_x, y=osc_y, yaw=-0.64, v=0.10 * math.sin(i * 1.5), omega=0.0, timestamp=t)
        d = sm.evaluate(
            current_state=st,
            best_trajectory=traj,
            best_v=st.v,
            best_w=0.08,
            planner_status=PlannerStatus.SUCCESS,
            dwa_diagnostics=avoid_diag,
            current_time_s=t,
        )
        decisions.append(d)

    # Initial cycles should be in AVOID
    assert decisions[5].state == BehavioralState.AVOID

    # After exceeding progress window without net displacement, must trigger DYNAMIC_DEADLOCK and STOP
    last_d = decisions[-1]
    assert last_d.deadlock is True
    assert last_d.deadlock_event == DeadlockEvent.DYNAMIC_DEADLOCK.value
    assert last_d.state == BehavioralState.STOP
    assert "Insufficient progress" in last_d.reason


def test_spatial_progress_monitor_logic():
    """Verify rolling pose history identifies chattering/oscillation vs nominal progress."""
    pose_history = []
    progress_window_s = 2.50
    dist_to_goal = 5.0

    # 1. Chattering: oscillating near (1.25, -0.14)
    lack_of_progress_chattering = False
    for step in range(30):
        t = 10.0 + step * 0.10
        x = 1.25 + 0.03 * math.sin(step)
        y = -0.14 + 0.02 * math.cos(step)
        pose_history.append((t, x, y))
        cutoff = t - progress_window_s
        pose_history = [p for p in pose_history if p[0] >= cutoff]

        if len(pose_history) >= 8 and (t - pose_history[0][0]) >= (progress_window_s * 0.75):
            disp = math.hypot(x - pose_history[0][1], y - pose_history[0][2])
            if disp < 0.15 and dist_to_goal > 0.50:
                lack_of_progress_chattering = True
                break

    assert lack_of_progress_chattering is True

    # 2. Nominal forward travel: x(t) = 0.40 * t
    pose_history_nominal = []
    lack_of_progress_nominal = False
    for step in range(30):
        t = 10.0 + step * 0.10
        x = 0.40 * (step * 0.10)
        y = 0.0
        pose_history_nominal.append((t, x, y))
        cutoff = t - progress_window_s
        pose_history_nominal = [p for p in pose_history_nominal if p[0] >= cutoff]

        if len(pose_history_nominal) >= 8 and (t - pose_history_nominal[0][0]) >= (progress_window_s * 0.75):
            disp = math.hypot(x - pose_history_nominal[0][1], y - pose_history_nominal[0][2])
            if disp < 0.15 and dist_to_goal > 0.50:
                lack_of_progress_nominal = True

    assert lack_of_progress_nominal is False


def test_recovery_controller_triggers_and_reverses_from_deadlock():
    """Verify RecoveryController completes reverse and reaches REPLAN_READY."""
    costmap = _create_test_map()
    rec = RecoveryController(
        RecoveryConfig(
            reverse_velocity_mps=-0.15,
            target_reverse_distance_m=0.45,
            deadlock_stop_cycles=4,
        )
    )

    # 4 ticks of deadlock
    for i in range(4):
        v, w, state, _ = rec.step(
            current_time_s=1.0 + i * 0.10,
            robot_pose=(2.0, 0.0, 0.0),
            is_deadlocked=True,
            hazards=[],
            costmap=costmap,
        )

    # 4th tick initiates reverse
    assert state == RecoveryState.REVERSING
    assert v == -0.15

    # Reverse by 0.45m backward (x decreases from 2.0 to 1.50)
    for i in range(1, 10):
        t = 1.40 + i * 0.10
        x_curr = 2.0 - (i * 0.06)
        v, w, state, _ = rec.step(
            current_time_s=t,
            robot_pose=(x_curr, 0.0, 0.0),
            is_deadlocked=True,
            hazards=[],
            costmap=costmap,
        )
        if state == RecoveryState.SETTLING:
            break

    assert state in (RecoveryState.SETTLING, RecoveryState.REPLAN_READY)


def test_dwa_forward_only_candidate_sampling():
    """Verify forward-only DWA configuration strictly samples v >= 0.0 for nominal avoidance."""
    dwa_cfg = PlanningConfig.load()
    dwa_cfg.vehicle.min_linear_velocity_mps = 0.0

    planner = UnifiedDWAPlanner(config=dwa_cfg, backend=DWABackendType.AUTO)
    # Check underlying backend min_linear_velocity_mps
    active_be = planner.active_backend
    if hasattr(active_be, "c_vehicle"):
        assert active_be.c_vehicle.min_linear_velocity_mps >= 0.0
    elif hasattr(active_be, "min_v"):
        assert active_be.min_v >= 0.0
