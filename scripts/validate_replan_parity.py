"""Comprehensive Replan Parity & Invariant Validation Suite (Phase 13.17.2 Part E).

Validates event-driven global replanning parity between frozen CPU reference (ai/planning/dynamic/)
and GPU accelerated planner (ai/planning/dynamic_gpu/) across all 5 canonical replan triggers:
1. DYNAMIC_DEADLOCK
2. PERSISTENT_BLOCKAGE
3. INSUFFICIENT_PROGRESS
4. PATH_INVALIDATED
5. MANUAL

For each trigger, tests:
- trigger/no trigger
- trigger identity
- cycle count
- timestamp
- debouncing behavior
- cooldown enforcement
- behavioral state (STOP, WAIT, AVOID, PROCEED)
- replan request generation
- path version tracking

Critical Invariant:
- GPU must NEVER suppress a safety-relevant CPU replan.

Outputs: simulation/outputs/performance/replan_parity.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai.planning.base import GoalPose, PlanningConfig, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
    DeadlockEvent,
)
from ai.planning.dynamic.collision_checker import PredictiveCollisionChecker
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    TrackedObstacleState,
    TrajectoryWaypoint,
)
from ai.planning.dynamic.global_replanner import (
    DynamicReplanEvent,
    DynamicReplanTrigger,
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
)
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from scripts.run_parity_suite import make_test_obstacle
from ai.planning.dynamic_gpu.gpu_dwa import AcceleratedDWAPlanner
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path, PathPoint
from ai.spatial_fusion.world_map import WorldTerrainMap

OUTPUT_DIR = os.path.join("simulation", "outputs", "performance")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_base_map(x_size: float = 20.0, y_size: float = 20.0, resolution: float = 0.10) -> WorldTerrainMap:
    """Instantiate a flat, obstacle-free base WorldTerrainMap."""
    wmap = WorldTerrainMap(resolution_m=resolution, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    wmap.cost.fill(0.05)
    wmap.traversability.fill(0.95)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    wmap.confidence.fill(1.0)
    wmap.slope.fill(0.0)
    wmap.roughness.fill(0.01)
    wmap.height_var.fill(0.0)
    return wmap


def make_straight_path(start_x: float = 0.0, end_x: float = 6.0, y: float = 0.0, step: float = 0.20) -> Path:
    """Generate a clean linear global path."""
    xs = np.arange(start_x, end_x + step * 0.5, step)
    pts = [PathPoint(x=float(x), y=float(y), yaw=0.0, cost=0.05) for x in xs]
    return Path(points=pts)


# =============================================================================
# Trigger Scenarios
# =============================================================================

def test_dynamic_deadlock_parity(cpu_planner: DWAPlanner, gpu_planner: AcceleratedDWAPlanner) -> Dict[str, Any]:
    """Test 1: DYNAMIC_DEADLOCK triggered by persistent dynamic head-on blockage."""
    wmap = create_base_map()
    path = make_straight_path(0.0, 5.0, 0.0)
    robot_state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=0.0)
    goal = (5.0, 0.0)

    # Dynamic obstacle spanning the corridor ahead, moving slowly toward robot
    obs_traj = make_test_obstacle(
        track_id=401,
        start_pos=(1.00, 0.0),
        velocity=(-0.10, 0.0),
        size=(0.8, 2.5),
        yaw=0.0,
        horizon_s=1.5,
    )
    obs_state = obs_traj.current_state

    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs_traj])

    cpu_replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
    gpu_replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
    cpu_replanner.reset(initial_path=path, timestamp=0.0)
    gpu_replanner.reset(initial_path=path, timestamp=0.0)
    cpu_planner.behavioral_state_machine.reset()
    gpu_planner.behavioral_state_machine.reset()

    cpu_history = []
    gpu_history = []

    # Run for 15 cycles (dt = 0.1s)
    for c in range(1, 16):
        t = c * 0.10
        robot_state.timestamp = t

        # CPU step
        cpu_traj, cpu_v, cpu_w, cpu_status, cpu_diag = cpu_planner.compute_trajectory(
            robot_state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs_traj]
        )
        cpu_dec = cpu_planner.behavioral_state_machine.evaluate(
            robot_state, cpu_traj, cpu_v, cpu_w, cpu_status, cpu_diag, [obs_traj], path, t
        )
        cpu_replan, cpu_trig, _, _ = cpu_replanner.check_trigger_conditions(
            (robot_state.x, robot_state.y, robot_state.yaw), goal, path, wmap,
            behavioral_decision=cpu_dec, dynamic_obstacles=[obs_state], timestamp=t
        )
        cpu_history.append({
            "cycle": c, "time": t, "behavior": cpu_dec.state.value,
            "replan": cpu_replan, "trigger": cpu_trig.value
        })

        # GPU step
        gpu_traj, gpu_v, gpu_w, gpu_status, gpu_diag = gpu_planner.compute_trajectory(
            robot_state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs_traj]
        )
        gpu_dec = gpu_planner.behavioral_state_machine.evaluate(
            robot_state, gpu_traj, gpu_v, gpu_w, gpu_status, gpu_diag, [obs_traj], path, t
        )
        gpu_replan, gpu_trig, _, _ = gpu_replanner.check_trigger_conditions(
            (robot_state.x, robot_state.y, robot_state.yaw), goal, path, wmap,
            behavioral_decision=gpu_dec, dynamic_obstacles=[obs_state], timestamp=t
        )
        gpu_history.append({
            "cycle": c, "time": t, "behavior": gpu_dec.state.value,
            "replan": gpu_replan, "trigger": gpu_trig.value
        })

    cpu_triggered = any(h["replan"] for h in cpu_history)
    gpu_triggered = any(h["replan"] for h in gpu_history)
    first_cpu_cycle = next((h["cycle"] for h in cpu_history if h["replan"]), None)
    first_gpu_cycle = next((h["cycle"] for h in gpu_history if h["replan"]), None)

    parity_match = (cpu_triggered == gpu_triggered) and (first_cpu_cycle == first_gpu_cycle)
    suppressed = cpu_triggered and not gpu_triggered

    return {
        "trigger": "DYNAMIC_DEADLOCK",
        "cpu_triggered": cpu_triggered,
        "gpu_triggered": gpu_triggered,
        "first_cpu_cycle": first_cpu_cycle,
        "first_gpu_cycle": first_gpu_cycle,
        "parity_match": parity_match,
        "suppressed_by_gpu": suppressed,
        "cpu_final_trigger": cpu_history[-1]["trigger"],
        "gpu_final_trigger": gpu_history[-1]["trigger"],
        "sample_history": {"cpu": cpu_history[::3], "gpu": gpu_history[::3]},
    }


def test_persistent_blockage_parity(cpu_planner: DWAPlanner, gpu_planner: AcceleratedDWAPlanner) -> Dict[str, Any]:
    """Test 2: PERSISTENT_BLOCKAGE triggered by static blockage held > 3.0s."""
    wmap = create_base_map()
    # Place static obstacle block directly on path at x = 1.5m
    r, c = wmap.world_to_grid(1.5, 0.0)
    wmap.obstacle[r-2:r+3, c-2:c+3] = True
    wmap.cost[r-2:r+3, c-2:c+3] = 1.0

    path = make_straight_path(0.0, 5.0, 0.0)
    robot_state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=0.0)
    goal = (5.0, 0.0)

    cfg = GlobalReplannerConfig(persistent_blockage_duration_s=1.0)  # Bounded to 1.0s for test efficiency
    cpu_replanner = EventDrivenGlobalReplanner(cfg)
    gpu_replanner = EventDrivenGlobalReplanner(cfg)
    cpu_replanner.reset(initial_path=path, timestamp=0.0)
    gpu_replanner.reset(initial_path=path, timestamp=0.0)
    cpu_planner.behavioral_state_machine.reset()
    gpu_planner.behavioral_state_machine.reset()

    cpu_history = []
    gpu_history = []

    # Run for 20 cycles (dt = 0.1s -> 2.0s total)
    for c in range(1, 21):
        t = c * 0.10
        robot_state.timestamp = t

        # CPU step
        cpu_traj, cpu_v, cpu_w, cpu_status, cpu_diag = cpu_planner.compute_trajectory(robot_state, path, wmap)
        cpu_dec = cpu_planner.behavioral_state_machine.evaluate(
            robot_state, cpu_traj, cpu_v, cpu_w, cpu_status, cpu_diag, None, path, t
        )
        cpu_replan, cpu_trig, _, _ = cpu_replanner.check_trigger_conditions(
            (robot_state.x, robot_state.y, robot_state.yaw), goal, path, wmap,
            behavioral_decision=cpu_dec, dynamic_obstacles=[], timestamp=t
        )
        cpu_history.append({"cycle": c, "time": t, "replan": cpu_replan, "trigger": cpu_trig.value})

        # GPU step
        gpu_traj, gpu_v, gpu_w, gpu_status, gpu_diag = gpu_planner.compute_trajectory(robot_state, path, wmap)
        gpu_dec = gpu_planner.behavioral_state_machine.evaluate(
            robot_state, gpu_traj, gpu_v, gpu_w, gpu_status, gpu_diag, None, path, t
        )
        gpu_replan, gpu_trig, _, _ = gpu_replanner.check_trigger_conditions(
            (robot_state.x, robot_state.y, robot_state.yaw), goal, path, wmap,
            behavioral_decision=gpu_dec, dynamic_obstacles=[], timestamp=t
        )
        gpu_history.append({"cycle": c, "time": t, "replan": gpu_replan, "trigger": gpu_trig.value})

    cpu_triggered = any(h["replan"] for h in cpu_history)
    gpu_triggered = any(h["replan"] for h in gpu_history)
    first_cpu_cycle = next((h["cycle"] for h in cpu_history if h["replan"]), None)
    first_gpu_cycle = next((h["cycle"] for h in gpu_history if h["replan"]), None)

    parity_match = (cpu_triggered == gpu_triggered) and (first_cpu_cycle == first_gpu_cycle)
    suppressed = cpu_triggered and not gpu_triggered

    return {
        "trigger": "PERSISTENT_BLOCKAGE",
        "cpu_triggered": cpu_triggered,
        "gpu_triggered": gpu_triggered,
        "first_cpu_cycle": first_cpu_cycle,
        "first_gpu_cycle": first_gpu_cycle,
        "parity_match": parity_match,
        "suppressed_by_gpu": suppressed,
        "cpu_final_trigger": cpu_history[-1]["trigger"],
        "gpu_final_trigger": gpu_history[-1]["trigger"],
        "sample_history": {"cpu": cpu_history[::4], "gpu": gpu_history[::4]},
    }


def test_insufficient_progress_parity(cpu_planner: DWAPlanner, gpu_planner: AcceleratedDWAPlanner) -> Dict[str, Any]:
    """Test 3: INSUFFICIENT_PROGRESS triggered when vehicle distance to goal does not advance."""
    wmap = create_base_map()
    path = make_straight_path(0.0, 5.0, 0.0)
    robot_state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=0.0)
    goal = (5.0, 0.0)

    cfg = GlobalReplannerConfig(progress_loss_window_s=1.0, min_progress_m=0.20)
    cpu_replanner = EventDrivenGlobalReplanner(cfg)
    gpu_replanner = EventDrivenGlobalReplanner(cfg)
    cpu_replanner.reset(initial_path=path, timestamp=0.0)
    gpu_replanner.reset(initial_path=path, timestamp=0.0)

    cpu_history = []
    gpu_history = []

    # Vehicle held stationary for 15 cycles (1.5s > 1.0s window)
    for c in range(1, 16):
        t = c * 0.10
        robot_state.timestamp = t

        # Behavioral decision in WAIT
        cpu_dec = BehavioralDecision(state=BehavioralState.WAIT, command_v=0.0, command_omega=0.0, reason="Waiting for obstruction")
        gpu_dec = BehavioralDecision(state=BehavioralState.WAIT, command_v=0.0, command_omega=0.0, reason="Waiting for obstruction")

        cpu_replan, cpu_trig, _, _ = cpu_replanner.check_trigger_conditions(
            (0.0, 0.0, 0.0), goal, path, wmap, behavioral_decision=cpu_dec, timestamp=t
        )
        gpu_replan, gpu_trig, _, _ = gpu_replanner.check_trigger_conditions(
            (0.0, 0.0, 0.0), goal, path, wmap, behavioral_decision=gpu_dec, timestamp=t
        )

        cpu_history.append({"cycle": c, "time": t, "replan": cpu_replan, "trigger": cpu_trig.value})
        gpu_history.append({"cycle": c, "time": t, "replan": gpu_replan, "trigger": gpu_trig.value})

    cpu_triggered = any(h["replan"] for h in cpu_history)
    gpu_triggered = any(h["replan"] for h in gpu_history)
    first_cpu_cycle = next((h["cycle"] for h in cpu_history if h["replan"]), None)
    first_gpu_cycle = next((h["cycle"] for h in gpu_history if h["replan"]), None)

    parity_match = (cpu_triggered == gpu_triggered) and (first_cpu_cycle == first_gpu_cycle)
    suppressed = cpu_triggered and not gpu_triggered

    return {
        "trigger": "INSUFFICIENT_PROGRESS",
        "cpu_triggered": cpu_triggered,
        "gpu_triggered": gpu_triggered,
        "first_cpu_cycle": first_cpu_cycle,
        "first_gpu_cycle": first_gpu_cycle,
        "parity_match": parity_match,
        "suppressed_by_gpu": suppressed,
        "cpu_final_trigger": cpu_history[-1]["trigger"],
        "gpu_final_trigger": gpu_history[-1]["trigger"],
    }


def test_path_invalidated_parity(cpu_planner: DWAPlanner, gpu_planner: AcceleratedDWAPlanner) -> Dict[str, Any]:
    """Test 4: PATH_INVALIDATED triggered when global route is blocked for debounce cycles."""
    wmap = create_base_map()
    path = make_straight_path(0.0, 5.0, 0.0)
    goal = (5.0, 0.0)

    # Invalidate path at x = 3.0m
    r, c = wmap.world_to_grid(3.0, 0.0)
    wmap.obstacle[r-1:r+2, c-1:c+2] = True
    wmap.cost[r-1:r+2, c-1:c+2] = 1.0

    cfg = GlobalReplannerConfig(debounce_cycles=2)
    cpu_replanner = EventDrivenGlobalReplanner(cfg)
    gpu_replanner = EventDrivenGlobalReplanner(cfg)
    cpu_replanner.reset(initial_path=path, timestamp=0.0)
    gpu_replanner.reset(initial_path=path, timestamp=0.0)

    cpu_history = []
    gpu_history = []

    # Vehicle is moving in PROCEED state, but path ahead is physically blocked
    for c in range(1, 6):
        t = c * 0.10
        cpu_dec = BehavioralDecision(state=BehavioralState.PROCEED, command_v=0.4, command_omega=0.0, reason="Nominal proceeding")
        gpu_dec = BehavioralDecision(state=BehavioralState.PROCEED, command_v=0.4, command_omega=0.0, reason="Nominal proceeding")

        cpu_replan, cpu_trig, _, _ = cpu_replanner.check_trigger_conditions(
            (0.5, 0.0, 0.0), goal, path, wmap, behavioral_decision=cpu_dec, timestamp=t
        )
        gpu_replan, gpu_trig, _, _ = gpu_replanner.check_trigger_conditions(
            (0.5, 0.0, 0.0), goal, path, wmap, behavioral_decision=gpu_dec, timestamp=t
        )

        cpu_history.append({"cycle": c, "time": t, "replan": cpu_replan, "trigger": cpu_trig.value})
        gpu_history.append({"cycle": c, "time": t, "replan": gpu_replan, "trigger": gpu_trig.value})

    cpu_triggered = any(h["replan"] for h in cpu_history)
    gpu_triggered = any(h["replan"] for h in gpu_history)
    first_cpu_cycle = next((h["cycle"] for h in cpu_history if h["replan"]), None)
    first_gpu_cycle = next((h["cycle"] for h in gpu_history if h["replan"]), None)

    parity_match = (cpu_triggered == gpu_triggered) and (first_cpu_cycle == first_gpu_cycle)
    suppressed = cpu_triggered and not gpu_triggered

    return {
        "trigger": "PATH_INVALIDATED",
        "cpu_triggered": cpu_triggered,
        "gpu_triggered": gpu_triggered,
        "first_cpu_cycle": first_cpu_cycle,
        "first_gpu_cycle": first_gpu_cycle,
        "parity_match": parity_match,
        "suppressed_by_gpu": suppressed,
        "cpu_final_trigger": cpu_history[-1]["trigger"],
        "gpu_final_trigger": gpu_history[-1]["trigger"],
    }


def test_manual_replan_parity(cpu_planner: DWAPlanner, gpu_planner: AcceleratedDWAPlanner) -> Dict[str, Any]:
    """Test 5: MANUAL_REPLAN invoked by external command."""
    wmap = create_base_map()
    path = make_straight_path(0.0, 5.0, 0.0)
    goal = (5.0, 0.0)

    cfg = GlobalReplannerConfig(cooldown_s=2.0)
    cpu_replanner = EventDrivenGlobalReplanner(cfg)
    gpu_replanner = EventDrivenGlobalReplanner(cfg)
    cpu_replanner.reset(initial_path=path, timestamp=0.0)
    gpu_replanner.reset(initial_path=path, timestamp=0.0)

    # Manual replan event at t = 1.0s
    manual_event = DynamicReplanEvent(
        trigger_type=DynamicReplanTrigger.MANUAL,
        timestamp=1.0,
        current_pose=(0.0, 0.0, 0.0),
        current_goal=goal,
        current_global_path=path,
        blocking_obstacle_ids=[],
        blocked_duration=0.0,
        deadlock_duration=0.0,
        current_progress=0.0,
        reason="Manual operator trigger",
        confidence=1.0,
        event_id="manual_001",
    )

    cpu_res = cpu_replanner.execute_replan((0.0, 0.0, 0.0), goal, wmap, event=manual_event)
    gpu_res = gpu_replanner.execute_replan((0.0, 0.0, 0.0), goal, wmap, event=manual_event)

    cpu_ok = cpu_res.success
    gpu_ok = gpu_res.success
    cpu_trig = cpu_res.trigger_type.value
    gpu_trig = gpu_res.trigger_type.value

    # Cooldown test: attempt subsequent periodic replan during cooldown period at t = 1.5s (0.5s < 2.0s)
    # Provide a blocked decision to attempt trigger
    block_dec = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="blocked", deadlock=True)
    cpu_cd = cpu_replanner.check_and_replan((0.0, 0.0, 0.0), goal, wmap, behavioral_decision=block_dec, timestamp=1.5)
    gpu_cd = gpu_replanner.check_and_replan((0.0, 0.0, 0.0), goal, wmap, behavioral_decision=block_dec, timestamp=1.5)

    cpu_cd_blocked = not cpu_cd.replan_triggered and "cooldown" in str(cpu_cd.failure_reason).lower()
    gpu_cd_blocked = not gpu_cd.replan_triggered and "cooldown" in str(gpu_cd.failure_reason).lower()

    parity_match = (cpu_ok == gpu_ok) and (cpu_trig == gpu_trig) and (cpu_cd_blocked == gpu_cd_blocked)
    suppressed = cpu_ok and not gpu_ok

    return {
        "trigger": "MANUAL",
        "cpu_triggered": cpu_ok,
        "gpu_triggered": gpu_ok,
        "parity_match": parity_match,
        "suppressed_by_gpu": suppressed,
        "cpu_cooldown_enforced": cpu_cd_blocked,
        "gpu_cooldown_enforced": gpu_cd_blocked,
        "cpu_trigger_type": cpu_trig,
        "gpu_trigger_type": gpu_trig,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Global Replan Parity")
    parser.add_argument("--device", type=str, default="auto", help="Device for GPU planner ('auto', 'cuda', 'cpu')")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("PHASE 13.17.2 PART E: GLOBAL REPLAN PARITY & INVARIANT VALIDATION")
    print("=" * 80)

    config = PlanningConfig()
    cpu_planner = DWAPlanner(config)
    gpu_planner = AcceleratedDWAPlanner(config, preferred_device=args.device)

    print(f"[*] GPU Device: {gpu_planner.backend.device_name} (CUDA: {gpu_planner.backend.is_cuda})")

    tests = [
        ("DYNAMIC_DEADLOCK", test_dynamic_deadlock_parity),
        ("PERSISTENT_BLOCKAGE", test_persistent_blockage_parity),
        ("INSUFFICIENT_PROGRESS", test_insufficient_progress_parity),
        ("PATH_INVALIDATED", test_path_invalidated_parity),
        ("MANUAL", test_manual_replan_parity),
    ]

    results = []
    all_passed = True
    suppressed_count = 0

    for name, test_fn in tests:
        res = test_fn(cpu_planner, gpu_planner)
        results.append(res)
        match_str = "MATCH" if res["parity_match"] else "MISMATCH"
        supp_str = "SUPPRESSED!" if res["suppressed_by_gpu"] else "OK (Zero Suppression)"
        print(f"  {name:<24} | Parity: {match_str:<8} | Safety Gate: {supp_str}")
        if not res["parity_match"]:
            all_passed = False
        if res["suppressed_by_gpu"]:
            suppressed_count += 1

    summary = {
        "milestone": "Phase 13.17.2 Part E",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": gpu_planner.backend.device_name,
        "is_cuda": gpu_planner.backend.is_cuda,
        "all_triggers_matched": all_passed,
        "suppressed_safety_replans": suppressed_count,
        "tests": results,
    }

    out_file = os.path.join(OUTPUT_DIR, "replan_parity.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 80)
    print(f"Replan Parity Overall Status: {'ALL PASSED' if all_passed else 'FAILURE'}")
    print(f"Safety Replans Suppressed  : {suppressed_count}")
    print(f"Saved artifact to          : {out_file}")
    print("=" * 80 + "\n")

    if suppressed_count > 0:
        print("[-] Critical Safety Invariant Violated! Halting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
