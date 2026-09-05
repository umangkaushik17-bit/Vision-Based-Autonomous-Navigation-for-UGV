"""Benchmark and Validation Script for Milestone 13.10: Behavioral State Machine.

Evaluates 10 deterministic operational scenarios (A through J):
- Scenario A: Unobstructed travel (PROCEED)
- Scenario B: Obstacle crossing path with safe detour (AVOID)
- Scenario C: Obstacle stopped in narrow corridor (STOP)
- Scenario D: Temporary obstacle that later clears (WAIT -> PROCEED)
- Scenario E: All paths blocked (STOP)
- Scenario F: Unknown prediction horizon (STOP / UNKNOWN_HORIZON)
- Scenario G: Obstacle disappears while waiting (WAIT -> PROCEED recovery)
- Scenario H: Repeated blockage / deadlock (STOP -> DYNAMIC_DEADLOCK)
- Scenario I: Noisy obstacle state (Hysteresis churn suppression)
- Scenario J: Multiple dynamic obstacles (Multi-agent tactical arbitration)

Measures and reports:
- Isolated BehavioralStateMachine latency (mean, median, P95, max)
- State sequences across cycles
- Selected commands (v, omega)
- Active blocking obstacle IDs
- Wait elapsed durations
- Deadlock events
- Whether stop occurred
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ai.planning.base import PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
    DeadlockEvent,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
)
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory, TrajectoryPoint


def make_test_obstacle(
    track_id: int,
    start_pos: Tuple[float, float],
    velocity: Tuple[float, float],
    size: Tuple[float, float] = (0.6, 0.4),
    yaw: float = 0.0,
    horizon_s: float = 2.0,
    dt: float = 0.10,
    base_unc: float = 0.10,
    unc_growth: float = 0.05,
) -> PredictedObstacleTrajectory:
    """Generate synthetic obstacle trajectory."""
    cov = np.diag([0.04, 0.04, 0.01, 0.01]).astype(np.float64)
    state = TrackedObstacleState(
        track_id=track_id,
        timestamp=0.0,
        x=float(start_pos[0]),
        y=float(start_pos[1]),
        vx=float(velocity[0]),
        vy=float(velocity[1]),
        covariance=cov,
        status="TRACKING",
        is_confirmed=True,
        bbox_size=size,
        bbox_yaw=yaw,
    )
    n_steps = int(round(horizon_s / dt))
    wps: List[TrajectoryWaypoint] = []
    polys: List[List[Tuple[float, float]]] = []

    for step in range(n_steps + 1):
        t = round(step * dt, 4)
        x = start_pos[0] + velocity[0] * t
        y = start_pos[1] + velocity[1] * t
        r_unc = base_unc + unc_growth * t
        cov_2d = np.diag([(r_unc / 3.0) ** 2, (r_unc / 3.0) ** 2]).astype(np.float64)
        wps.append(
            TrajectoryWaypoint(
                time_offset=t,
                timestamp=t,
                x=x,
                y=y,
                vx=velocity[0],
                vy=velocity[1],
                covariance_2d=cov_2d,
                uncertainty_radius=r_unc,
            )
        )
        polys.append(compute_obb_corners((x, y), size, yaw, margin_m=r_unc))

    return PredictedObstacleTrajectory(
        track_id=track_id,
        current_state=state,
        horizon_s=horizon_s,
        dt=dt,
        waypoints=wps,
        bounding_polygon_per_step=polys,
    )


def run_benchmark_scenarios(n_warmup: int = 50, n_measure: int = 200) -> Dict[str, Any]:
    """Execute 10 deterministic scenarios and measure isolated state machine latency."""
    cfg = BehavioralConfig(
        wait_timeout_s=2.5,
        min_dwell_time_s=0.20,
        recovery_confirmation_cycles=2,
        deadlock_consecutive_stop_cycles=5,
    )

    scenarios = [
        "Scenario A: Unobstructed Travel",
        "Scenario B: Obstacle Crossing with Safe Detour",
        "Scenario C: Obstacle Stopped in Narrow Corridor",
        "Scenario D: Temporary Obstacle that Clears",
        "Scenario E: All Paths Blocked",
        "Scenario F: Unknown Prediction Horizon",
        "Scenario G: Obstacle Disappears While Waiting",
        "Scenario H: Repeated Blockage / Deadlock",
        "Scenario I: Noisy Obstacle State (Hysteresis)",
        "Scenario J: Multiple Dynamic Obstacles",
    ]

    scenario_reports: List[Dict[str, Any]] = []
    all_latencies_us: List[float] = []

    for sc_name in scenarios:
        sm = BehavioralStateMachine(cfg)
        decisions: List[BehavioralDecision] = []
        latencies: List[float] = []

        # Define 10-step temporal progression for this scenario
        for step in range(10):
            t = step * 0.20  # 5 Hz control cycle

            if "Scenario A" in sc_name:
                # Unobstructed straight travel
                state = RobotState(x=t * 0.4, y=0.0, yaw=0.0, v=0.40, omega=0.0, timestamp=t)
                traj = Trajectory([TrajectoryPoint(0, 0, 0, 0, 0.40, 0.0, 0.1, 1.5)])
                status = PlannerStatus.SUCCESS
                diag = {"dynamic_collision": False, "dynamic_risk": "FREE", "valid_trajectories": 100}
                tracks = []

            elif "Scenario B" in sc_name:
                # Obstacle crossing at x=2.0, UGV executes safe right detour
                state = RobotState(x=t * 0.3, y=0.0, yaw=-0.1, v=0.30, omega=-0.12, timestamp=t)
                obs = make_test_obstacle(track_id=1, start_pos=(2.0, -1.0 + t * 0.5), velocity=(0.0, 0.5))
                traj = Trajectory([TrajectoryPoint(0, 0, 0, 0, 0.30, -0.12, 0.2, 1.2)])
                status = PlannerStatus.SUCCESS
                diag = {
                    "dynamic_collision": False,
                    "dynamic_risk": "MEDIUM",
                    "valid_trajectories": 25,
                    "dynamically_penalized_candidates": 40,
                    "blocking_obstacle_id": 1,
                }
                tracks = [obs]

            elif "Scenario C" in sc_name:
                # Obstacle stopped right in narrow corridor, non-clearing
                state = RobotState(x=1.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=t)
                obs = make_test_obstacle(track_id=2, start_pos=(1.8, 0.0), velocity=(0.0, 0.0), size=(2.0, 0.8))
                status = PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY
                diag = {
                    "dynamic_collision": True,
                    "dynamic_risk": "LETHAL",
                    "valid_trajectories": 0,
                    "blocking_obstacle_id": 2,
                    "emergency_stop_triggered": True,
                }
                traj = None
                tracks = [obs]

            elif "Scenario D" in sc_name:
                # Obstacle crossing at t in [0.0, 0.8], clears at t >= 1.0
                state = RobotState(x=0.5, y=0.0, yaw=0.0, v=0.0 if t < 1.0 else 0.3, omega=0.0, timestamp=t)
                # Moving at 1.2 m/s laterally
                obs_y = -0.6 + t * 1.2
                obs = make_test_obstacle(track_id=3, start_pos=(1.5, obs_y), velocity=(0.0, 1.2))
                if t < 0.8:
                    status = PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY
                    diag = {"dynamic_collision": False, "dynamic_risk": "HIGH", "valid_trajectories": 0}
                    traj = None
                else:
                    status = PlannerStatus.SUCCESS
                    diag = {"dynamic_collision": False, "dynamic_risk": "FREE", "valid_trajectories": 50}
                    traj = Trajectory([TrajectoryPoint(0, 0, 0, 0, 0.35, 0.0, 0.1, 1.5)])
                tracks = [obs]

            elif "Scenario E" in sc_name:
                # All paths blocked by terrain and obstacles
                state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=t)
                status = PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY
                diag = {"dynamic_collision": False, "dynamic_risk": "LETHAL", "valid_trajectories": 0, "emergency_stop_triggered": True}
                traj = None
                tracks = []

            elif "Scenario F" in sc_name:
                # Obstacle prediction horizon exceeded (unknown horizon)
                state = RobotState(x=t * 0.2, y=0.0, yaw=0.0, v=0.20, omega=0.0, timestamp=t)
                obs = make_test_obstacle(track_id=4, start_pos=(1.2, 0.0), velocity=(0.0, 0.0), horizon_s=0.5)
                status = PlannerStatus.SUCCESS
                diag = {
                    "dynamic_collision": False,
                    "dynamic_risk": "UNKNOWN_HORIZON",
                    "unknown_horizon": True,
                    "valid_trajectories": 10,
                }
                traj = Trajectory([TrajectoryPoint(0, 0, 0, 0, 0.20, 0.0, 0.1, 1.0)])
                tracks = [obs]

            elif "Scenario G" in sc_name:
                # Obstacle present initially, then disappears at step >= 3
                state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0 if step < 4 else 0.35, omega=0.0, timestamp=t)
                if step < 3:
                    obs = make_test_obstacle(track_id=5, start_pos=(1.2, 0.1), velocity=(0.0, 0.8))
                    status = PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY
                    diag = {"dynamic_collision": False, "dynamic_risk": "HIGH", "valid_trajectories": 0}
                    traj = None
                    tracks = [obs]
                else:
                    status = PlannerStatus.SUCCESS
                    diag = {"dynamic_collision": False, "dynamic_risk": "FREE", "valid_trajectories": 80}
                    traj = Trajectory([TrajectoryPoint(0, 0, 0, 0, 0.35, 0.0, 0.1, 1.5)])
                    tracks = []

            elif "Scenario H" in sc_name:
                # Persistent blockage creating DYNAMIC_DEADLOCK after 5 cycles
                state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=t)
                status = PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY
                diag = {"dynamic_collision": True, "dynamic_risk": "LETHAL", "valid_trajectories": 0, "emergency_stop_triggered": True}
                traj = None
                tracks = []

            elif "Scenario I" in sc_name:
                # Noisy obstacle state (alternating single-frame risk spikes)
                state = RobotState(x=t * 0.3, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=t)
                is_spike = (step % 2 == 1) and (step < 6)
                status = PlannerStatus.SUCCESS
                diag = {
                    "dynamic_collision": False,
                    "dynamic_risk": "HIGH" if is_spike else "FREE",
                    "valid_trajectories": 15 if is_spike else 90,
                    "dynamically_rejected_candidates": 10 if is_spike else 0,
                }
                traj = Trajectory([TrajectoryPoint(0, 0, 0, 0, 0.30, 0.0, 0.1, 1.2)])
                tracks = []

            else:  # Scenario J
                # Multiple dynamic obstacles crossing simultaneously
                state = RobotState(x=t * 0.25, y=0.0, yaw=0.1, v=0.25, omega=0.08, timestamp=t)
                obs1 = make_test_obstacle(track_id=1, start_pos=(1.5, -0.8 + t * 0.4), velocity=(0.0, 0.4))
                obs2 = make_test_obstacle(track_id=2, start_pos=(2.2, 0.8 - t * 0.3), velocity=(0.0, -0.3))
                status = PlannerStatus.SUCCESS
                diag = {
                    "dynamic_collision": False,
                    "dynamic_risk": "MEDIUM",
                    "valid_trajectories": 30,
                    "dynamically_penalized_candidates": 20,
                    "blocking_obstacle_id": 1,
                }
                traj = Trajectory([TrajectoryPoint(0, 0, 0, 0, 0.25, 0.08, 0.15, 1.2)])
                tracks = [obs1, obs2]

            # Measure isolated state machine latency
            t0 = time.perf_counter()
            dec = sm.evaluate(
                current_state=state,
                best_trajectory=traj,
                best_v=traj.points[-1].v if traj else 0.0,
                best_w=traj.points[-1].omega if traj else 0.0,
                planner_status=status,
                dwa_diagnostics=diag,
                predicted_tracks=tracks,
                current_time_s=t,
            )
            lat_us = (time.perf_counter() - t0) * 1e6
            latencies.append(lat_us)
            all_latencies_us.append(lat_us)
            decisions.append(dec)

        state_seq = " -> ".join([d.state.value for d in decisions])
        last_d = decisions[-1]
        deadlock_occurred = any(d.deadlock for d in decisions)
        stop_occurred = any(d.state == BehavioralState.STOP for d in decisions)
        max_wait = max((d.wait_elapsed_s for d in decisions), default=0.0)

        scenario_reports.append({
            "scenario": sc_name,
            "state_sequence": state_seq,
            "final_state": last_d.state.value,
            "command": f"({last_d.command_v:.2f}, {last_d.command_omega:.2f})",
            "reason": last_d.reason,
            "obstacle_id": last_d.blocking_obstacle_id,
            "max_wait_duration_s": max_wait,
            "deadlock": deadlock_occurred,
            "stop_occurred": stop_occurred,
            "latency_mean_us": float(np.mean(latencies)),
            "latency_p95_us": float(np.percentile(latencies, 95)),
        })

    # High-repetition isolated latency benchmark
    sm_bench = BehavioralStateMachine(cfg)
    test_state = RobotState(x=1.0, y=0.0, yaw=0.0, v=0.3, omega=0.0, timestamp=1.0)
    test_traj = Trajectory([TrajectoryPoint(0, 0, 0, 0, 0.3, 0.0, 0.1, 1.0)])
    test_diag = {"dynamic_collision": False, "dynamic_risk": "FREE", "valid_trajectories": 50}

    # Warmup
    for _ in range(n_warmup):
        sm_bench.evaluate(test_state, test_traj, 0.3, 0.0, PlannerStatus.SUCCESS, test_diag, current_time_s=1.0)

    perf_latencies_us = []
    for i in range(n_measure):
        t0 = time.perf_counter()
        sm_bench.evaluate(test_state, test_traj, 0.3, 0.0, PlannerStatus.SUCCESS, test_diag, current_time_s=1.0 + i * 0.1)
        perf_latencies_us.append((time.perf_counter() - t0) * 1e6)

    return {
        "scenarios": scenario_reports,
        "overall_latency_us": {
            "mean": float(np.mean(perf_latencies_us)),
            "median": float(np.median(perf_latencies_us)),
            "p95": float(np.percentile(perf_latencies_us, 95)),
            "max": float(np.max(perf_latencies_us)),
        },
    }


if __name__ == "__main__":
    print("\n" + "=" * 115)
    print("MILESTONE 13.10: BEHAVIORAL STATE MACHINE BENCHMARK & SCENARIO VALIDATION")
    print("=" * 115)

    res = run_benchmark_scenarios()
    scenarios = res["scenarios"]

    header = f"{'Scenario':<42} | {'Final State':<11} | {'Command (v, w)':<14} | {'Wait (s)':<8} | {'Stop?':<5} | {'Deadlock?':<9} | {'Mean (us)'}"
    print(header)
    print("-" * 115)

    for sc in scenarios:
        print(
            f"{sc['scenario']:<42} | "
            f"{sc['final_state']:<11} | "
            f"{sc['command']:<14} | "
            f"{sc['max_wait_duration_s']:<8.2f} | "
            f"{str(sc['stop_occurred']):<5} | "
            f"{str(sc['deadlock']):<9} | "
            f"{sc['latency_mean_us']:<8.2f}"
        )

    print("-" * 115)
    print("\nDETAILED SCENARIO STATE SEQUENCES:")
    for sc in scenarios:
        print(f"\n[{sc['scenario']}]")
        print(f"  State Sequence: {sc['state_sequence']}")
        print(f"  Final Reason:   {sc['reason']}")
        print(f"  Obstacle ID:    {sc['obstacle_id']}")
        print(f"  Deadlock Event: {sc['deadlock']}")

    lat = res["overall_latency_us"]
    print("\n" + "=" * 115)
    print("BEHAVIORAL STATE MACHINE ISOLATED EVALUATION LATENCY (N=200 cycles):")
    print(f"  Mean:   {lat['mean']:.3f} us  ({lat['mean']/1000.0:.4f} ms)")
    print(f"  Median: {lat['median']:.3f} us  ({lat['median']/1000.0:.4f} ms)")
    print(f"  P95:    {lat['p95']:.3f} us  ({lat['p95']/1000.0:.4f} ms)")
    print(f"  Max:    {lat['max']:.3f} us  ({lat['max']/1000.0:.4f} ms)")
    print("=" * 115 + "\n")
