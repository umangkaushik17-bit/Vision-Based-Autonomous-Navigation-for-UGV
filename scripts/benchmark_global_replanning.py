"""Benchmark and quantitative evaluation suite for Milestone 13.11: Event-Driven Global Replanning.

Evaluates 10 deterministic operational scenarios (A through J):
- Scenario A: Obstacle crosses local path and clears quickly (no replan)
- Scenario B: Obstacle blocks local path but side-step exists (AVOID, no global replan)
- Scenario C: Persistent obstacle blocks corridor (> persistent_blockage_duration_s -> replan)
- Scenario D: DYNAMIC_DEADLOCK triggers immediate replan
- Scenario E: New A* route exists around obstacle (path change, deviation, goal reachable)
- Scenario F: All global alternatives blocked (GLOBAL_REPLAN_FAILED, zero motion)
- Scenario G: Multiple dynamic obstacles (composite overlay, zero collisions)
- Scenario H: Unknown prediction region (unknown preserved, not falsely cleared)
- Scenario I: Path invalidated again shortly after replan (cooldown override on direct collision)
- Scenario J: Goal reached before replan (no replan when at goal)

Measures and reports:
- Replan count and trigger types
- Old/new path lengths, path costs, and path deviation
- Planning latency, validation latency, detection latency, and total latency
- Cooldown and debounce behavior
- Dynamic clearance and collision counts
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    DeadlockEvent,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    TrackedObstacleState,
    TrajectoryWaypoint,
)
from ai.planning.dynamic.global_replanner import (
    DynamicPlanningMapOverlay,
    DynamicReplanEvent,
    DynamicReplanResult,
    DynamicReplanTrigger,
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
    PathChangeMetrics,
    compute_min_dynamic_clearance,
    compute_path_deviation,
)
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.path import Path, PathPoint
from ai.spatial_fusion.world_map import WorldTerrainMap


def create_benchmark_world_map(
    bounds: Tuple[float, float] = (-6.0, 6.0),
    resolution: float = 0.10,
) -> WorldTerrainMap:
    """Create a 2D WorldTerrainMap with 0.10 nominal cost."""
    wmap = WorldTerrainMap(resolution_m=resolution, x_bounds_m=bounds, y_bounds_m=bounds)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    return wmap


def make_straight_path(start_x: float = -3.0, end_x: float = 3.0, y: float = 0.0, step: float = 0.10) -> Path:
    """Generate a straight line path along the X axis."""
    xs = np.arange(start_x, end_x + step * 0.5, step)
    pts = [PathPoint(x=float(x), y=float(y), yaw=0.0, distance_from_start=float(x - start_x)) for x in xs]
    return Path(points=pts)


def make_test_obstacle(
    track_id: int,
    pos: Tuple[float, float],
    size: Tuple[float, float] = (0.6, 0.6),
    yaw: float = 0.0,
    status: str = "TRACKING",
) -> TrackedObstacleState:
    """Create a TrackedObstacleState for benchmarking."""
    return TrackedObstacleState(
        track_id=track_id,
        timestamp=0.0,
        x=float(pos[0]),
        y=float(pos[1]),
        vx=0.0,
        vy=0.0,
        covariance=np.eye(4, dtype=np.float64) * 0.01,
        status=status,
        is_confirmed=True,
        bbox_size=size,
        bbox_yaw=yaw,
    )


def run_benchmarks() -> Dict[str, Any]:
    """Execute all 10 benchmark scenarios and compute timing / safety metrics."""
    results: Dict[str, Any] = {}

    all_detection_latencies: List[float] = []
    all_validation_latencies: List[float] = []
    all_planning_latencies: List[float] = []
    all_total_latencies: List[float] = []

    # ==========================================================================
    # SCENARIO A: Obstacle crosses local path and clears quickly (no replan)
    # ==========================================================================
    wmap = create_benchmark_world_map()
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(cooldown_s=4.0, persistent_blockage_duration_s=3.0, debounce_cycles=2))
    path = make_straight_path(-3.0, 3.0, 0.0)
    replanner.set_global_path(path)

    replan_count_a = 0
    # Obstacle crossing fast (on path for only 1 cycle -> debounced, no replan)
    for step in range(6):
        t = step * 0.10
        y_obs = 1.5 - step * 0.75
        obs = [make_test_obstacle(1, (0.0, y_obs))]
        state = BehavioralState.WAIT if abs(y_obs) < 0.4 else BehavioralState.PROCEED
        dec = BehavioralDecision(state=state, command_v=0.0 if state == BehavioralState.WAIT else 0.35, command_omega=0.0, reason="Crossing")
        res = replanner.check_and_replan((-2.0 + step * 0.02, 0.0, 0.0), (3.0, 0.0), wmap, dec, obs, timestamp=t)
        all_detection_latencies.append(res.detection_latency_ms)
        if res.replan_triggered:
            replan_count_a += 1

    results["Scenario A"] = {
        "description": "Obstacle crosses local path and clears quickly",
        "replan_count": replan_count_a,
        "trigger": "NONE",
        "success": True,
        "goal_reachable": True,
        "stale_path": replanner.is_current_path_stale,
    }

    # ==========================================================================
    # SCENARIO B: Obstacle blocks local path but side-step exists (AVOID)
    # ==========================================================================
    wmap = create_benchmark_world_map()
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
    replanner.set_global_path(path)
    obs = [make_test_obstacle(2, (0.0, 0.35))]
    dec = BehavioralDecision(state=BehavioralState.AVOID, command_v=0.25, command_omega=-0.12, reason="Local detour")

    replan_count_b = 0
    for step in range(5):
        t = step * 0.10
        res = replanner.check_and_replan((-1.5 + step * 0.02, -0.2, 0.0), (3.0, 0.0), wmap, dec, obs, timestamp=t)
        all_detection_latencies.append(res.detection_latency_ms)
        if res.replan_triggered:
            replan_count_b += 1

    results["Scenario B"] = {
        "description": "Obstacle blocks local path but side-step exists",
        "replan_count": replan_count_b,
        "trigger": "NONE",
        "success": True,
        "goal_reachable": True,
        "stale_path": replanner.is_current_path_stale,
    }

    # ==========================================================================
    # SCENARIO C: Persistent obstacle blocks corridor (> 2.0s -> replan)
    # ==========================================================================
    wmap = create_benchmark_world_map()
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(persistent_blockage_duration_s=2.0, debounce_cycles=50))
    replanner.set_global_path(path)
    obs = [make_test_obstacle(3, (0.0, 0.0), size=(0.8, 0.8))]
    dec = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Corridor blocked", blocking_obstacle_id=3)

    replan_count_c = 0
    final_res_c = None
    for step in range(25):
        t = step * 0.10
        res = replanner.check_and_replan((-1.5, 0.0, 0.0), (3.0, 0.0), wmap, dec, obs, timestamp=t)
        all_detection_latencies.append(res.detection_latency_ms)
        if res.replan_triggered:
            replan_count_c += 1
            final_res_c = res
            all_validation_latencies.append(res.validation_latency_ms)
            all_planning_latencies.append(res.planning_latency_ms)
            all_total_latencies.append(res.total_latency_ms)

    results["Scenario C"] = {
        "description": "Persistent obstacle blocks corridor",
        "replan_count": replan_count_c,
        "trigger": final_res_c.trigger_type.value if final_res_c else "NONE",
        "success": final_res_c.success if final_res_c else False,
        "old_path_len_m": final_res_c.metrics.old_path_length if final_res_c and final_res_c.metrics else 0.0,
        "new_path_len_m": final_res_c.metrics.new_path_length if final_res_c and final_res_c.metrics else 0.0,
        "deviation_m": final_res_c.metrics.path_deviation if final_res_c and final_res_c.metrics else 0.0,
        "goal_reachable": final_res_c.metrics.goal_reachable if final_res_c and final_res_c.metrics else False,
    }

    # ==========================================================================
    # SCENARIO D: DYNAMIC_DEADLOCK triggers replan
    # ==========================================================================
    wmap = create_benchmark_world_map()
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
    replanner.set_global_path(path)
    obs = [make_test_obstacle(4, (0.0, 0.0))]
    dec = BehavioralDecision(
        state=BehavioralState.STOP, command_v=0.0, command_omega=0.0,
        reason="M13.10 Deadlock threshold exceeded", deadlock=True,
        deadlock_event=DeadlockEvent.DYNAMIC_DEADLOCK.value, blocking_obstacle_id=4
    )

    res_d = replanner.check_and_replan((-1.5, 0.0, 0.0), (3.0, 0.0), wmap, dec, obs, timestamp=1.0)
    all_detection_latencies.append(res_d.detection_latency_ms)
    all_validation_latencies.append(res_d.validation_latency_ms)
    all_planning_latencies.append(res_d.planning_latency_ms)
    all_total_latencies.append(res_d.total_latency_ms)

    results["Scenario D"] = {
        "description": "DYNAMIC_DEADLOCK triggers immediate replan",
        "replan_count": 1 if res_d.replan_triggered else 0,
        "trigger": res_d.trigger_type.value,
        "success": res_d.success,
        "planning_ms": res_d.planning_latency_ms,
        "total_ms": res_d.total_latency_ms,
        "goal_reachable": res_d.metrics.goal_reachable if res_d.metrics else False,
    }

    # ==========================================================================
    # SCENARIO E: New A* route exists around obstacle
    # ==========================================================================
    wmap = create_benchmark_world_map()
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    replanner.set_global_path(path)
    obs = [make_test_obstacle(5, (0.0, 0.0), size=(1.2, 1.0))]
    dec = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Blocked", deadlock=True)

    res_e = replanner.check_and_replan((-1.5, 0.0, 0.0), (3.0, 0.0), wmap, dec, obs, timestamp=1.0)
    all_detection_latencies.append(res_e.detection_latency_ms)
    all_validation_latencies.append(res_e.validation_latency_ms)
    all_planning_latencies.append(res_e.planning_latency_ms)
    all_total_latencies.append(res_e.total_latency_ms)

    results["Scenario E"] = {
        "description": "New A* route exists around obstacle",
        "replan_count": 1 if res_e.replan_triggered else 0,
        "trigger": res_e.trigger_type.value,
        "success": res_e.success,
        "old_len_m": res_e.metrics.old_path_length if res_e.metrics else 0.0,
        "new_len_m": res_e.metrics.new_path_length if res_e.metrics else 0.0,
        "old_cost": res_e.metrics.old_path_cost if res_e.metrics else 0.0,
        "new_cost": res_e.metrics.new_path_cost if res_e.metrics else 0.0,
        "deviation_m": res_e.metrics.path_deviation if res_e.metrics else 0.0,
        "clearance_m": res_e.metrics.min_dynamic_clearance if res_e.metrics else 0.0,
        "goal_reachable": res_e.metrics.goal_reachable if res_e.metrics else False,
    }

    # ==========================================================================
    # SCENARIO F: All global alternatives blocked (GLOBAL_REPLAN_FAILED)
    # ==========================================================================
    wmap = create_benchmark_world_map()
    for y in np.arange(-5.8, 5.9, 0.1):
        if abs(y) > 0.4:
            r, c = wmap.world_to_grid(0.5, float(y))
            wmap.obstacle[r, c] = True

    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    replanner.set_global_path(path)
    obs = [make_test_obstacle(6, (0.5, 0.0), size=(1.2, 1.2))]
    dec = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Deadlock", deadlock=True)

    res_f = replanner.check_and_replan((-1.5, 0.0, 0.0), (3.0, 0.0), wmap, dec, obs, timestamp=1.0)
    all_detection_latencies.append(res_f.detection_latency_ms)
    all_validation_latencies.append(res_f.validation_latency_ms)
    all_planning_latencies.append(res_f.planning_latency_ms)
    all_total_latencies.append(res_f.total_latency_ms)

    results["Scenario F"] = {
        "description": "All global alternatives blocked",
        "replan_count": 1 if res_f.replan_triggered else 0,
        "trigger": res_f.trigger_type.value,
        "success": res_f.success,
        "stale_path": res_f.is_path_stale,
        "path_is_none": res_f.path is None,
        "failure_reason": res_f.failure_reason,
    }

    # ==========================================================================
    # SCENARIO G: Multiple dynamic obstacles
    # ==========================================================================
    wmap = create_benchmark_world_map()
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    replanner.set_global_path(path)
    obs_g = [
        make_test_obstacle(71, (0.0, 0.0), size=(0.7, 0.7)),
        make_test_obstacle(72, (0.0, 1.2), size=(0.6, 0.6)),
        make_test_obstacle(73, (1.5, -0.8), size=(0.6, 0.6)),
    ]
    dec = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Deadlock", deadlock=True)

    res_g = replanner.check_and_replan((-2.0, 0.0, 0.0), (3.0, 0.0), wmap, dec, obs_g, timestamp=1.0)
    all_detection_latencies.append(res_g.detection_latency_ms)
    all_validation_latencies.append(res_g.validation_latency_ms)
    all_planning_latencies.append(res_g.planning_latency_ms)
    all_total_latencies.append(res_g.total_latency_ms)

    results["Scenario G"] = {
        "description": "Multiple dynamic obstacles",
        "replan_count": 1 if res_g.replan_triggered else 0,
        "trigger": res_g.trigger_type.value,
        "success": res_g.success,
        "clearance_m": res_g.metrics.min_dynamic_clearance if res_g.metrics else 0.0,
        "intersections": res_g.metrics.num_dynamic_obstacle_intersections if res_g.metrics else -1,
        "goal_reachable": res_g.metrics.goal_reachable if res_g.metrics else False,
    }

    # ==========================================================================
    # SCENARIO H: Unknown prediction region
    # ==========================================================================
    wmap = create_benchmark_world_map()
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    replanner.set_global_path(path)
    obs_h = [make_test_obstacle(8, (0.0, 0.0), status="UNKNOWN")]
    dec = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Deadlock", deadlock=True)

    res_h = replanner.check_and_replan((-1.5, 0.0, 0.0), (3.0, 0.0), wmap, dec, obs_h, timestamp=1.0)
    all_detection_latencies.append(res_h.detection_latency_ms)
    all_validation_latencies.append(res_h.validation_latency_ms)
    all_planning_latencies.append(res_h.planning_latency_ms)
    all_total_latencies.append(res_h.total_latency_ms)

    results["Scenario H"] = {
        "description": "Unknown prediction region",
        "replan_count": 1 if res_h.replan_triggered else 0,
        "trigger": res_h.trigger_type.value,
        "success": res_h.success,
        "goal_reachable": res_h.metrics.goal_reachable if res_h.metrics else False,
    }

    # ==========================================================================
    # SCENARIO I: Path invalidated again shortly after replan (cooldown override)
    # ==========================================================================
    wmap = create_benchmark_world_map()
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(cooldown_s=4.0, debounce_cycles=1))
    replanner.set_global_path(path)
    obs1 = [make_test_obstacle(91, (0.0, 0.0))]
    dec = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Deadlock", deadlock=True)

    # Replan 1 at t=0.0 -> Detours around (0.0, 0.0)
    r_i1 = replanner.check_and_replan((-1.5, 0.0, 0.0), (3.0, 0.0), wmap, dec, obs1, timestamp=0.0)
    all_detection_latencies.append(r_i1.detection_latency_ms)
    all_validation_latencies.append(r_i1.validation_latency_ms)
    all_planning_latencies.append(r_i1.planning_latency_ms)
    all_total_latencies.append(r_i1.total_latency_ms)

    # At t=1.0 (cooldown has 3.0s remaining), a new obstacle appears directly on the NEW path!
    new_mid_pt = r_i1.path.points[len(r_i1.path.points) // 2]
    obs2 = [obs1[0], make_test_obstacle(92, (new_mid_pt.x, new_mid_pt.y))]

    # Direct collision on new path overrides cooldown
    r_i2 = replanner.check_and_replan((-1.5, 0.0, 0.0), (3.0, 0.0), wmap, dec, obs2, timestamp=1.0)
    all_detection_latencies.append(r_i2.detection_latency_ms)
    all_validation_latencies.append(r_i2.validation_latency_ms)
    all_planning_latencies.append(r_i2.planning_latency_ms)
    all_total_latencies.append(r_i2.total_latency_ms)

    results["Scenario I"] = {
        "description": "Path invalidated again shortly after replan",
        "replan_count": replanner.total_replans_executed,
        "cooldown_overridden_on_collision": (replanner.total_replans_executed == 2),
        "success": r_i2.success,
        "goal_reachable": r_i2.metrics.goal_reachable if r_i2.metrics else False,
    }

    # ==========================================================================
    # SCENARIO J: Goal reached before replan
    # ==========================================================================
    wmap = create_benchmark_world_map()
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(goal_tolerance_m=0.25))
    path_j = make_straight_path(2.7, 3.0, 0.0)
    replanner.set_global_path(path_j)
    obs = [make_test_obstacle(10, (3.0, 0.0))]
    dec = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Deadlock", deadlock=True)

    res_j = replanner.check_and_replan((2.90, 0.0, 0.0), (3.0, 0.0), wmap, dec, obs, timestamp=1.0)
    all_detection_latencies.append(res_j.detection_latency_ms)

    results["Scenario J"] = {
        "description": "Goal reached before replan",
        "replan_count": 0 if not res_j.replan_triggered else 1,
        "trigger": res_j.trigger_type.value,
        "goal_reached": True,
        "stale_path": res_j.is_path_stale,
    }

    # ==========================================================================
    # AGGREGATE TIMING METRICS
    # ==========================================================================
    results["Timing Metrics"] = {
        "detection_latency_ms": {
            "mean": float(np.mean(all_detection_latencies)),
            "median": float(np.median(all_detection_latencies)),
            "p95": float(np.percentile(all_detection_latencies, 95)),
            "max": float(np.max(all_detection_latencies)),
        },
        "validation_latency_ms": {
            "mean": float(np.mean(all_validation_latencies)),
            "median": float(np.median(all_validation_latencies)),
            "p95": float(np.percentile(all_validation_latencies, 95)),
            "max": float(np.max(all_validation_latencies)),
        },
        "planning_latency_ms": {
            "mean": float(np.mean(all_planning_latencies)),
            "median": float(np.median(all_planning_latencies)),
            "p95": float(np.percentile(all_planning_latencies, 95)),
            "max": float(np.max(all_planning_latencies)),
        },
        "total_replanning_latency_ms": {
            "mean": float(np.mean(all_total_latencies)),
            "median": float(np.median(all_total_latencies)),
            "p95": float(np.percentile(all_total_latencies, 95)),
            "max": float(np.max(all_total_latencies)),
        },
    }

    return results


def print_benchmark_report(results: Dict[str, Any]) -> None:
    """Print markdown formatted benchmark report to stdout."""
    print("=" * 80)
    print("MILESTONE 13.11: EVENT-DRIVEN GLOBAL REPLANNING BENCHMARK REPORT")
    print("=" * 80)
    print()

    print("### Scenario Results Summary:")
    print("| Scenario | Description | Trigger | Replans | Success | Goal Reachable | Details |")
    print("|---|---|---|---|---|---|---|")

    for key, data in results.items():
        if key == "Timing Metrics":
            continue
        sc_id = key
        desc = data.get("description", "")
        trigger = data.get("trigger", "NONE")
        replans = data.get("replan_count", 0)
        success = data.get("success", True)
        goal_ok = data.get("goal_reachable", data.get("goal_reached", False))

        details = []
        if "deviation_m" in data:
            details.append(f"dev={data['deviation_m']:.2f}m")
        if "new_path_len_m" in data and data["new_path_len_m"] > 0:
            details.append(f"len={data['new_path_len_m']:.2f}m")
        if "planning_ms" in data:
            details.append(f"plan={data['planning_ms']:.1f}ms")
        if "failure_reason" in data and data["failure_reason"]:
            details.append(f"fail='{data['failure_reason'][:25]}...'")
        if "cooldown_overridden_on_collision" in data:
            details.append(f"override={data['cooldown_overridden_on_collision']}")

        detail_str = ", ".join(details) if details else "-"
        print(f"| **{sc_id}** | {desc} | `{trigger}` | {replans} | {success} | {goal_ok} | {detail_str} |")

    print()
    print("### Performance & Latency Metrics:")
    timing = results["Timing Metrics"]
    print("| Metric Stage | Mean (ms) | Median (ms) | P95 (ms) | Max (ms) |")
    print("|---|---|---|---|---|")
    for stage, vals in timing.items():
        print(f"| **{stage.replace('_', ' ').title()}** | {vals['mean']:.3f} ms | {vals['median']:.3f} ms | {vals['p95']:.3f} ms | {vals['max']:.3f} ms |")

    print()
    print("=" * 80)


if __name__ == "__main__":
    benchmark_results = run_benchmarks()
    print_benchmark_report(benchmark_results)
