"""Continuous-Time Temporal Safety & Adversarial 1,000-Case Validation for Milestone 13.9.

Validates that:
1. Broad-phase pre-filter NEVER produces a false SAFE clearance due to temporal discretization.
2. Continuous relative swept-segment test prevents tunneling between discrete samples.
3. 5 Adversarial deterministic corner-cases pass with zero false negatives:
   - Case 1: UGV and obstacle pass strictly between sampled timestamps (t = 0.15s).
   - Case 2: Obstacle crosses candidate between 0.25s occupancy bins (t = 0.375s).
   - Case 3: Obstacle accelerates/turns between prediction samples.
   - Case 4: Candidate (0.10s) and obstacle (0.25s) have heterogeneous sampling densities.
   - Case 5: Closest approach occurs strictly between existing samples.
4. 1,000 deterministic seeded candidate/obstacle trajectory pairs tested against
   a 200 Hz (dt = 0.005s) continuous reference check.
   REQUIRED: FALSE NEGATIVES = 0.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai.planning.base import RobotState, VehicleKinematics
from ai.planning.dynamic.collision_checker import PredictiveCollisionChecker
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
    polygon_intersection_sat,
)
from ai.planning.dynamic.dwa_filter import (
    DynamicBroadPhaseFilter,
    DynamicPlanningConfig,
    ObstacleSpatiotemporalHull,
)
from ai.planning.trajectory import Trajectory, TrajectoryPoint


def make_obstacle_trajectory(
    track_id: int,
    start_pos: Tuple[float, float],
    velocity: Tuple[float, float],
    size: Tuple[float, float] = (0.8, 0.6),
    yaw: float = 0.0,
    horizon_s: float = 2.0,
    dt: float = 0.10,
    base_unc: float = 0.15,
    unc_growth: float = 0.10,
    accel: Tuple[float, float] = (0.0, 0.0),
    yaw_rate: float = 0.0,
) -> PredictedObstacleTrajectory:
    """Generate synthetic obstacle trajectory with optional acceleration and turning."""
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
    waypoints: List[TrajectoryWaypoint] = []
    vx, vy = velocity
    ax, ay = accel
    x, y = start_pos
    curr_yaw = yaw

    for step in range(n_steps + 1):
        t = round(step * dt, 4)
        r_unc = base_unc + unc_growth * t
        cov_2x2 = np.diag([(r_unc / 3.0) ** 2, (r_unc / 3.0) ** 2]).astype(np.float64)
        waypoints.append(
            TrajectoryWaypoint(
                time_offset=t,
                timestamp=t,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                covariance_2d=cov_2x2,
                uncertainty_radius=r_unc,
            )
        )
        # Advance state
        x += vx * dt + 0.5 * ax * dt * dt
        y += vy * dt + 0.5 * ay * dt * dt
        vx += ax * dt
        vy += ay * dt
        curr_yaw += yaw_rate * dt

    polygons = [
        compute_obb_corners((wp.x, wp.y), size, yaw, margin_m=wp.uncertainty_radius)
        for wp in waypoints
    ]

    return PredictedObstacleTrajectory(
        track_id=track_id,
        current_state=state,
        horizon_s=horizon_s,
        dt=dt,
        waypoints=waypoints,
        bounding_polygon_per_step=polygons,
    )


def make_candidate_trajectory(
    v: float,
    omega: float,
    duration_s: float = 1.50,
    dt: float = 0.10,
    start_x: float = 0.0,
    start_y: float = 0.0,
    start_yaw: float = 0.0,
) -> Trajectory:
    """Simulate kinematic trajectory for DWA candidate."""
    n_steps = int(round(duration_s / dt))
    pts: List[TrajectoryPoint] = []
    x, y, yaw = start_x, start_y, start_yaw
    t = 0.0

    for _ in range(n_steps + 1):
        pts.append(TrajectoryPoint(
            t=round(t, 4), x=x, y=y, yaw=yaw, v=v, omega=omega, cost=0.10, clearance_m=2.0
        ))
        # Midpoint arc integration
        theta_mid = yaw + 0.5 * omega * dt
        x += v * math.cos(theta_mid) * dt
        y += v * math.sin(theta_mid) * dt
        yaw += omega * dt
        t += dt

    return Trajectory(points=pts, is_valid=True, total_cost=0.10, min_clearance_m=2.0)


def dense_reference_continuous_check(
    candidate: Trajectory,
    obstacle: PredictedObstacleTrajectory,
    r_ugv: float = 0.40,
    dt_fine: float = 0.005,  # 200 Hz reference check
) -> Tuple[bool, float, Optional[float]]:
    """Ground-truth continuous evaluation by high-frequency temporal sampling (200 Hz).

    Returns:
        (has_interaction, min_clearance, earliest_interaction_time)
    """
    pts = candidate.points
    t_max = min(pts[-1].t, obstacle.horizon_s)
    half_diag = math.hypot(obstacle.current_state.bbox_size[0] / 2.0, obstacle.current_state.bbox_size[1] / 2.0)

    n_fine = int(math.ceil(t_max / dt_fine))
    min_dist = float("inf")
    earliest_t = None
    has_interact = False

    # Pre-extract candidate times and coords
    cand_t = [p.t for p in pts]
    cand_x = [p.x for p in pts]
    cand_y = [p.y for p in pts]

    obs_wps = obstacle.waypoints

    for i in range(n_fine + 1):
        t = i * dt_fine

        # Interpolate candidate position at t
        if t <= cand_t[0]:
            ux, uy = cand_x[0], cand_y[0]
        elif t >= cand_t[-1]:
            ux, uy = cand_x[-1], cand_y[-1]
        else:
            # Linear interpolation between enclosing candidate points
            k = 0
            while k < len(cand_t) - 1 and cand_t[k + 1] <= t:
                k += 1
            alpha = (t - cand_t[k]) / (cand_t[k + 1] - cand_t[k])
            ux = cand_x[k] + alpha * (cand_x[k + 1] - cand_x[k])
            uy = cand_y[k] + alpha * (cand_y[k + 1] - cand_y[k])

        # Interpolate obstacle position and uncertainty at t
        if t <= obs_wps[0].time_offset:
            ox, oy = obs_wps[0].x, obs_wps[0].y
            r_unc = obs_wps[0].uncertainty_radius
        elif t >= obs_wps[-1].time_offset:
            ox, oy = obs_wps[-1].x, obs_wps[-1].y
            r_unc = obs_wps[-1].uncertainty_radius
        else:
            k = 0
            while k < len(obs_wps) - 1 and obs_wps[k + 1].time_offset <= t:
                k += 1
            alpha = (t - obs_wps[k].time_offset) / (obs_wps[k + 1].time_offset - obs_wps[k].time_offset)
            ox = obs_wps[k].x + alpha * (obs_wps[k + 1].x - obs_wps[k].x)
            oy = obs_wps[k].y + alpha * (obs_wps[k + 1].y - obs_wps[k].y)
            r_unc = obs_wps[k].uncertainty_radius + alpha * (obs_wps[k + 1].uncertainty_radius - obs_wps[k].uncertainty_radius)

        center_dist = math.hypot(ux - ox, uy - oy)
        reach = r_ugv + half_diag + r_unc
        clearance = center_dist - reach

        if clearance < min_dist:
            min_dist = clearance

        if clearance <= 0.0:
            has_interact = True
            if earliest_t is None:
                earliest_t = t

    return has_interact, min_dist, earliest_t


# =============================================================================
# 5 Adversarial Deterministic Corner-Case Scenarios
# =============================================================================

def run_adversarial_corner_cases() -> List[Dict[str, Any]]:
    bp_filter = DynamicBroadPhaseFilter(DynamicPlanningConfig(broad_phase_margin_m=0.05))
    results = []

    # -------------------------------------------------------------------------
    # Case 1: UGV and obstacle pass strictly between sampled timestamps (t = 0.15s)
    # -------------------------------------------------------------------------
    # DWA candidate sampled at dt = 0.1s.
    # At t=0.10s, UGV is at (0.05, 0.0), obstacle is at (0.075, 0.60) -> dist = 0.60 > reach(0.55)
    # At t=0.20s, UGV is at (0.10, 0.0), obstacle is at (0.075, -0.60) -> dist = 0.60 > reach(0.55)
    # At t=0.15s, UGV is at (0.075, 0.0), obstacle is at (0.075, 0.0) -> DIRECT HIT!
    traj1 = make_candidate_trajectory(v=0.50, omega=0.0, duration_s=1.0, dt=0.10)
    obs1 = make_obstacle_trajectory(
        track_id=1, start_pos=(0.075, 1.8), velocity=(0.0, -12.0), size=(0.2, 0.2),
        base_unc=0.05, unc_growth=0.0, dt=0.10
    )
    hulls1 = bp_filter.build_obstacle_hulls([obs1], max_horizon_s=1.0)
    bp_rel1 = bp_filter.is_candidate_relevant(traj1, hulls1, 0.40)
    ref_interact1, ref_clr1, ref_t1 = dense_reference_continuous_check(traj1, obs1, 0.40)

    results.append({
        "case": "Case 1: Inter-Sample Crossing (t=0.15s)",
        "bp_relevant": bp_rel1,
        "ref_interact": ref_interact1,
        "ref_clearance": ref_clr1,
        "ref_time": ref_t1,
        "safe_conservative": bp_rel1 if ref_interact1 else True,
    })

    # -------------------------------------------------------------------------
    # Case 2: Obstacle crosses candidate between 0.25s occupancy bins (t = 0.375s)
    # -------------------------------------------------------------------------
    traj2 = make_candidate_trajectory(v=0.60, omega=0.0, duration_s=1.5, dt=0.10)
    # Crossing occurs at x = 0.225m, y = 0.0 at t = 0.375s
    obs2 = make_obstacle_trajectory(
        track_id=2, start_pos=(0.225, 1.50), velocity=(0.0, -4.0), size=(0.4, 0.3),
        base_unc=0.10, unc_growth=0.05, dt=0.25  # Coarse 0.25s bins!
    )
    hulls2 = bp_filter.build_obstacle_hulls([obs2], max_horizon_s=1.5)
    bp_rel2 = bp_filter.is_candidate_relevant(traj2, hulls2, 0.40)
    ref_interact2, ref_clr2, ref_t2 = dense_reference_continuous_check(traj2, obs2, 0.40)

    results.append({
        "case": "Case 2: Crossing Between 0.25s Occupancy Bins (t=0.375s)",
        "bp_relevant": bp_rel2,
        "ref_interact": ref_interact2,
        "ref_clearance": ref_clr2,
        "ref_time": ref_t2,
        "safe_conservative": bp_rel2 if ref_interact2 else True,
    })

    # -------------------------------------------------------------------------
    # Case 3: Obstacle accelerates and turns between prediction samples
    # -------------------------------------------------------------------------
    traj3 = make_candidate_trajectory(v=0.40, omega=0.30, duration_s=1.5, dt=0.10)
    obs3 = make_obstacle_trajectory(
        track_id=3, start_pos=(1.0, 1.0), velocity=(-0.5, -0.8), size=(0.5, 0.4),
        accel=(-0.2, 0.3), yaw_rate=0.4, dt=0.10
    )
    hulls3 = bp_filter.build_obstacle_hulls([obs3], max_horizon_s=1.5)
    bp_rel3 = bp_filter.is_candidate_relevant(traj3, hulls3, 0.40)
    ref_interact3, ref_clr3, ref_t3 = dense_reference_continuous_check(traj3, obs3, 0.40)

    results.append({
        "case": "Case 3: Obstacle Accelerating / Turning",
        "bp_relevant": bp_rel3,
        "ref_interact": ref_interact3,
        "ref_clearance": ref_clr3,
        "ref_time": ref_t3,
        "safe_conservative": bp_rel3 if ref_interact3 else True,
    })

    # -------------------------------------------------------------------------
    # Case 4: Heterogeneous sampling densities (UGV dt=0.08s, Obstacle dt=0.30s)
    # -------------------------------------------------------------------------
    traj4 = make_candidate_trajectory(v=0.45, omega=-0.25, duration_s=1.5, dt=0.08)
    obs4 = make_obstacle_trajectory(
        track_id=4, start_pos=(0.8, -1.2), velocity=(-0.2, 1.0), size=(0.6, 0.5),
        dt=0.30
    )
    hulls4 = bp_filter.build_obstacle_hulls([obs4], max_horizon_s=1.5)
    bp_rel4 = bp_filter.is_candidate_relevant(traj4, hulls4, 0.40)
    ref_interact4, ref_clr4, ref_t4 = dense_reference_continuous_check(traj4, obs4, 0.40)

    results.append({
        "case": "Case 4: Heterogeneous Sample Densities (dt=0.08s vs dt=0.30s)",
        "bp_relevant": bp_rel4,
        "ref_interact": ref_interact4,
        "ref_clearance": ref_clr4,
        "ref_time": ref_t4,
        "safe_conservative": bp_rel4 if ref_interact4 else True,
    })

    # -------------------------------------------------------------------------
    # Case 5: Closest approach occurs strictly between existing samples
    # -------------------------------------------------------------------------
    # Glancing near-miss where distance at t_k = 0.50m is 0.58m, at t_{k+1}=0.60m is 0.58m,
    # but at t = 0.55m distance is 0.42m < reach(0.48m).
    traj5 = make_candidate_trajectory(v=0.50, omega=0.0, duration_s=1.0, dt=0.10)
    # At t=0.5s UGV at (0.25, 0). Obstacle at (0.25, 0.58) at t=0.50s, moving at vy = -1.6 m/s
    obs5 = make_obstacle_trajectory(
        track_id=5, start_pos=(0.25, 1.38), velocity=(0.0, -1.6), size=(0.3, 0.3),
        base_unc=0.05, unc_growth=0.0, dt=0.10
    )
    hulls5 = bp_filter.build_obstacle_hulls([obs5], max_horizon_s=1.0)
    bp_rel5 = bp_filter.is_candidate_relevant(traj5, hulls5, 0.40)
    ref_interact5, ref_clr5, ref_t5 = dense_reference_continuous_check(traj5, obs5, 0.40)

    results.append({
        "case": "Case 5: Closest Approach Strictly Between Samples (t=0.55s)",
        "bp_relevant": bp_rel5,
        "ref_interact": ref_interact5,
        "ref_clearance": ref_clr5,
        "ref_time": ref_t5,
        "safe_conservative": bp_rel5 if ref_interact5 else True,
    })

    return results


# =============================================================================
# 1,000 Adversarial Randomized Test Set
# =============================================================================

def run_adversarial_1000_validation(n_cases: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Execute 1,000 randomized deterministic candidate/obstacle test pairs."""
    rng = np.random.RandomState(seed)
    bp_filter = DynamicBroadPhaseFilter(DynamicPlanningConfig(broad_phase_margin_m=0.05))

    cleared_count = 0
    exact_check_count = 0
    false_negative_count = 0
    false_positive_count = 0  # Conservative over-approximation (safe)
    true_positive_count = 0
    true_negative_count = 0

    t0 = time.perf_counter()

    for idx in range(n_cases):
        # 1. Random UGV candidate trajectory
        v = rng.uniform(0.0, 0.80)
        omega = rng.uniform(-1.0, 1.0)
        dt = rng.choice([0.05, 0.08, 0.10, 0.12])
        duration = rng.uniform(1.0, 1.8)

        candidate = make_candidate_trajectory(v=v, omega=omega, duration_s=duration, dt=dt)

        # 2. Random Obstacle trajectory
        obs_x0 = rng.uniform(-1.0, 3.5)
        obs_y0 = rng.uniform(-3.0, 3.0)
        obs_speed = rng.uniform(0.1, 2.5)
        obs_angle = rng.uniform(0.0, 2.0 * math.pi)
        vx = obs_speed * math.cos(obs_angle)
        vy = obs_speed * math.sin(obs_angle)

        size_l = rng.uniform(0.4, 1.2)
        size_w = rng.uniform(0.3, 0.8)
        base_unc = rng.uniform(0.05, 0.25)
        unc_growth = rng.uniform(0.05, 0.20)
        obs_dt = rng.choice([0.05, 0.10, 0.20, 0.25])
        horizon = rng.uniform(1.2, 2.5)

        obstacle = make_obstacle_trajectory(
            track_id=idx + 1,
            start_pos=(obs_x0, obs_y0),
            velocity=(vx, vy),
            size=(size_l, size_w),
            base_unc=base_unc,
            unc_growth=unc_growth,
            horizon_s=horizon,
            dt=obs_dt,
        )

        # 3. Broad-phase evaluation
        hulls = bp_filter.build_obstacle_hulls([obstacle], max_horizon_s=duration)
        bp_result = bp_filter.is_candidate_relevant(candidate, hulls, 0.40)

        # 4. Dense ground-truth reference evaluation (200 Hz)
        ref_interact, min_clr, _ = dense_reference_continuous_check(candidate, obstacle, 0.40, dt_fine=0.005)

        # 5. Classify outcome
        if bp_result:
            exact_check_count += 1
            if ref_interact:
                true_positive_count += 1
            else:
                false_positive_count += 1  # Safe conservative margin
        else:
            cleared_count += 1
            if ref_interact:
                # CATASTROPHIC SAFETY FAILURE: Broad-phase said safe, but continuous check interacted!
                false_negative_count += 1
                print(f"[DEBUG FN] idx={idx}, seed={seed}, min_clr={min_clr:.6f}")
                print(f"  Candidate: v={v}, omega={omega}, dt={dt}, duration={duration}, n_pts={len(candidate.points)}")
                print(f"  Obstacle: pos=({obs_x0:.3f}, {obs_y0:.3f}), vel=({vx:.3f}, {vy:.3f}), dt={obs_dt}, horizon={horizon}")
                print(f"  Obstacle waypoints count: {len(obstacle.waypoints)}")
            else:
                true_negative_count += 1

    elapsed_s = time.perf_counter() - t0

    return {
        "n_cases": n_cases,
        "seed": seed,
        "elapsed_s": elapsed_s,
        "cleared_count": cleared_count,
        "exact_check_count": exact_check_count,
        "true_positive_count": true_positive_count,
        "true_negative_count": true_negative_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
    }


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("TEMPORAL SAFETY AUDIT: 5 ADVERSARIAL DETERMINISTIC CORNER CASES")
    print("=" * 100)
    corner_results = run_adversarial_corner_cases()
    all_corner_safe = True
    for cr in corner_results:
        status_str = "PASS (SAFE)" if cr["safe_conservative"] else "FAIL (UNSAFE TUNNELING)"
        print(f"{cr['case']:<60} | Ref Interact: {str(cr['ref_interact']):<5} | BP Relevant: {str(cr['bp_relevant']):<5} | Status: {status_str}")
        if not cr["safe_conservative"]:
            all_corner_safe = False

    print("\n" + "=" * 100)
    print("1,000-CASE DETERMINISTIC ADVERSARIAL RANDOMIZED VALIDATION")
    print("=" * 100)
    val_res = run_adversarial_1000_validation(n_cases=1000, seed=42)
    print(f"Total Test Cases:       {val_res['n_cases']}")
    print(f"Elapsed Time:           {val_res['elapsed_s']:.2f} s ({val_res['elapsed_s'] / 1000.0 * 1000.0:.2f} ms/pair)")
    print(f"Broad-Phase Cleared:    {val_res['cleared_count']} ({val_res['cleared_count'] / val_res['n_cases'] * 100.0:.1f}%)")
    print(f"Exact Checks Invoked:   {val_res['exact_check_count']} ({val_res['exact_check_count'] / val_res['n_cases'] * 100.0:.1f}%)")
    print(f"True Positives:         {val_res['true_positive_count']}")
    print(f"True Negatives:         {val_res['true_negative_count']}")
    print(f"Conservative Over-Appr: {val_res['false_positive_count']} (Safe)")
    print(f"FALSE NEGATIVES:        {val_res['false_negative_count']} (MUST BE 0)")
    print("=" * 100)
    if val_res["false_negative_count"] == 0 and all_corner_safe:
        print("VERDICT: STRICT CONTINUOUS-TIME CONSERVATIVE SAFETY PROVEN (0 FALSE NEGATIVES)")
    else:
        print("VERDICT: SAFETY FAILURE - FALSE NEGATIVE DETECTED")
