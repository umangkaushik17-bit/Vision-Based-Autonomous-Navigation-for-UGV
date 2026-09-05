"""Comprehensive evaluation and benchmarking suite for Phase 13 Milestone 13.6 — Motion Prediction.

Evaluates:
1. Deterministic synthetic evaluation trajectories (A through J):
   A. Stationary
   B. Constant velocity
   C. Diagonal constant velocity
   D. Positive acceleration
   E. Deceleration (braking)
   F. Direction change (turning)
   G. Noisy filtered state
   H. High-uncertainty state
   I. Low-confidence/degraded track
   J. Lost track
2. Quantitative prediction metrics at horizons [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00] s:
   - x RMSE, y RMSE, Euclidean RMSE, MAE, Max error
   - Ground-truth coverage by uncertainty envelope
   - Covariance trace vs horizon
   - Uncertainty radius vs horizon
3. Uncertainty calibration check (empirical error vs predicted uncertainty envelope).
4. Model mismatch analysis (accelerating, braking, turning targets).
5. Multi-track performance profiling (1, 2, 4, 10 tracks) and combined Kalman + prediction latency.
6. Generation of validation figures.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ai.planning.dynamic.data_model import (
    TrackedObstacleArray,
    TrackedObstacleState,
    compute_obb_corners,
    point_in_polygon,
)
from ai.planning.dynamic.kalman_filter import KalmanFilterConfig, KalmanTrackFilter
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig


HORIZONS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]


def run_synthetic_scenarios() -> Dict[str, Any]:
    """Execute evaluation across the 10 defined synthetic scenarios."""
    predictor = MotionPredictor(PredictorConfig(process_noise_spectral_density=1.0, sigma_scale=3.0))

    # Define scenarios: each returns (track_state, ground_truth_fn(dt) -> (x, y))
    scenarios: Dict[str, Tuple[TrackedObstacleState, Any]] = {}

    # A. Stationary
    cov_nom = np.diag([0.04, 0.04, 0.09, 0.09]).astype(np.float64)
    scenarios["A_Stationary"] = (
        TrackedObstacleState(
            track_id=1, timestamp=0.0, x=5.0, y=5.0, vx=0.0, vy=0.0,
            covariance=cov_nom.copy(), bbox_size=(1.5, 0.8), bbox_yaw=0.0, status="TRACKING",
        ),
        lambda dt: (5.0, 5.0),
    )

    # B. Constant Velocity (linear X)
    scenarios["B_ConstantVelocity"] = (
        TrackedObstacleState(
            track_id=2, timestamp=0.0, x=0.0, y=2.0, vx=2.0, vy=0.0,
            covariance=cov_nom.copy(), bbox_size=(1.5, 0.8), bbox_yaw=0.0, status="TRACKING",
        ),
        lambda dt: (2.0 * dt, 2.0),
    )

    # C. Diagonal Constant Velocity
    scenarios["C_DiagonalConstantVelocity"] = (
        TrackedObstacleState(
            track_id=3, timestamp=0.0, x=1.0, y=1.0, vx=1.5, vy=2.0,
            covariance=cov_nom.copy(), bbox_size=(1.5, 0.8), bbox_yaw=math.atan2(2.0, 1.5), status="TRACKING",
        ),
        lambda dt: (1.0 + 1.5 * dt, 1.0 + 2.0 * dt),
    )

    # D. Positive Acceleration (ax = 1.5 m/s^2)
    ax_d = 1.5
    scenarios["D_PositiveAcceleration"] = (
        TrackedObstacleState(
            track_id=4, timestamp=0.0, x=0.0, y=0.0, vx=1.0, vy=0.0,
            covariance=cov_nom.copy(), bbox_size=(1.5, 0.8), bbox_yaw=0.0, status="TRACKING",
        ),
        lambda dt: (1.0 * dt + 0.5 * ax_d * (dt ** 2), 0.0),
    )

    # E. Deceleration / Braking (ax = -1.5 m/s^2)
    ax_e = -1.5
    scenarios["E_Deceleration"] = (
        TrackedObstacleState(
            track_id=5, timestamp=0.0, x=0.0, y=0.0, vx=3.0, vy=0.0,
            covariance=cov_nom.copy(), bbox_size=(1.5, 0.8), bbox_yaw=0.0, status="TRACKING",
        ),
        lambda dt: (3.0 * dt + 0.5 * ax_e * (dt ** 2), 0.0),
    )

    # F. Direction Change / Turning (circular arc R=5m, omega=0.8 rad/s, v=4m/s)
    R_f = 5.0
    omega_f = 0.8
    scenarios["F_DirectionChange"] = (
        TrackedObstacleState(
            track_id=6, timestamp=0.0, x=0.0, y=0.0, vx=4.0, vy=0.0,
            covariance=cov_nom.copy(), bbox_size=(1.5, 0.8), bbox_yaw=0.0, status="TRACKING",
        ),
        lambda dt: (R_f * math.sin(omega_f * dt), R_f * (1.0 - math.cos(omega_f * dt))),
    )

    # G. Noisy Filtered State (realistic Kalman output with off-diagonal cov)
    cov_noisy = np.array([
        [0.052, 0.011, 0.015, 0.003],
        [0.011, 0.048, 0.002, 0.014],
        [0.015, 0.002, 0.120, 0.018],
        [0.003, 0.014, 0.018, 0.115],
    ], dtype=np.float64)
    scenarios["G_NoisyFilteredState"] = (
        TrackedObstacleState(
            track_id=7, timestamp=0.0, x=3.0, y=-2.0, vx=1.8, vy=-0.5,
            covariance=cov_noisy, bbox_size=(1.5, 0.8), bbox_yaw=-0.27, status="TRACKING",
        ),
        lambda dt: (3.0 + 1.8 * dt, -2.0 - 0.5 * dt),
    )

    # H. High Uncertainty State
    cov_high = np.diag([0.25, 0.25, 0.64, 0.64]).astype(np.float64)
    scenarios["H_HighUncertaintyState"] = (
        TrackedObstacleState(
            track_id=8, timestamp=0.0, x=10.0, y=10.0, vx=0.5, vy=0.5,
            covariance=cov_high, bbox_size=(1.5, 0.8), bbox_yaw=0.0, status="TRACKING",
        ),
        lambda dt: (10.0 + 0.5 * dt, 10.0 + 0.5 * dt),
    )

    # I. Low-confidence / DEGRADED Track
    scenarios["I_LowConfidenceDegraded"] = (
        TrackedObstacleState(
            track_id=9, timestamp=0.0, x=0.0, y=5.0, vx=1.2, vy=0.0,
            covariance=cov_nom.copy(), bbox_size=(1.5, 0.8), bbox_yaw=0.0, status="DEGRADED",
        ),
        lambda dt: (1.2 * dt, 5.0),
    )

    # J. LOST Track (coast horizon capped at 1.0s)
    scenarios["J_LostTrack"] = (
        TrackedObstacleState(
            track_id=10, timestamp=0.0, x=2.0, y=2.0, vx=1.0, vy=1.0,
            covariance=cov_nom.copy(), bbox_size=(1.5, 0.8), bbox_yaw=0.0, status="LOST",
        ),
        lambda dt: (2.0 + 1.0 * dt, 2.0 + 1.0 * dt),
    )

    # Collect predictions and errors
    results: Dict[str, Any] = {"scenarios": {}, "horizon_stats": {h: [] for h in HORIZONS}}

    for name, (track, gt_fn) in scenarios.items():
        pred = predictor.predict_track(track, time_offsets_s=[0.0] + HORIZONS)
        scenario_data = {"track": track, "pred": pred, "errors": {}}

        for wp in pred.waypoints:
            h = wp.time_offset
            if h < 1e-6:
                continue
            gt_x, gt_y = gt_fn(h)
            err_x = wp.x - gt_x
            err_y = wp.y - gt_y
            euc_err = math.hypot(err_x, err_y)
            # Check coverage: true point inside polygon or within uncertainty radius
            poly = compute_obb_corners((wp.x, wp.y), track.bbox_size, track.bbox_yaw, margin_m=wp.uncertainty_radius)
            covered = point_in_polygon((gt_x, gt_y), poly) or (euc_err <= wp.uncertainty_radius + 0.5 * max(track.bbox_size))

            stat_point = {
                "scenario": name,
                "h": h,
                "err_x": err_x,
                "err_y": err_y,
                "euc_err": euc_err,
                "covered": covered,
                "r_unc": wp.uncertainty_radius,
                "trace_cov": float(np.trace(wp.covariance_2d)),
            }
            scenario_data["errors"][h] = stat_point
            if h in results["horizon_stats"]:
                results["horizon_stats"][h].append(stat_point)

        results["scenarios"][name] = scenario_data

    return results


def compute_aggregate_metrics(results: Dict[str, Any]) -> Dict[float, Dict[str, float]]:
    """Compute RMSE, MAE, Max Error, Coverage, Covariance Trace, and Uncertainty Radius per horizon."""
    metrics_per_horizon: Dict[float, Dict[str, float]] = {}

    for h in HORIZONS:
        stats = results["horizon_stats"][h]
        if not stats:
            continue
        errs_x = np.array([s["err_x"] for s in stats])
        errs_y = np.array([s["err_y"] for s in stats])
        euc_errs = np.array([s["euc_err"] for s in stats])
        coverages = [s["covered"] for s in stats]
        r_uncs = [s["r_unc"] for s in stats]
        traces = [s["trace_cov"] for s in stats]

        rmse_x = float(np.sqrt(np.mean(errs_x ** 2)))
        rmse_y = float(np.sqrt(np.mean(errs_y ** 2)))
        rmse_euc = float(np.sqrt(np.mean(euc_errs ** 2)))
        mae = float(np.mean(euc_errs))
        max_err = float(np.max(euc_errs))
        cov_pct = float(np.mean(coverages) * 100.0)
        mean_r_unc = float(np.mean(r_uncs))
        mean_trace = float(np.mean(traces))

        metrics_per_horizon[h] = {
            "rmse_x": rmse_x,
            "rmse_y": rmse_y,
            "rmse_euc": rmse_euc,
            "mae": mae,
            "max_err": max_err,
            "coverage_pct": cov_pct,
            "mean_r_unc": mean_r_unc,
            "mean_trace": mean_trace,
        }

    return metrics_per_horizon


def benchmark_predictor_latency() -> Dict[str, Any]:
    """Measure single-track and batch prediction latencies for 1, 2, 4, and 10 tracks."""
    predictor = MotionPredictor()
    track_counts = [1, 2, 4, 10]
    iterations = 500
    warmup_cycles = 50

    def make_tracks(n: int) -> List[TrackedObstacleState]:
        cov = np.diag([0.04, 0.04, 0.16, 0.16]).astype(np.float64)
        return [
            TrackedObstacleState(
                track_id=i, timestamp=10.0, x=float(i * 3), y=float(i * 2),
                vx=1.5, vy=-0.5, covariance=cov.copy(),
                bbox_size=(1.5, 0.8), bbox_yaw=0.2, status="TRACKING",
            )
            for i in range(n)
        ]

    # Warmup
    warmup_tracks = make_tracks(10)
    for _ in range(warmup_cycles):
        predictor.predict_tracks(warmup_tracks, horizon_s=2.0, dt_s=0.25)
        predictor.predict_track(warmup_tracks[0], horizon_s=2.0, dt_s=0.25)

    # 1. Single track prediction latency
    single_track = make_tracks(1)[0]
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        predictor.predict_track(single_track, horizon_s=2.0, dt_s=0.25)
    t1 = time.perf_counter_ns()
    single_track_lat_us = ((t1 - t0) / iterations) / 1000.0

    # 2. Batch track predictions
    batch_latencies_us: Dict[int, float] = {}
    for n in track_counts:
        tracks = make_tracks(n)
        arr = TrackedObstacleArray(header_timestamp=10.0, tracks=tracks)
        t0 = time.perf_counter_ns()
        for _ in range(iterations):
            predictor.predict_tracks(arr, horizon_s=2.0, dt_s=0.25)
        t1 = time.perf_counter_ns()
        batch_latencies_us[n] = ((t1 - t0) / iterations) / 1000.0

    # 3. Kalman filter predict + update latency
    kf_latencies_us: Dict[int, float] = {}
    for n in track_counts:
        filters = [KalmanTrackFilter(KalmanFilterConfig()) for _ in range(n)]
        for i, kf in enumerate(filters):
            kf.initialize((float(i), float(i)), timestamp_s=0.0)

        t0 = time.perf_counter_ns()
        for it in range(iterations):
            t_curr = (it + 1) * 0.1
            for kf in filters:
                kf.predict(t_curr)
                kf.update((1.0, 1.0), t_curr)
        t1 = time.perf_counter_ns()
        kf_latencies_us[n] = ((t1 - t0) / iterations) / 1000.0

    # 4. Combined Kalman + Prediction latency
    combined_latencies_us: Dict[int, float] = {}
    for n in track_counts:
        filters = [KalmanTrackFilter(KalmanFilterConfig()) for _ in range(n)]
        for i, kf in enumerate(filters):
            kf.initialize((float(i), float(i)), timestamp_s=0.0)

        t0 = time.perf_counter_ns()
        for it in range(iterations):
            t_curr = (it + 1) * 0.1
            states = []
            for i, kf in enumerate(filters):
                kf.predict(t_curr)
                kf.update((float(i) + t_curr, float(i) + 0.5 * t_curr), t_curr)
                # Build TrackedObstacleState from filtered state
                s = TrackedObstacleState(
                    track_id=i, timestamp=t_curr,
                    x=kf.position[0], y=kf.position[1],
                    vx=kf.velocity[0], vy=kf.velocity[1],
                    covariance=kf._P.copy(),
                    bbox_size=(1.5, 0.8), bbox_yaw=0.0,
                    status="TRACKING",
                )
                states.append(s)
            arr = TrackedObstacleArray(header_timestamp=t_curr, tracks=states)
            predictor.predict_tracks(arr, horizon_s=2.0, dt_s=0.25)
        t1 = time.perf_counter_ns()
        combined_latencies_us[n] = ((t1 - t0) / iterations) / 1000.0

    return {
        "iterations": iterations,
        "single_track_us": single_track_lat_us,
        "batch_latencies_us": batch_latencies_us,
        "kf_latencies_us": kf_latencies_us,
        "combined_latencies_us": combined_latencies_us,
    }


def generate_plots(results: Dict[str, Any], metrics: Dict[float, Dict[str, float]], perf: Dict[str, Any], output_path: str) -> None:
    """Generate the 4 required validation figures."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Plot 1: Trajectory with Prediction Envelopes (Turning target Scenario F vs Linear CV)
    ax1 = axes[0, 0]
    scen_f = results["scenarios"]["F_DirectionChange"]
    pred_f = scen_f["pred"]
    # Ground truth
    t_fine = np.linspace(0.0, 2.0, 100)
    R_f, omega_f = 5.0, 0.8
    gt_xs = [R_f * math.sin(omega_f * t) for t in t_fine]
    gt_ys = [R_f * (1.0 - math.cos(omega_f * t)) for t in t_fine]

    pred_xs = [wp.x for wp in pred_f.waypoints]
    pred_ys = [wp.y for wp in pred_f.waypoints]

    ax1.plot(gt_xs, gt_ys, "g-", linewidth=2.5, label="Ground Truth (Turning)")
    ax1.plot(pred_xs, pred_ys, "b--o", linewidth=2.0, label="CV Predicted Path")
    ax1.scatter([pred_f.current_state.x], [pred_f.current_state.y], color="black", s=90, zorder=5, label="Current Obstacle")

    # Plot bounding envelopes
    for i, poly in enumerate(pred_f.bounding_polygon_per_step):
        poly_closed = np.vstack([poly, poly[0]])
        alpha = 0.25 + 0.08 * i
        ax1.fill(poly_closed[:, 0], poly_closed[:, 1], color="blue", alpha=0.12)
        ax1.plot(poly_closed[:, 0], poly_closed[:, 1], "b:", alpha=0.6)

    ax1.set_title("1. Predicted Trajectory vs Ground Truth & Envelopes", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (m) [odom frame]")
    ax1.set_ylabel("Y (m) [odom frame]")
    ax1.axis("equal")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=9)

    # Plot 2: Euclidean Error vs Horizon across Scenarios (Model Mismatch)
    ax2 = axes[0, 1]
    horizons = sorted(list(metrics.keys()))
    scenarios_to_plot = [
        ("B_ConstantVelocity", "green", "-", "Constant Velocity (Matched)"),
        ("D_PositiveAcceleration", "orange", "-.", "Accelerating (+1.5 m/s^2)"),
        ("E_Deceleration", "purple", "--", "Decelerating (-1.5 m/s^2)"),
        ("F_DirectionChange", "red", "-", "Turning Target (Mismatch)"),
    ]
    for sc_name, col, ls, label in scenarios_to_plot:
        sc_errs = [results["scenarios"][sc_name]["errors"][h]["euc_err"] for h in horizons]
        ax2.plot(horizons, sc_errs, color=col, linestyle=ls, marker="o", label=label)

    ax2.set_title("2. Prediction Error vs Horizon (Model Mismatch)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Prediction Horizon (s)")
    ax2.set_ylabel("Euclidean Error (m)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9)

    # Plot 3: Uncertainty Envelope (Radius & Trace vs Horizon)
    ax3 = axes[1, 0]
    radii = [metrics[h]["mean_r_unc"] for h in horizons]
    traces = [metrics[h]["mean_trace"] for h in horizons]

    ax3_twin = ax3.twinx()
    l1 = ax3.plot(horizons, radii, "b-o", linewidth=2.0, label="Uncertainty Radius (m)")
    l2 = ax3_twin.plot(horizons, traces, "m--s", linewidth=2.0, label="Covariance Trace (m^2)")

    ax3.set_title("3. Uncertainty Envelope Expansion vs Horizon", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Prediction Horizon (s)")
    ax3.set_ylabel("Uncertainty Radius (m)", color="blue")
    ax3_twin.set_ylabel("Trace(P_pos) (m^2)", color="purple")
    ax3.grid(True, linestyle="--", alpha=0.5)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper left", fontsize=9)

    # Plot 4: Latency Scaling (Kalman vs Predictor vs Combined)
    ax4 = axes[1, 1]
    track_counts = [1, 2, 4, 10]
    pred_lat = [perf["batch_latencies_us"][n] for n in track_counts]
    kf_lat = [perf["kf_latencies_us"][n] for n in track_counts]
    comb_lat = [perf["combined_latencies_us"][n] for n in track_counts]

    ax4.plot(track_counts, pred_lat, "b-o", label="M13.6 Predictor Latency")
    ax4.plot(track_counts, kf_lat, "g--s", label="M13.5 Kalman Filter Latency")
    ax4.plot(track_counts, comb_lat, "r-^", label="Combined Filter + Predictor")

    ax4.set_title("4. Multi-Track Latency Scaling (us)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Number of Tracks")
    ax4.set_ylabel("Mean Execution Time (us)")
    ax4.set_xticks(track_counts)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Validation plot saved to: {output_path}")


def main() -> None:
    print("============================================================")
    print("STARTING MILESTONE 13.6 MOTION PREDICTION BENCHMARK")
    print("============================================================")

    res = run_synthetic_scenarios()
    metrics = compute_aggregate_metrics(res)
    perf = benchmark_predictor_latency()

    print("\n--- PREDICTION ACCURACY METRICS ACROSS HORIZONS ---")
    print("| Horizon | Position RMSE | Mean Error | Max Error | Uncertainty Radius | Coverage | Trace(P_pos) |")
    print("|---------|---------------|------------|-----------|--------------------|----------|--------------|")
    for h in HORIZONS:
        m = metrics[h]
        print(
            f"| {h:5.2f}s  | {m['rmse_euc']:13.4f} | {m['mae']:10.4f} | {m['max_err']:9.4f} | "
            f"{m['mean_r_unc']:18.4f} | {m['coverage_pct']:7.1f}% | {m['mean_trace']:12.4f} |"
        )

    print("\n--- MODEL MISMATCH & PER-SCENARIO BREAKDOWN AT 2.0s (1.0s for LOST) ---")
    print("| Scenario Class               | Scenario Name             | Status   | Horizon | Error (m) | r_unc (m) | Covered |")
    print("|------------------------------|---------------------------|----------|---------|-----------|-----------|---------|")
    scenario_classes = [
        ("Matched Model (Stationary)", "A_Stationary", 2.0),
        ("Matched Model (Linear CV)", "B_ConstantVelocity", 2.0),
        ("Matched Model (Diagonal CV)", "C_DiagonalConstantVelocity", 2.0),
        ("Model Mismatch (Acceleration)", "D_PositiveAcceleration", 2.0),
        ("Model Mismatch (Braking)", "E_Deceleration", 2.0),
        ("Model Mismatch (Turning)", "F_DirectionChange", 2.0),
        ("Noisy Filtered Covariance", "G_NoisyFilteredState", 2.0),
        ("High Initial Uncertainty", "H_HighUncertaintyState", 2.0),
        ("Lifecycle Adaptation (Degraded)", "I_LowConfidenceDegraded", 2.0),
        ("Lifecycle Adaptation (Lost)", "J_LostTrack", 1.0),
    ]
    for cat, sc, h_sc in scenario_classes:
        err = res["scenarios"][sc]["errors"][h_sc]
        st = res["scenarios"][sc]["track"].status
        print(
            f"| {cat:28s} | {sc:25s} | {st:8s} | {h_sc:4.1f}s   | {err['euc_err']:9.4f} | "
            f"{err['r_unc']:9.4f} | {str(err['covered']):7s} |"
        )

    print("\n--- UNCERTAINTY CALIBRATION STATISTICAL CHECK ---")
    total_checks = 0
    covered_checks = 0
    for h in HORIZONS:
        for s in res["horizon_stats"][h]:
            total_checks += 1
            if s["covered"]:
                covered_checks += 1
    cov_ratio = (covered_checks / total_checks) * 100.0
    print(f"Total spatiotemporal checks: {total_checks}, In-envelope: {covered_checks} ({cov_ratio:.1f}%)")
    if cov_ratio >= 95.0:
        print("Uncertainty Envelope Assessment: Approximately calibrated to conservatively bounding 3-sigma envelope.")
    else:
        print(f"Uncertainty Envelope Assessment: Honest measured coverage is {cov_ratio:.1f}%.")

    print("\n--- PERFORMANCE PROFILING ---")
    print(f"Single-track predict latency: {perf['single_track_us']:.2f} us")
    for n in [1, 2, 4, 10]:
        print(
            f"Batch {n:2d} tracks -> Predictor: {perf['batch_latencies_us'][n]:6.2f} us | "
            f"Kalman: {perf['kf_latencies_us'][n]:6.2f} us | "
            f"Combined: {perf['combined_latencies_us'][n]:6.2f} us"
        )

    plot_file = os.path.join("docs", "images", "milestone_13_6_prediction.png")
    generate_plots(res, metrics, perf, plot_file)
    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
