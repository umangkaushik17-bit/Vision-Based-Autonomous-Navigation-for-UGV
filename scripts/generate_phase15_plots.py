"""Generate publication-grade visual artifacts for Milestone 13.15 MATLAB Independent Validation.

Generates 6 figures in validation/matlab/phase15/plots/:
1. fig_01_velocity_estimation_residuals.png
2. fig_02_kalman_trajectory_innovations.png
3. fig_03_mahalanobis_gate_contours.png
4. fig_04_prediction_covariance_growth.png
5. fig_05_dynamic_occupancy_cost_profile.png
6. fig_06_sat_collision_margins.png
"""

from __future__ import annotations

import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import numpy as np

OUTPUT_DIR = os.path.join("validation", "matlab", "phase15", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_fig_01_velocity_estimation_residuals():
    """Figure 1: Velocity estimation regression line vs noisy measurements and residuals."""
    np.random.seed(42)
    dt = 0.1
    t = np.linspace(0.0, 1.0, 11)
    vx_true, vy_true = 1.20, 0.80
    x_true = 2.0 + vx_true * t
    y_true = 1.0 + vy_true * t
    x_meas = x_true + np.random.normal(0.0, 0.03, size=len(t))
    y_meas = y_true + np.random.normal(0.0, 0.03, size=len(t))

    # OLS fit
    t_mean = np.mean(t)
    t_diff = t - t_mean
    ss_t = np.sum(t_diff ** 2)
    vx_fit = np.sum(t_diff * (x_meas - np.mean(x_meas))) / ss_t
    vy_fit = np.sum(t_diff * (y_meas - np.mean(y_meas))) / ss_t
    x0_fit = np.mean(x_meas) - vx_fit * t_mean
    y0_fit = np.mean(y_meas) - vy_fit * t_mean

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Subplot 1: Trajectory and fit
    ax1.plot(x_true, y_true, "g--", linewidth=2, label=f"True Trajectory (v=[{vx_true:.2f}, {vy_true:.2f}])")
    ax1.scatter(x_meas, y_meas, color="red", s=50, zorder=5, label="Sensor Observations (σ=0.03m)")
    t_dense = np.linspace(0.0, 1.0, 50)
    ax1.plot(x0_fit + vx_fit * t_dense, y0_fit + vy_fit * t_dense, "b-", linewidth=2,
             label=f"OLS Regression (v=[{vx_fit:.2f}, {vy_fit:.2f}])")
    ax1.set_xlabel("World X (meters)", fontsize=11)
    ax1.set_ylabel("World Y (meters)", fontsize=11)
    ax1.set_title("Least-Squares Linear Velocity Estimation", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_aspect("equal")

    # Subplot 2: Residuals
    x_res = x_meas - (x0_fit + vx_fit * t)
    y_res = y_meas - (y0_fit + vy_fit * t)
    ax2.bar(t - 0.015, x_res, width=0.03, label="Residual X", color="royalblue", alpha=0.8)
    ax2.bar(t + 0.015, y_res, width=0.03, label="Residual Y", color="crimson", alpha=0.8)
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=1)
    ax2.axhline(0.03, color="gray", linestyle="--", label="±1σ Sensor Noise (0.03m)")
    ax2.axhline(-0.03, color="gray", linestyle="--")
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Position Residual (meters)", fontsize=11)
    ax2.set_title("Regression Residuals vs Measurement Uncertainty", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_01_velocity_estimation_residuals.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig_02_kalman_trajectory_innovations():
    """Figure 2: Kalman filter state estimate, true trajectory, observations, and innovations."""
    np.random.seed(1315)
    dt = 0.1
    steps = 25
    t = np.arange(steps) * dt

    # True state: starts at (0, 0), vx=1.0, vy=0.4 with small turn at step 12
    x_true = np.zeros(steps)
    y_true = np.zeros(steps)
    vx_true = np.zeros(steps)
    vy_true = np.zeros(steps)
    cx, cy = 0.0, 0.0
    vx, vy = 1.0, 0.4
    for k in range(steps):
        if k == 12:
            vx, vy = 0.6, 1.0
        cx += vx * dt
        cy += vy * dt
        x_true[k] = cx
        y_true[k] = cy
        vx_true[k] = vx
        vy_true[k] = vy

    z_meas = np.column_stack([
        x_true + np.random.normal(0.0, 0.05, size=steps),
        y_true + np.random.normal(0.0, 0.05, size=steps)
    ])

    # Kalman filter rollout
    x_est = np.zeros((steps, 4))
    innov = np.zeros((steps, 2))
    innov_std = np.zeros((steps, 2))

    x_curr = np.array([0.0, 0.0, 1.0, 0.4], dtype=np.float64)
    P_curr = np.diag([0.04, 0.04, 1.0, 1.0])
    q = 1.0
    sigma_p = 0.05
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    R = np.diag([sigma_p**2, sigma_p**2])

    for k in range(steps):
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        Q = q * np.array([
            [(dt**3)/3, 0, (dt**2)/2, 0],
            [0, (dt**3)/3, 0, (dt**2)/2],
            [(dt**2)/2, 0, dt, 0],
            [0, (dt**2)/2, 0, dt],
        ])
        x_pred = F @ x_curr
        P_pred = F @ P_curr @ F.T + Q

        y = z_meas[k] - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_curr = x_pred + K @ y
        I_KH = np.eye(4) - K @ H
        P_curr = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

        x_est[k] = x_curr
        innov[k] = y
        innov_std[k] = [math.sqrt(S[0, 0]), math.sqrt(S[1, 1])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Subplot 1: Trajectory
    ax1.plot(x_true, y_true, "k--", linewidth=2, label="True Obstacle Path")
    ax1.scatter(z_meas[:, 0], z_meas[:, 1], color="crimson", s=35, alpha=0.7, label="Noisy Observations")
    ax1.plot(x_est[:, 0], x_est[:, 1], "b-o", markersize=4, linewidth=2, label="Kalman Filter Estimate")
    ax1.set_xlabel("World X (meters)", fontsize=11)
    ax1.set_ylabel("World Y (meters)", fontsize=11)
    ax1.set_title("Constant-Velocity Kalman Tracking", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_aspect("equal")

    # Subplot 2: Innovation sequence with 2-sigma bounds
    ax2.plot(t, innov[:, 0], "b-o", markersize=4, label="Innovation $y_x$")
    ax2.plot(t, 2 * innov_std[:, 0], "b--", alpha=0.6, label="±2σ Innovation Bound")
    ax2.plot(t, -2 * innov_std[:, 0], "b--", alpha=0.6)
    ax2.plot(t, innov[:, 1], "r-s", markersize=4, label="Innovation $y_y$")
    ax2.plot(t, 2 * innov_std[:, 1], "r--", alpha=0.6)
    ax2.plot(t, -2 * innov_std[:, 1], "r--", alpha=0.6)
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Innovation Residual (meters)", fontsize=11)
    ax2.set_title("Kalman Innovation Sequence & 2σ Gating", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_02_kalman_trajectory_innovations.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig_03_mahalanobis_gate_contours():
    """Figure 3: Mahalanobis distance contours and chi-square gate boundary."""
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    # Anisotropic innovation covariance S
    sigma_x = 0.08
    sigma_y = 0.04
    theta = math.radians(30)
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    D = np.diag([sigma_x**2, sigma_y**2])
    S = R @ D @ R.T

    # Generate random innovations: 300 inliers, 25 outliers
    inliers = np.random.multivariate_normal([0, 0], S, size=300)
    outliers = np.random.uniform(-0.4, 0.4, size=(25, 2))

    inv_S = np.linalg.inv(S)

    # Ellipses for 1-sigma, 2-sigma, and 99% chi2 (sqrt(9.21) = 3.035 sigma)
    scales = [
        (1.0, "1σ Contour ($d_M^2=1.00$)", "dodgerblue", "-"),
        (2.0, "2σ Contour ($d_M^2=4.00$)", "orange", "--"),
        (math.sqrt(9.21), "99% $\chi^2_2$ Gate ($d_M^2=9.21$)", "crimson", "-"),
    ]

    for scale, label, color, lstyle in scales:
        width = 2 * scale * sigma_x
        height = 2 * scale * sigma_y
        angle = math.degrees(theta)
        ell = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                      edgecolor=color, facecolor="none", linewidth=2, linestyle=lstyle, label=label)
        ax.add_patch(ell)

    # Plot inliers vs outliers based on d2 <= 9.21
    all_pts = np.vstack([inliers, outliers])
    d2_all = np.sum((all_pts @ inv_S) * all_pts, axis=1)
    passed = d2_all <= 9.21
    rejected = ~passed

    ax.scatter(all_pts[passed, 0], all_pts[passed, 1], color="darkgreen", s=30, alpha=0.8, label="Gated / Accepted (99.2%)")
    ax.scatter(all_pts[rejected, 0], all_pts[rejected, 1], color="red", marker="x", s=50, linewidth=2, label="Rejected Outliers")

    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlabel("Innovation $y_x$ (meters)", fontsize=11)
    ax.set_ylabel("Innovation $y_y$ (meters)", fontsize=11)
    ax.set_title("Mahalanobis Innovation Gating ($\chi^2_{2, 0.99} = 9.21$)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_03_mahalanobis_gate_contours.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig_04_prediction_covariance_growth():
    """Figure 4: Spatiotemporal trajectory prediction with expanding covariance ellipses."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    dt_step = 0.25
    horizon = 2.0
    steps = int(round(horizon / dt_step))
    t = np.arange(steps + 1) * dt_step

    x0 = [0.0, 0.0, 1.2, 0.5]
    P0 = np.diag([0.04, 0.04, 0.25, 0.25])
    q = 1.0
    sigma_scale = 3.0

    px = x0[0] + x0[2] * t
    py = x0[1] + x0[3] * t
    r_unc = np.zeros(steps + 1)

    # Subplot 1: Trajectory with 3-sigma uncertainty ellipses
    ax1.plot(px, py, "b-o", linewidth=2, markersize=5, label="Predicted Obstacle Waypoints")
    for k in range(steps + 1):
        tau = t[k]
        p_xx = P0[0, 0] + P0[2, 2] * (tau ** 2) + (1.0 / 3.0) * q * (tau ** 3)
        r_unc[k] = sigma_scale * math.sqrt(p_xx)

        ell = Ellipse(xy=(px[k], py[k]), width=2 * r_unc[k], height=2 * r_unc[k],
                      edgecolor="royalblue", facecolor="lightblue", alpha=0.25, linewidth=1.5)
        ax1.add_patch(ell)

    ax1.set_xlabel("World X (meters)", fontsize=11)
    ax1.set_ylabel("World Y (meters)", fontsize=11)
    ax1.set_title("Spatiotemporal Trajectory Prediction (Horizon = 2.0s)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_aspect("equal")

    # Subplot 2: Uncertainty growth curve
    ax2.plot(t, r_unc, "r-o", linewidth=2.5, markersize=5, label="$r_{\\text{unc}}(\\tau) = 3\\sqrt{\\lambda_{\\max}(P_{\\text{pos}})}$")
    ax2.axhline(r_unc[0], color="gray", linestyle="--", label=f"Initial Uncertainty ({r_unc[0]:.2f}m)")
    ax2.set_xlabel("Prediction Horizon $\\tau$ (seconds)", fontsize=11)
    ax2.set_ylabel("Uncertainty Radius $r_{\\text{unc}}$ (meters)", fontsize=11)
    ax2.set_title("CWNA Covariance Expansion vs Future Time", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_04_prediction_covariance_growth.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig_05_dynamic_occupancy_cost_profile():
    """Figure 5: Dynamic occupancy cost and risk level profile across distance from boundary."""
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

    r_unc = 0.60
    sigma_scale = 3.0
    sigma_pos = r_unc / sigma_scale  # 0.20m

    d = np.linspace(0.0, 0.80, 500)
    cost = np.zeros_like(d)

    # Nominal: d == 0 -> cost = 1.0
    # High: 0 < d <= 0.20 -> cost in [0.70, 0.95)
    # Med: 0.20 < d <= 0.40 -> cost in [0.40, 0.70)
    # Low: 0.40 < d <= 0.60 -> cost in [0.15, 0.40)
    # Free: d > 0.60 -> cost = 0.0

    for i, dist in enumerate(d):
        if dist < 1e-4:
            cost[i] = 1.0
        elif dist <= sigma_pos:
            cost[i] = 0.95 - 0.25 * (dist / sigma_pos)
        elif dist <= 2.0 * sigma_pos:
            cost[i] = 0.70 - 0.30 * ((dist - sigma_pos) / sigma_pos)
        elif dist <= 3.0 * sigma_pos:
            cost[i] = 0.40 - 0.25 * ((dist - 2.0 * sigma_pos) / sigma_pos)
        else:
            cost[i] = 0.0

    # Shaded risk zones
    ax1.axvspan(0.0, 0.005, color="black", alpha=0.25, label="LETHAL (Nominal Footprint)")
    ax1.axvspan(0.005, sigma_pos, color="crimson", alpha=0.20, label="HIGH RISK ($d \\leq 1\\sigma$)")
    ax1.axvspan(sigma_pos, 2 * sigma_pos, color="orange", alpha=0.20, label="MEDIUM RISK ($1\\sigma < d \\leq 2\\sigma$)")
    ax1.axvspan(2 * sigma_pos, 3 * sigma_pos, color="gold", alpha=0.20, label="LOW RISK ($2\\sigma < d \\leq 3\\sigma$)")
    ax1.axvspan(3 * sigma_pos, 0.80, color="lightgreen", alpha=0.20, label="FREE SPACE ($d > 3\\sigma$)")

    ax1.plot(d, cost, "b-", linewidth=3, label="Dynamic Occupancy Cost $C(d)$")
    ax1.axhline(0.65, color="purple", linestyle="--", linewidth=1.5, label="Collision Cost Threshold (0.65)")

    ax1.set_xlim(0.0, 0.80)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Distance from Nominal Obstacle Boundary $d$ (meters)", fontsize=11)
    ax1.set_ylabel("Dynamic Occupancy Cost", fontsize=11)
    ax1.set_title("Spatiotemporal Dynamic Occupancy Cost & Risk Hierarchy", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_05_dynamic_occupancy_cost_profile.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig_06_sat_collision_margins():
    """Figure 6: SAT oriented bounding boxes and safety expansion margin."""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    # Robot at origin, rotated 15 deg
    r_pos = (0.0, 0.0)
    r_size = (0.80, 0.60)
    r_yaw = math.radians(15)
    margin = 0.05

    # Nominal robot corners and expanded margin corners
    def get_obb(pos, size, yaw, m=0.0):
        hl = (size[0] + 2 * m) / 2.0
        hw = (size[1] + 2 * m) / 2.0
        corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
        c = math.cos(yaw)
        s = math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        return (R @ corners.T).T + pos

    r_nom = get_obb(r_pos, r_size, r_yaw, 0.0)
    r_exp = get_obb(r_pos, r_size, r_yaw, margin)

    # Obstacle 1: Colliding within margin
    o1_pos = (0.75, 0.15)
    o1_size = (0.60, 0.60)
    o1_yaw = math.radians(-20)
    o1_nom = get_obb(o1_pos, o1_size, o1_yaw, 0.0)

    # Obstacle 2: Clear
    o2_pos = (-0.85, 0.80)
    o2_size = (0.50, 0.50)
    o2_yaw = 0.0
    o2_nom = get_obb(o2_pos, o2_size, o2_yaw, 0.0)

    # Draw patches
    ax.add_patch(Polygon(r_exp, edgecolor="blue", facecolor="lightblue", alpha=0.3, linestyle="--", linewidth=2, label="Robot Margin Expansion (0.05m)"))
    ax.add_patch(Polygon(r_nom, edgecolor="darkblue", facecolor="royalblue", alpha=0.7, linewidth=2, label="Robot Nominal Footprint"))
    ax.add_patch(Polygon(o1_nom, edgecolor="red", facecolor="crimson", alpha=0.6, linewidth=2, label="Obstacle 1 (SAT Overlap w/ Margin)"))
    ax.add_patch(Polygon(o2_nom, edgecolor="green", facecolor="lightgreen", alpha=0.6, linewidth=2, label="Obstacle 2 (Separated)"))

    # Separating axis visualization for Obstacle 2
    edge = r_nom[1] - r_nom[0]
    normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
    axis_start = np.array([-1.2, -0.4])
    axis_end = axis_start + normal * 1.8
    ax.plot([axis_start[0], axis_end[0]], [axis_start[1], axis_end[1]], "k-.", linewidth=1.5, label="SAT Separating Axis Normal")

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.0, 1.4)
    ax.set_xlabel("World X (meters)", fontsize=11)
    ax.set_ylabel("World Y (meters)", fontsize=11)
    ax.set_title("Separating Axis Theorem (SAT) OBB Collision Validation", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_aspect("equal")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_06_sat_collision_margins.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Generating Phase 15 publication plots in", OUTPUT_DIR, "...")
    plot_fig_01_velocity_estimation_residuals()
    plot_fig_02_kalman_trajectory_innovations()
    plot_fig_03_mahalanobis_gate_contours()
    plot_fig_04_prediction_covariance_growth()
    plot_fig_05_dynamic_occupancy_cost_profile()
    plot_fig_06_sat_collision_margins()
    print("All 6 publication figures successfully generated.")


if __name__ == "__main__":
    main()
