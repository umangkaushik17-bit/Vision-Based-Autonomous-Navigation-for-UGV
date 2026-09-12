#!/usr/bin/env python3
"""Milestone 13.17: Dynamic Navigation Visualization Pipeline.

Generates reproducible visual inspection artifacts for Phase 13 Dynamic Obstacle Navigation:
1. XY Trajectory Plot (UGV actual path, obstacles, predictions, goal)
2. Velocity vs Time Profiles (linear/angular speeds, behavioral state machine overlays)
3. Dynamic Clearance vs Time (safety margins, lethal boundaries, distance to goal)
4. Forward Prediction Fan (spatiotemporal waypoints and expanding 3-sigma spatial uncertainty)
5. Dynamic Obstacle Tracking & Multi-Agent Analysis
6. Coordinated 4-Panel Engineering Dashboard
7. Frame-by-Frame Closed-Loop Avoidance Animation (GIF)

Consumes empirical benchmark telemetry logs from Phase 13.16 / 13.14 without modifying production AI code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@dataclass
class TelemetryFrame:
    """Strongly-typed container for a single simulation control step."""
    step: int
    sim_time: float
    robot_x: float
    robot_y: float
    robot_yaw: float
    robot_v: float
    robot_w: float
    cmd_v: float
    cmd_w: float
    behavior_state: str
    replan_triggered: bool
    emergency_stop: bool
    deadlock: bool
    min_clearance_m: float
    dist_to_goal: float
    tracked_details: List[Dict[str, Any]] = field(default_factory=list)
    prediction_details: List[Dict[str, Any]] = field(default_factory=list)
    collision_status: str = "SAFE"
    gt_collision: bool = False


@dataclass
class TelemetryDataset:
    """Container for complete mission telemetry and metadata."""
    scenario_id: str
    scenario_name: str
    seed: int
    category: str
    frames: List[TelemetryFrame]
    metadata: Dict[str, Any]


class TelemetryParser:
    """Parses and validates empirical closed-loop telemetry logs."""

    @staticmethod
    def load_file(file_path: str | Path) -> TelemetryDataset:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Telemetry file does not exist: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in telemetry file: {path}") from exc

        # Handle two schema variants: Phase 16 mission dict vs Phase 14 list
        if isinstance(raw_data, dict):
            metadata = raw_data.get("scenario_metadata", {})
            raw_points = raw_data.get("telemetry_points", [])
            scenario_id = metadata.get("scenario_id", metadata.get("scenario_key", path.stem))
            scenario_name = metadata.get("scenario_name", scenario_id)
            seed = metadata.get("seed", 42)
            category = metadata.get("category", "DYNAMIC_NAVIGATION")
        elif isinstance(raw_data, list):
            metadata = {"scenario_id": path.stem}
            raw_points = raw_data
            scenario_id = path.stem
            scenario_name = path.stem
            seed = 42
            category = "STRESS_FAILURE_TEST"
        else:
            raise ValueError(f"Unrecognized telemetry structure in {path}: expected dict or list")

        if not raw_points:
            raise ValueError(f"Telemetry file {path} contains no step records.")

        frames: List[TelemetryFrame] = []
        for pt in raw_points:
            # Check for non-finite values in coordinates/speeds
            rx = float(pt.get("robot_x", 0.0))
            ry = float(pt.get("robot_y", 0.0))
            rv = float(pt.get("robot_v", 0.0))
            rw = float(pt.get("robot_w", 0.0))
            cv = float(pt.get("cmd_v", rv))
            cw = float(pt.get("cmd_w", rw))

            for val, name in [(rx, "robot_x"), (ry, "robot_y"), (rv, "robot_v"), (rw, "robot_w")]:
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite value {val} encountered for {name} in step {pt.get('step')}")

            raw_c = pt.get("min_clearance_m", 10.0)
            try:
                c_val = float(raw_c) if raw_c is not None else 10.0
                if not math.isfinite(c_val) or c_val > 10.0:
                    c_val = 10.0
            except (ValueError, TypeError):
                c_val = 10.0

            frames.append(
                TelemetryFrame(
                    step=int(pt.get("step", len(frames))),
                    sim_time=float(pt.get("sim_time", len(frames) * 0.1)),
                    robot_x=rx,
                    robot_y=ry,
                    robot_yaw=float(pt.get("robot_yaw", 0.0)),
                    robot_v=rv,
                    robot_w=rw,
                    cmd_v=cv,
                    cmd_w=cw,
                    behavior_state=str(pt.get("behavior_state", "PROCEED")),
                    replan_triggered=bool(pt.get("replan_triggered", False)),
                    emergency_stop=bool(pt.get("emergency_stop", False)),
                    deadlock=bool(pt.get("deadlock", False)),
                    min_clearance_m=c_val,
                    dist_to_goal=float(pt.get("dist_to_goal", 0.0)),
                    tracked_details=pt.get("tracked_details", []),
                    prediction_details=pt.get("prediction_details", []),
                    collision_status=str(pt.get("collision_status", "SAFE")),
                    gt_collision=bool(pt.get("gt_collision", False)),
                )
            )

        return TelemetryDataset(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            seed=seed,
            category=category,
            frames=frames,
            metadata=metadata,
        )


class DynamicVisualizationEngine:
    """Generates publication-quality diagnostic plots and animations from telemetry."""

    def __init__(self, dataset: TelemetryDataset, output_dir: str | Path) -> None:
        self.dataset = dataset
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Style configuration
        self.colors = {
            "robot": "#1f77b4",
            "robot_trail": "#4ba3e3",
            "goal": "#2ca02c",
            "obstacle_static": "#7f7f7f",
            "obstacle_dynamic": "#ff7f0e",
            "obs_pred": "#d62728",
            "clearance": "#9467bd",
            "state_proceed": "#d4edda",
            "state_avoid": "#fff3cd",
            "state_wait": "#cce5ff",
            "state_stop": "#f8d7da",
        }

    def generate_all(self, create_animation: bool = True) -> Dict[str, Path]:
        """Generates the full suite of Phase 13.17 visualization artifacts."""
        artifacts: Dict[str, Path] = {}
        artifacts["trajectory_xy"] = self.plot_trajectory_xy()
        artifacts["velocity_profiles"] = self.plot_velocity_profiles()
        artifacts["clearance_vs_time"] = self.plot_clearance_vs_time()
        artifacts["prediction_fan"] = self.plot_prediction_fan()
        artifacts["dynamic_tracking"] = self.plot_dynamic_tracking()
        artifacts["engineering_dashboard"] = self.plot_engineering_dashboard()

        if create_animation:
            artifacts["avoidance_animation"] = self.generate_avoidance_animation()

        return artifacts

    def plot_trajectory_xy(self) -> Path:
        """Generates 2D XY Trajectory inspection plot with obstacle tracks and predictions."""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

        rx = [f.robot_x for f in self.dataset.frames]
        ry = [f.robot_y for f in self.dataset.frames]

        # Plot UGV Trajectory
        ax.plot(rx, ry, color=self.colors["robot"], lw=2.5, label="UGV Path (Closed-Loop)", zorder=4)
        ax.scatter(rx[0], ry[0], color="#004085", s=100, marker="o", edgecolors="black", label="UGV Start (0.0, 0.0)", zorder=6)
        ax.scatter(rx[-1], ry[-1], color="#004085", s=120, marker="s", edgecolors="black", label=f"UGV Final ({rx[-1]:.2f}, {ry[-1]:.2f})", zorder=6)

        # Goal Landmark
        goal_x, goal_y = 4.0, 0.0
        if "B01" in self.dataset.scenario_id or "B05" in self.dataset.scenario_id or "B06" in self.dataset.scenario_id:
            goal_x = 4.0
        ax.scatter([goal_x], [goal_y], color=self.colors["goal"], s=180, marker="*", edgecolors="black", label=f"Target Goal ({goal_x:.1f}, {goal_y:.1f})", zorder=7)
        goal_circle = plt.Circle((goal_x, goal_y), 0.25, color="green", fill=False, ls="--", lw=1.2, label="Goal Tolerance (0.25m)")
        ax.add_patch(goal_circle)

        # Collect dynamic obstacle trajectories
        obs_trajectories: Dict[int, List[Tuple[float, float]]] = {}
        for f in self.dataset.frames:
            for obs in f.tracked_details:
                tid = obs.get("track_id", 0)
                obs_trajectories.setdefault(tid, []).append((obs["x"], obs["y"]))

        obs_colors = ["#ff7f0e", "#e377c2", "#8c564b", "#17becf"]
        for idx, (tid, pts) in enumerate(obs_trajectories.items()):
            ox = [p[0] for p in pts]
            oy = [p[1] for p in pts]
            c = obs_colors[idx % len(obs_colors)]
            ax.plot(ox, oy, color=c, lw=2.0, ls="-.", label=f"Obstacle Track ID {tid}", zorder=5)
            ax.scatter(ox[0], oy[0], color=c, s=70, marker="v", edgecolors="black", zorder=6)
            ax.scatter(ox[-1], oy[-1], color=c, s=90, marker="^", edgecolors="black", zorder=6)

        # Draw prediction snapshots for obstacle 0
        pred_drawn = False
        sample_steps = [int(len(self.dataset.frames) * r) for r in [0.2, 0.4, 0.6]]
        for step_idx in sample_steps:
            if step_idx < len(self.dataset.frames):
                frame = self.dataset.frames[step_idx]
                for p_det in frame.prediction_details:
                    wps = p_det.get("waypoints", [])
                    if len(wps) > 1:
                        px = [w[1] for w in wps]
                        py = [w[2] for w in wps]
                        lbl = "Predicted Future Trajectory" if not pred_drawn else None
                        ax.plot(px, py, color="#d62728", ls=":", lw=1.5, label=lbl, zorder=3)
                        pred_drawn = True

        # Annotate vehicle footprint at midpoint
        mid_idx = len(self.dataset.frames) // 2
        mid_f = self.dataset.frames[mid_idx]
        fp = self._compute_footprint_corners(mid_f.robot_x, mid_f.robot_y, mid_f.robot_yaw, 0.65, 0.45)
        fp_poly = patches.Polygon(fp, closed=True, color="blue", alpha=0.25, ec="darkblue", lw=1.5, zorder=5, label="UGV Footprint (0.65x0.45m)")
        ax.add_patch(fp_poly)

        ax.set_title(f"Phase 13 Dynamic Avoidance: XY Trajectory [{self.dataset.scenario_name}]", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("World X [m] (Forward Navigation Axis)", fontsize=11)
        ax.set_ylabel("World Y [m] (Lateral Axis)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

        out_path = self.out_dir / "01_trajectory_xy.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_velocity_profiles(self) -> Path:
        """Plots linear and angular velocity profiles with behavioral state machine overlays."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, dpi=200)

        t = [f.sim_time for f in self.dataset.frames]
        v_act = [f.robot_v for f in self.dataset.frames]
        v_cmd = [f.cmd_v for f in self.dataset.frames]
        w_act = [f.robot_w for f in self.dataset.frames]
        w_cmd = [f.cmd_w for f in self.dataset.frames]
        states = [f.behavior_state for f in self.dataset.frames]

        # Background behavioral state bands
        self._apply_state_background(ax1, t, states)
        self._apply_state_background(ax2, t, states)

        # Linear velocity plot
        ax1.plot(t, v_cmd, color="black", lw=1.8, ls="--", label="Commanded Linear Velocity (cmd_v)", zorder=4)
        ax1.plot(t, v_act, color="#1f77b4", lw=2.2, label="Actual Linear Velocity (robot_v)", zorder=5)
        ax1.axhline(0.0, color="gray", lw=0.8, ls=":")
        ax1.set_ylabel("Linear Velocity [m/s]", fontsize=11, fontweight="bold")
        ax1.set_title(f"Phase 13 Commanded vs Actual Velocity with State Machine Arbitration [{self.dataset.scenario_id}]", fontsize=13, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

        # Angular velocity plot
        ax2.plot(t, w_cmd, color="black", lw=1.8, ls="--", label="Commanded Yaw Rate (cmd_w)", zorder=4)
        ax2.plot(t, w_act, color="#e377c2", lw=2.2, label="Actual Yaw Rate (robot_w)", zorder=5)
        ax2.axhline(0.0, color="gray", lw=0.8, ls=":")
        ax2.set_xlabel("Simulation Time [s]", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Angular Velocity [rad/s]", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

        # Annotate behavioral legend manually
        state_patches = [
            patches.Patch(facecolor=self.colors["state_proceed"], edgecolor="green", label="PROCEED"),
            patches.Patch(facecolor=self.colors["state_avoid"], edgecolor="orange", label="AVOID"),
            patches.Patch(facecolor=self.colors["state_wait"], edgecolor="blue", label="WAIT"),
            patches.Patch(facecolor=self.colors["state_stop"], edgecolor="red", label="STOP"),
        ]
        fig.legend(handles=state_patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), framealpha=0.95, fontsize=10)

        out_path = self.out_dir / "02_velocity_profiles.png"
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_clearance_vs_time(self) -> Path:
        """Plots dynamic clearance over time alongside safety margins and goal distance."""
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)

        t = [f.sim_time for f in self.dataset.frames]
        clearance = [f.min_clearance_m for f in self.dataset.frames]
        dist_goal = [f.dist_to_goal for f in self.dataset.frames]
        states = [f.behavior_state for f in self.dataset.frames]

        self._apply_state_background(ax1, t, states)

        # Plot dynamic clearance
        finite_c = [c for c in clearance if math.isfinite(c)]
        max_c = max(finite_c) if finite_c else 5.0
        line1 = ax1.plot(t, clearance, color="#6f42c1", lw=2.4, label="Minimum Dynamic Clearance [m]", zorder=5)

        # Safety thresholds
        ax1.axhline(0.35, color="red", lw=2.0, ls="-.", label="E-Stop Lethal Threshold (0.35m)", zorder=4)
        ax1.axhline(0.60, color="orange", lw=1.5, ls="--", label="Caution Avoidance Margin (0.60m)", zorder=4)
        ax1.axhspan(0.0, 0.35, color="red", alpha=0.10, label="Collision Risk Zone")

        ax1.set_xlabel("Simulation Time [s]", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Dynamic Clearance [m]", fontsize=11, fontweight="bold", color="#6f42c1")
        ax1.tick_params(axis="y", labelcolor="#6f42c1")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.set_ylim(0.0, max(min(max_c * 1.1, 10.0), 2.5))

        # Twin axis for distance to goal
        ax2 = ax1.twinx()
        line2 = ax2.plot(t, dist_goal, color="#28a745", lw=1.8, ls=":", label="Distance to Goal [m]", zorder=3)
        ax2.set_ylabel("Remaining Distance to Goal [m]", fontsize=11, fontweight="bold", color="#28a745")
        ax2.tick_params(axis="y", labelcolor="#28a745")

        # Combined Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines] + ["E-Stop Threshold (0.35m)", "Caution Margin (0.60m)"]
        ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

        ax1.set_title(f"Dynamic Clearance & Safety Margin Evolution [{self.dataset.scenario_id}]", fontsize=13, fontweight="bold", pad=10)

        out_path = self.out_dir / "03_clearance_vs_time.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_prediction_fan(self) -> Path:
        """Plots forward trajectory predictions with expanding 3-sigma spatial uncertainty envelopes."""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

        # Select an active step where obstacle is in mid-crossing
        target_frame = None
        for f in self.dataset.frames[len(self.dataset.frames)//4 : 3*len(self.dataset.frames)//4]:
            if f.prediction_details and len(f.prediction_details[0].get("waypoints", [])) >= 10:
                target_frame = f
                break

        if target_frame is None:
            target_frame = self.dataset.frames[len(self.dataset.frames)//2]

        # Plot UGV at snapshot
        ax.scatter([target_frame.robot_x], [target_frame.robot_y], color=self.colors["robot"], s=150, marker="o", edgecolors="black", label="UGV Position at Snapshot", zorder=6)
        ugv_fp = self._compute_footprint_corners(target_frame.robot_x, target_frame.robot_y, target_frame.robot_yaw, 0.65, 0.45)
        ax.add_patch(patches.Polygon(ugv_fp, closed=True, color="blue", alpha=0.3, ec="darkblue", lw=1.5, zorder=5))

        # Obstacle position and history
        all_obs_pts = []
        for f in self.dataset.frames[:target_frame.step + 1]:
            for obs in f.tracked_details:
                if obs.get("track_id", 0) == 0:
                    all_obs_pts.append((obs["x"], obs["y"]))

        if all_obs_pts:
            hx = [p[0] for p in all_obs_pts]
            hy = [p[1] for p in all_obs_pts]
            ax.plot(hx, hy, color="gray", lw=1.8, ls="--", label="Observed Past Track History", zorder=3)
            ax.scatter([hx[-1]], [hy[-1]], color=self.colors["obstacle_dynamic"], s=160, marker="D", edgecolors="black", label="Current Obstacle Position (t_0)", zorder=6)

        # Waypoints and expanding uncertainty ellipses
        if target_frame.prediction_details:
            pred = target_frame.prediction_details[0]
            wps = pred.get("waypoints", [])
            w_times = [w[0] - target_frame.sim_time for w in wps]
            w_x = [w[1] for w in wps]
            w_y = [w[2] for w in wps]
            w_sigma = [w[5] for w in wps]

            ax.plot(w_x, w_y, color="#d62728", lw=2.2, marker=".", label="Predicted Mean Trajectory (CV Model)", zorder=4)

            # Draw spatial uncertainty ellipses/circles for discrete horizons (0.5s, 1.0s, 1.5s, 2.0s)
            target_horizons = [0.5, 1.0, 1.5, 2.0]
            for th in target_horizons:
                idx = min(range(len(w_times)), key=lambda i: abs(w_times[i] - th))
                center_x, center_y = w_x[idx], w_y[idx]
                r_unc = w_sigma[idx]

                circle = patches.Circle(
                    (center_x, center_y),
                    r_unc,
                    color="#d62728",
                    fill=True,
                    alpha=0.15,
                    ls="--",
                    lw=1.5,
                    zorder=2,
                )
                ax.add_patch(circle)
                ax.scatter([center_x], [center_y], color="#d62728", s=60, zorder=5)
                ax.text(center_x + r_unc * 0.7, center_y + 0.05, f"+{th:.1f}s\n(r={r_unc:.2f}m)", fontsize=8, color="#8b0000", fontweight="bold", zorder=6)
        else:
            ax.text(0.5, 0.5, "No dynamic obstacles detected in this scenario\n(Open/Flat Corridor Baseline)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#ced4da"))

        ax.set_title(f"Milestone 13.6/13.17 Prediction Fan & CWNA Covariance Expansion (t={target_frame.sim_time:.1f}s)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("World X [m]", fontsize=11)
        ax.set_ylabel("World Y [m]", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_aspect("equal", adjustable="datalim")
        if target_frame.prediction_details:
            ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

        out_path = self.out_dir / "04_prediction_fan.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_dynamic_tracking(self) -> Path:
        """Plots tracked dynamic obstacles, estimated velocities, and multi-frame continuity."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=200)

        # Subplot 1: Track positions in XY plane
        track_map: Dict[int, List[Dict[str, Any]]] = {}
        for f in self.dataset.frames:
            for obs in f.tracked_details:
                tid = obs.get("track_id", 0)
                track_map.setdefault(tid, []).append(obs)

        colors = ["#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]
        for idx, (tid, points) in enumerate(track_map.items()):
            c = colors[idx % len(colors)]
            xs = [p["x"] for p in points]
            ys = [p["y"] for p in points]
            vxs = [p.get("vx", 0.0) for p in points]
            vys = [p.get("vy", 0.0) for p in points]

            ax1.plot(xs, ys, color=c, lw=2.0, label=f"Track ID {tid} (N={len(points)})")
            # Quiver arrows every 10 steps
            step_skip = max(1, len(points) // 6)
            for k in range(0, len(points), step_skip):
                ax1.quiver(xs[k], ys[k], vxs[k], vys[k], color=c, angles="xy", scale_units="xy", scale=1.5, width=0.007, headwidth=4)

        ax1.set_title("Multi-Frame MOT Position & Velocity Vectors", fontsize=12, fontweight="bold")
        ax1.set_xlabel("World X [m]", fontsize=10)
        ax1.set_ylabel("World Y [m]", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.set_aspect("equal", adjustable="datalim")
        if not track_map:
            ax1.text(0.5, 0.5, "Zero Dynamic Obstacles Active\n(Static / Clear Corridor)",
                     ha="center", va="center", transform=ax1.transAxes, fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#ced4da"))
            ax2.text(0.5, 0.5, "Zero Obstacle Velocity Tracks Recorded",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#ced4da"))
        else:
            ax1.legend(loc="best", fontsize=9)
            ax2.legend(loc="upper right", fontsize=9)

        fig.suptitle(f"Phase 13 Dynamic Obstacle Multi-Target Tracking Performance [{self.dataset.scenario_id}]", fontsize=14, fontweight="bold", y=0.98)

        out_path = self.out_dir / "05_dynamic_tracking.png"
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_engineering_dashboard(self) -> Path:
        """Generates a cohesive 4-panel engineering diagnostic dashboard."""
        fig = plt.figure(figsize=(16, 12), dpi=200)
        gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

        ax_traj = fig.add_subplot(gs[0, 0])
        ax_vel = fig.add_subplot(gs[0, 1])
        ax_clear = fig.add_subplot(gs[1, 0])
        ax_pred = fig.add_subplot(gs[1, 1])

        t = [f.sim_time for f in self.dataset.frames]
        rx = [f.robot_x for f in self.dataset.frames]
        ry = [f.robot_y for f in self.dataset.frames]
        v_cmd = [f.cmd_v for f in self.dataset.frames]
        w_cmd = [f.cmd_w for f in self.dataset.frames]
        clearance = [f.min_clearance_m for f in self.dataset.frames]
        states = [f.behavior_state for f in self.dataset.frames]

        # Panel 1: XY Trajectory
        ax_traj.plot(rx, ry, color=self.colors["robot"], lw=2.2, label="UGV Trajectory")
        ax_traj.scatter(rx[0], ry[0], color="blue", s=80, marker="o", label="Start")
        ax_traj.scatter(rx[-1], ry[-1], color="darkblue", s=90, marker="s", label="End")
        ax_traj.scatter([4.0], [0.0], color="green", s=140, marker="*", label="Goal")
        for f in self.dataset.frames:
            for obs in f.tracked_details:
                ax_traj.scatter([obs["x"]], [obs["y"]], color="#ff7f0e", s=4, alpha=0.3)
        ax_traj.set_title("Panel 1: Closed-Loop XY Path & Obstacle Field", fontsize=11, fontweight="bold")
        ax_traj.set_xlabel("X [m]")
        ax_traj.set_ylabel("Y [m]")
        ax_traj.grid(True, ls="--", alpha=0.5)
        ax_traj.set_aspect("equal", adjustable="datalim")
        ax_traj.legend(loc="upper right", fontsize=8)

        # Panel 2: Velocity Profiles
        self._apply_state_background(ax_vel, t, states)
        ax_vel.plot(t, v_cmd, color="black", lw=1.8, label="v_cmd [m/s]")
        ax_vel.plot(t, w_cmd, color="#d62728", lw=1.8, ls="--", label="w_cmd [rad/s]")
        ax_vel.set_title("Panel 2: Commanded Velocity & Behavioral States", fontsize=11, fontweight="bold")
        ax_vel.set_xlabel("Time [s]")
        ax_vel.set_ylabel("Velocity")
        ax_vel.grid(True, ls="--", alpha=0.5)
        ax_vel.legend(loc="upper right", fontsize=8)

        # Panel 3: Clearance vs Time
        self._apply_state_background(ax_clear, t, states)
        finite_c = [c for c in clearance if math.isfinite(c)]
        max_c = max(finite_c) if finite_c else 5.0
        ax_clear.plot(t, clearance, color="#6f42c1", lw=2.2, label="Dynamic Clearance")
        ax_clear.axhline(0.35, color="red", ls="-.", lw=1.8, label="E-Stop (0.35m)")
        ax_clear.axhline(0.60, color="orange", ls="--", lw=1.5, label="Caution (0.60m)")
        ax_clear.set_ylim(0.0, max(min(max_c * 1.1, 10.0), 2.5))
        ax_clear.set_title("Panel 3: Spatiotemporal Safety Clearance", fontsize=11, fontweight="bold")
        ax_clear.set_xlabel("Time [s]")
        ax_clear.set_ylabel("Clearance [m]")
        ax_clear.grid(True, ls="--", alpha=0.5)
        ax_clear.legend(loc="upper right", fontsize=8)

        # Panel 4: Forward Prediction Fan
        mid_idx = len(self.dataset.frames) // 2
        mid_f = self.dataset.frames[mid_idx]
        ax_pred.scatter([mid_f.robot_x], [mid_f.robot_y], color="blue", s=100, label="UGV Pose")
        if mid_f.prediction_details and mid_f.prediction_details[0].get("waypoints"):
            wps = mid_f.prediction_details[0].get("waypoints", [])
            wx = [w[1] for w in wps]
            wy = [w[2] for w in wps]
            ax_pred.plot(wx, wy, color="red", lw=2.0, marker=".", label="Predicted Obstacle")
            for th in [0.5, 1.0, 1.5, 2.0]:
                idx = min(range(len(wps)), key=lambda i: abs(wps[i][0] - mid_f.sim_time - th))
                circ = patches.Circle((wx[idx], wy[idx]), wps[idx][5], color="red", alpha=0.15)
                ax_pred.add_patch(circ)
            ax_pred.legend(loc="upper left", fontsize=8)
        else:
            ax_pred.text(0.5, 0.5, "No Active Predictions\n(Clear Corridor)",
                         ha="center", va="center", transform=ax_pred.transAxes, fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#ced4da"))
        ax_pred.set_title("Panel 4: Prediction Horizon & Uncertainty Fan", fontsize=11, fontweight="bold")
        ax_pred.set_xlabel("X [m]")
        ax_pred.set_ylabel("Y [m]")
        ax_pred.grid(True, ls="--", alpha=0.5)
        ax_pred.set_aspect("equal", adjustable="datalim")

        fig.suptitle(f"Phase 13.17 Dynamic Navigation Engineering Dashboard: {self.dataset.scenario_name}", fontsize=15, fontweight="bold", y=0.98)

        out_path = self.out_dir / "06_engineering_dashboard.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def generate_avoidance_animation(self, max_frames: int = 50, fps: int = 8) -> Path:
        """Generates an animated GIF demonstrating closed-loop dynamic obstacle avoidance."""
        frames_img: List[Image.Image] = []
        step_stride = max(1, len(self.dataset.frames) // max_frames)
        sampled_frames = self.dataset.frames[::step_stride]

        for idx, f in enumerate(sampled_frames):
            fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

            # Draw static boundaries and goal
            ax.scatter([4.0], [0.0], color="green", s=150, marker="*", label="Goal Beacon", zorder=6)

            # Draw UGV trail
            past_x = [sf.robot_x for sf in sampled_frames[:idx+1]]
            past_y = [sf.robot_y for sf in sampled_frames[:idx+1]]
            ax.plot(past_x, past_y, color="#1f77b4", lw=2.0, ls="--", zorder=3)

            # Draw UGV footprint
            corners = self._compute_footprint_corners(f.robot_x, f.robot_y, f.robot_yaw, 0.65, 0.45)
            ax.add_patch(patches.Polygon(corners, closed=True, color="#1f77b4", alpha=0.8, ec="black", lw=1.5, zorder=5))

            # Heading arrow
            arrow_len = 0.35
            ax.arrow(
                f.robot_x,
                f.robot_y,
                arrow_len * math.cos(f.robot_yaw),
                arrow_len * math.sin(f.robot_yaw),
                head_width=0.12,
                color="yellow",
                zorder=6,
            )

            # Draw dynamic obstacles and predictions
            for obs in f.tracked_details:
                ox, oy = obs["x"], obs["y"]
                ax.scatter([ox], [oy], color="#ff7f0e", s=120, marker="s", ec="black", zorder=6)
                vx, vy = obs.get("vx", 0.0), obs.get("vy", 0.0)
                ax.arrow(ox, oy, vx * 0.8, vy * 0.8, head_width=0.10, color="orange", zorder=6)

            for pred in f.prediction_details:
                wps = pred.get("waypoints", [])
                if len(wps) > 1:
                    ax.plot([w[1] for w in wps], [w[2] for w in wps], color="red", ls=":", lw=1.5, zorder=4)
                    # Future 1.0s circle
                    idx_1s = min(range(len(wps)), key=lambda i: abs(wps[i][0] - f.sim_time - 1.0))
                    ax.add_patch(patches.Circle((wps[idx_1s][1], wps[idx_1s][2]), wps[idx_1s][5], color="red", alpha=0.15))

            # HUD text banner
            state_color = {"PROCEED": "green", "AVOID": "darkorange", "WAIT": "blue", "STOP": "red"}.get(f.behavior_state, "black")
            hud_text = (
                f"Time: {f.sim_time:4.1f}s | Step: {f.step:3d}\n"
                f"State: {f.behavior_state:7s} | Clearance: {f.min_clearance_m:.2f}m\n"
                f"v_cmd: {f.cmd_v:.2f}m/s | w_cmd: {f.cmd_w:.2f}rad/s"
            )
            ax.text(
                0.03,
                0.95,
                hud_text,
                transform=ax.transAxes,
                fontsize=9,
                fontfamily="monospace",
                fontweight="bold",
                va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=state_color, lw=2.0, alpha=0.9),
                zorder=10,
            )

            ax.set_xlim(-0.5, 4.8)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect("equal")
            ax.grid(True, ls="--", alpha=0.5)
            ax.set_title(f"Dynamic Obstacle Avoidance Animation [{self.dataset.scenario_id}]", fontsize=11, fontweight="bold")

            fig.tight_layout()
            fig.canvas.draw()
            # Convert canvas to PIL Image
            rgba = np.asarray(fig.canvas.buffer_rgba())
            img = Image.fromarray(rgba)
            frames_img.append(img.copy())
            plt.close(fig)

        out_path = self.out_dir / "07_avoidance_animation.gif"
        if frames_img:
            frames_img[0].save(
                out_path,
                save_all=True,
                append_images=frames_img[1:],
                duration=int(1000 / fps),
                loop=0,
                optimize=True,
            )

        return out_path

    @staticmethod
    def _apply_state_background(ax: plt.Axes, times: List[float], states: List[str]) -> None:
        """Adds colored vertical spans corresponding to behavioral state machine states."""
        if not times or len(times) != len(states):
            return

        state_colors = {
            "PROCEED": "#d4edda",  # Light green
            "AVOID": "#fff3cd",    # Light yellow
            "WAIT": "#cce5ff",     # Light blue
            "STOP": "#f8d7da",     # Light red
        }

        curr_state = states[0]
        start_t = times[0]

        for i in range(1, len(times)):
            if states[i] != curr_state or i == len(times) - 1:
                end_t = times[i]
                c = state_colors.get(curr_state, "#f0f0f0")
                ax.axvspan(start_t, end_t, facecolor=c, alpha=0.35, zorder=1)
                curr_state = states[i]
                start_t = end_t

    @staticmethod
    def _compute_footprint_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
        """Computes four corners of rectangular vehicle footprint."""
        hl, hw = length / 2.0, width / 2.0
        local = np.array([
            [hl, -hw],
            [hl, hw],
            [-hl, hw],
            [-hl, -hw],
        ])
        rot = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw), math.cos(yaw)],
        ])
        return local @ rot.T + np.array([x, y])


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 13.17 Dynamic Navigation Visualization Generator")
    parser.add_argument("--telemetry", type=str, default=None, help="Path to JSON telemetry file")
    parser.add_argument("--output-dir", type=str, default="simulation/outputs/dynamic_visualization", help="Destination folder for artifacts")
    parser.add_argument("--all-representative", action="store_true", help="Generate plots for all 5 canonical representative scenarios")
    parser.add_argument("--skip-animation", action="store_true", help="Skip GIF animation generation")
    args = parser.parse_args()

    out_base = Path(args.output_dir)

    if args.all_representative:
        rep_files = [
            ("B01", "simulation/outputs/phase16/telemetry/B01_flat_baseline_seed42.json"),
            ("B05", "simulation/outputs/phase16/telemetry/B05_single_dynamic_seed42.json"),
            ("B06", "simulation/outputs/phase16/telemetry/B06_multi_dynamic_seed42.json"),
            ("B07", "simulation/outputs/phase16/telemetry/B07_mixed_static_dynamic_seed42.json"),
            ("B11", "simulation/outputs/phase16/telemetry/B11_sensor_recovery_seed42.json"),
        ]
        for key, fpath in rep_files:
            if not Path(fpath).exists():
                print(f"[WARN] Representative file missing: {fpath}, skipping")
                continue
            print(f"--> Processing representative scenario: {key} ({fpath})")
            ds = TelemetryParser.load_file(fpath)
            target_dir = out_base / key
            engine = DynamicVisualizationEngine(ds, target_dir)
            create_anim = (key == "B05" and not args.skip_animation)
            res = engine.generate_all(create_animation=create_anim)
            for k, p in res.items():
                print(f"    Generated [{k}]: {p} ({p.stat().st_size} bytes)")
        return 0

    target_file = args.telemetry
    if target_file is None:
        target_file = "simulation/outputs/phase16/telemetry/B05_single_dynamic_seed42.json"

    print(f"Loading telemetry from: {target_file}")
    dataset = TelemetryParser.load_file(target_file)
    engine = DynamicVisualizationEngine(dataset, out_base)
    artifacts = engine.generate_all(create_animation=not args.skip_animation)

    print("\nSuccessfully generated Phase 13.17 visualization artifacts:")
    for name, path in artifacts.items():
        print(f"  - {name:22s}: {path} ({path.stat().st_size} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
