"""Diagnostic plotter producing structured engineering figures for dynamic SIL scenarios."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from simulation.scenarios.dynamic.motion_models import BaseObstacleMotionModel


class ScenarioDiagnosticPlotter:
    """Generates comprehensive multi-panel diagnostic plots for SIL scenario runs."""

    @staticmethod
    def plot_scenario(
        scenario_id: str,
        telemetry: List[Dict[str, Any]],
        motion_models: Sequence[BaseObstacleMotionModel],
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        output_filepath: str,
    ) -> str:
        """Create and save a 2-panel diagnostic figure for a completed scenario."""
        if not telemetry:
            return ""

        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

        fig, (ax_map, ax_time) = plt.subplots(
            2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1.8, 1.0]}
        )
        fig.suptitle(f"Dynamic SIL Scenario: {scenario_id}", fontsize=14, fontweight="bold")

        # ---------------------------------------------------------------------
        # Panel 1: 2D Spatiotemporal Trajectory Map
        # ---------------------------------------------------------------------
        ax_map.set_title("2D Trajectory Map (Ground Truth vs. Perception/Planning)", fontsize=11)

        # UGV Executed Trajectory
        ugv_x = [t["robot_x"] for t in telemetry]
        ugv_y = [t["robot_y"] for t in telemetry]
        ax_map.plot(ugv_x, ugv_y, "b-", linewidth=2.5, label="UGV Executed Trajectory", zorder=4)
        ax_map.plot(start_pose[0], start_pose[1], "go", markersize=9, label="Start Pose", zorder=5)
        ax_map.plot(goal_pose[0], goal_pose[1], "g*", markersize=14, label="Goal Pose", zorder=5)

        # Ground Truth Obstacle Trajectories
        sim_times = [t["sim_time"] for t in telemetry]
        for idx, model in enumerate(motion_models):
            gt_pts = []
            for t_sim in sim_times:
                gt = model.evaluate(t_sim)
                if gt is not None and gt.is_active:
                    gt_pts.append((gt.x, gt.y))
            if gt_pts:
                gx, gy = zip(*gt_pts)
                ax_map.plot(
                    gx, gy, "r--", linewidth=2.0, alpha=0.85,
                    label=f"GT Obstacle #{model.spec.obstacle_id}" if idx == 0 else None,
                    zorder=3,
                )
                # Plot initial and final obstacle locations
                ax_map.plot(gx[0], gy[0], "rs", markersize=6, alpha=0.7, zorder=3)
                ax_map.plot(gx[-1], gy[-1], "rx", markersize=8, alpha=0.7, zorder=3)

        # Estimated Tracks (from tracker)
        track_x, track_y = [], []
        for t in telemetry:
            for trk in t.get("tracked_details", []):
                track_x.append(trk["x"])
                track_y.append(trk["y"])
        if track_x:
            ax_map.scatter(
                track_x, track_y, color="orange", s=18, marker="o",
                alpha=0.6, label="Tracked Obstacles (Filter Est.)", zorder=4,
            )

        # Predicted Obstacle Trajectories (sampled at intervals)
        pred_x, pred_y = [], []
        sample_step = max(1, len(telemetry) // 6)
        for i, t in enumerate(telemetry):
            if i % sample_step == 0:
                for pred in t.get("prediction_details", []):
                    for wp in pred.get("waypoints", []):
                        pred_x.append(wp[1])
                        pred_y.append(wp[2])
        if pred_x:
            ax_map.scatter(
                pred_x, pred_y, color="purple", s=12, marker=".",
                alpha=0.4, label="Predicted Trajectories (Horizon)", zorder=2,
            )

        # Minimum Clearance Annotation
        min_clearance_step = min(telemetry, key=lambda t: t.get("min_clearance_m", float("inf")), default=None)
        if min_clearance_step is not None:
            mc = min_clearance_step.get("min_clearance_m", 0.0)
            if mc < float("inf"):
                ax_map.annotate(
                    f"Min Clearance: {mc:.2f}m",
                    xy=(min_clearance_step["robot_x"], min_clearance_step["robot_y"]),
                    xytext=(min_clearance_step["robot_x"] + 0.3, min_clearance_step["robot_y"] + 0.4),
                    arrowprops=dict(facecolor="black", arrowstyle="->"),
                    fontsize=9, fontweight="semibold",
                )

        ax_map.set_xlabel("X Position (m)", fontsize=10)
        ax_map.set_ylabel("Y Position (m)", fontsize=10)
        ax_map.grid(True, linestyle=":", alpha=0.6)
        ax_map.legend(loc="best", fontsize=8, framealpha=0.9)
        ax_map.axis("equal")

        # ---------------------------------------------------------------------
        # Panel 2: Velocity & Behavioral State Timeline
        # ---------------------------------------------------------------------
        ax_time.set_title("Vehicle Kinematics & Behavioral State Timeline", fontsize=11)
        times = [t["sim_time"] for t in telemetry]
        v_cmd = [t["cmd_v"] for t in telemetry]
        v_act = [t["robot_v"] for t in telemetry]
        clearance = [min(5.0, t.get("min_clearance_m", 5.0)) for t in telemetry]

        ax_time.plot(times, v_cmd, "b--", linewidth=1.5, label="Command v (m/s)")
        ax_time.plot(times, v_act, "b-", linewidth=2.0, label="Actual v (m/s)")
        ax_time.plot(times, clearance, "g:", linewidth=1.5, label="Clearance (m, capped at 5m)")

        # Shading by behavioral state
        state_colors = {
            "PROCEED": "lightgreen",
            "AVOID": "orange",
            "WAIT": "yellow",
            "STOP": "red",
        }
        for i in range(len(telemetry) - 1):
            st = telemetry[i].get("behavior_state", "PROCEED")
            color = state_colors.get(st, "white")
            ax_time.axvspan(times[i], times[i + 1], color=color, alpha=0.15)

        ax_time.set_xlabel("Simulation Time (s)", fontsize=10)
        ax_time.set_ylabel("Velocity / Clearance", fontsize=10)
        ax_time.grid(True, linestyle=":", alpha=0.6)
        ax_time.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_filepath, dpi=150)
        plt.close(fig)
        return output_filepath
