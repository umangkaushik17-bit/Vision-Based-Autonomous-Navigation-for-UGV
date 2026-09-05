"""Export Phase 1-6 costmaps, trajectories, scenarios, and benchmarks into native MATLAB format (.mat, .json, .csv)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import scipy.io as sio

from ai.costmap import BEVTerrainCostmap
from ai.planning import (
    AStarPlanner,
    DWAPlanner,
    GoalState,
    PFRRTStarPlanner,
    PurePursuitTracker,
    RobotState,
)
from ai.traversability.geometry import CameraExtrinsics, CameraIntrinsics

MATLAB_DATA_DIR = Path("matlab/data")
MATLAB_RESULTS_DIR = Path("matlab/results")


def export_camera_geometry() -> dict:
    """Export camera intrinsic and extrinsic calibration parameters."""
    intrinsics = CameraIntrinsics.default_640x480(fov_deg=69.4)
    extrinsics = CameraExtrinsics(pitch_deg=15.0, height_m=0.40, forward_offset_m=0.15)

    cam_data = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "K": intrinsics.K.astype(np.float64),
        "K_inv": intrinsics.K_inv.astype(np.float64),
        "fov_deg": 69.4,
        "camera_pitch_deg": extrinsics.pitch_deg,
        "camera_height_m": extrinsics.height_m,
        "camera_forward_offset_m": extrinsics.forward_offset_m,
        "R_robot_to_camera": extrinsics.rotation_matrix.astype(np.float64),
        "t_robot_to_camera": extrinsics.translation_vector.astype(np.float64),
        "is_physical_calibrated": False,
        "status": "PENDING PHYSICAL DATA"
    }
    return cam_data


def costmap_to_mat_dict(cm: BEVTerrainCostmap) -> dict:
    """Convert BEVTerrainCostmap into clean dictionary suitable for scipy.io.savemat."""
    return {
        "cost": cm.cost.astype(np.float64),
        "traversability": cm.traversability.astype(np.float64),
        "obstacle": cm.obstacle.astype(np.uint8),
        "hazard": cm.hazard.astype(np.float64),
        "unknown": cm.unknown.astype(np.uint8),
        "confidence": cm.confidence.astype(np.float64),
        "slope_deg": cm.slope_deg.astype(np.float64),
        "roughness": cm.roughness.astype(np.float64),
        "height_var": cm.height_var.astype(np.float64),
        "observation_count": cm.observation_count.astype(np.int32),
        "resolution_m": float(cm.resolution_m),
        "x_range_m": np.array(cm.x_range_m, dtype=np.float64),
        "y_range_m": np.array(cm.y_range_m, dtype=np.float64),
        "shape": np.array(cm.shape, dtype=np.int32)
    }


def export_scenarios_and_costmaps() -> tuple[dict, dict]:
    """Export canonical costmaps from Phase 5 and Phase 6 scenarios."""
    raw_maps = {}
    mat_scenarios = {}

    # 1. Nominal flat costmap
    cm_flat = BEVTerrainCostmap.create_empty(x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0), resolution_m=0.10, default_cost=0.10)
    cm_flat.unknown.fill(False)
    raw_maps["nominal_flat"] = cm_flat
    mat_scenarios["nominal_flat"] = costmap_to_mat_dict(cm_flat)

    # 2. Section 61 Terrain Detour scenario (steep ridge in middle)
    cm_detour = BEVTerrainCostmap.create_empty(x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0), resolution_m=0.10, default_cost=0.10)
    cm_detour.unknown.fill(False)
    r_mid, c_mid = cm_detour.meter_to_grid(2.8, 0.0)
    cm_detour.slope_deg[r_mid-4:r_mid+5, c_mid-8:c_mid+9] = 30.0
    cm_detour.cost[r_mid-4:r_mid+5, c_mid-8:c_mid+9] = 1.00
    cm_detour.obstacle[r_mid-4:r_mid+5, c_mid-8:c_mid+9] = True
    raw_maps["terrain_detour"] = cm_detour
    mat_scenarios["terrain_detour"] = costmap_to_mat_dict(cm_detour)

    # 3. Creek water hazard scenario
    cm_creek = BEVTerrainCostmap.create_empty(x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0), resolution_m=0.10, default_cost=0.10)
    cm_creek.unknown.fill(False)
    cm_creek.hazard[r_mid-5:r_mid+6, c_mid-10:c_mid+11] = 1.00
    cm_creek.cost[r_mid-5:r_mid+6, c_mid-10:c_mid+11] = 1.00
    cm_creek.obstacle[r_mid-5:r_mid+6, c_mid-10:c_mid+11] = True
    raw_maps["creek_water"] = cm_creek
    mat_scenarios["creek_water"] = costmap_to_mat_dict(cm_creek)

    # 4. Negative ditch scenario
    cm_ditch = BEVTerrainCostmap.create_empty(x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0), resolution_m=0.10, default_cost=0.10)
    cm_ditch.unknown.fill(False)
    cm_ditch.height_var[r_mid-4:r_mid+5, c_mid-15:c_mid+16] = 0.30
    cm_ditch.cost[r_mid-4:r_mid+5, c_mid-15:c_mid+16] = 1.00
    cm_ditch.obstacle[r_mid-4:r_mid+5, c_mid-15:c_mid+16] = True
    raw_maps["negative_ditch"] = cm_ditch
    mat_scenarios["negative_ditch"] = costmap_to_mat_dict(cm_ditch)

    # 5. Unknown territory scenario
    cm_unk = BEVTerrainCostmap.create_empty(x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0), resolution_m=0.10, default_cost=0.10)
    cm_unk.unknown.fill(False)
    cm_unk.unknown[r_mid-8:r_mid+9, c_mid-10:c_mid+11] = True
    cm_unk.cost[r_mid-8:r_mid+9, c_mid-10:c_mid+11] = 0.65
    raw_maps["unknown_patch"] = cm_unk
    mat_scenarios["unknown_patch"] = costmap_to_mat_dict(cm_unk)

    # 6. Completely blocked corridor
    cm_blocked = BEVTerrainCostmap.create_empty(x_range_m=(0.2, 5.8), y_range_m=(-3.0, 3.0), resolution_m=0.10, default_cost=0.10)
    cm_blocked.unknown.fill(False)
    cm_blocked.obstacle[r_mid-3:r_mid+4, :] = True
    cm_blocked.cost[r_mid-3:r_mid+4, :] = 1.00
    raw_maps["blocked_corridor"] = cm_blocked
    mat_scenarios["blocked_corridor"] = costmap_to_mat_dict(cm_blocked)

    return raw_maps, mat_scenarios


def export_plans_and_trajectories(raw_maps: dict) -> dict:
    """Generate and export A*, PF-RRT*, DWA, and Pure Pursuit paths."""
    plans = {}
    cm_detour = raw_maps["terrain_detour"]
    start = RobotState(x=0.8, y=0.0, yaw=0.0, v=0.0, omega=0.0)
    goal = GoalState(x=4.8, y=0.0)

    # 1. A* plan on detour scenario
    astar = AStarPlanner()
    path_a, s_a, diag_a = astar.plan(cm_detour, start, goal)
    if path_a:
        plans["astar_detour"] = {
            "waypoints": path_a.waypoints_xy.astype(np.float64),
            "length_m": float(path_a.length),
            "status": s_a.value,
            "planning_time_ms": float(diag_a.get("planning_time_ms", 0.0)),
            "min_clearance_m": float(diag_a.get("min_clearance_m", 0.0))
        }

    # 2. PF-RRT* plan on detour scenario
    rrt = PFRRTStarPlanner()
    path_r, s_r, diag_r = rrt.plan(cm_detour, start, goal)
    if path_r:
        plans["pfrrt_detour"] = {
            "waypoints": path_r.waypoints_xy.astype(np.float64),
            "length_m": float(path_r.length),
            "status": s_r.value,
            "planning_time_ms": float(diag_r.get("planning_time_ms", 0.0))
        }

    # 3. Direct geometric reference (shortest straight line crossing hazard)
    direct_pts = np.linspace([start.x, start.y], [goal.x, goal.y], 21)
    plans["direct_straight_line"] = {
        "waypoints": direct_pts.astype(np.float64),
        "length_m": float(np.hypot(goal.x - start.x, goal.y - start.y))
    }

    # 4. DWA trajectory rollout
    dwa = DWAPlanner()
    traj_dwa, v_dwa, w_dwa, s_dwa, diag_dwa = dwa.compute_trajectory(start, path_a, cm_detour)
    if traj_dwa:
        plans["dwa_trajectory"] = {
            "trajectory": traj_dwa.trajectory_xy.astype(np.float64),
            "v_cmd": float(v_dwa),
            "w_cmd": float(w_dwa),
            "duration_s": float(traj_dwa.duration),
            "min_clearance_m": float(traj_dwa.min_clearance_m),
            "status": s_dwa.value
        }

    return plans


def export_synthetic_slam_trajectories() -> dict:
    """Generate synthetic ground truth and perturbed trajectories for ATE/RPE verification."""
    t = np.linspace(0.0, 30.0, 301)  # 30 seconds at 10 Hz
    # Figure-8 trajectory
    x_gt = 2.5 * np.sin(2 * np.pi * t / 30.0)
    y_gt = 1.5 * np.sin(4 * np.pi * t / 30.0)
    dx = np.gradient(x_gt, t)
    dy = np.gradient(y_gt, t)
    yaw_gt = np.arctan2(dy, dx)

    gt_trajectory = np.column_stack([t, x_gt, y_gt, yaw_gt])

    # Perturbed 1: Pure translation drift (0.02 m/s linear drift)
    drift_rate = 0.02
    x_drift = x_gt + drift_rate * t
    y_drift = y_gt + 0.5 * drift_rate * t
    traj_drift = np.column_stack([t, x_drift, y_drift, yaw_gt])

    # Perturbed 2: Gaussian noise (std = 0.03m)
    rng = np.random.default_rng(42)
    noise_x = x_gt + rng.normal(0.0, 0.03, len(t))
    noise_y = y_gt + rng.normal(0.0, 0.03, len(t))
    traj_noisy = np.column_stack([t, noise_x, noise_y, yaw_gt])

    # Perturbed 3: Loop closure jump (drifts then corrects at t=20s)
    x_lc = x_gt.copy()
    y_lc = y_gt.copy()
    for i, ti in enumerate(t):
        if ti < 20.0:
            x_lc[i] += 0.03 * ti
            y_lc[i] += 0.02 * ti
        else:
            # Loop closure correction reduces error back to 0.02m
            x_lc[i] += 0.02
            y_lc[i] += 0.01
    traj_lc = np.column_stack([t, x_lc, y_lc, yaw_gt])

    return {
        "ground_truth": gt_trajectory,
        "estimated_drift": traj_drift,
        "estimated_noisy": traj_noisy,
        "estimated_loop_closure": traj_lc,
        "metadata": {
            "duration_s": 30.0,
            "rate_hz": 10.0,
            "type": "SYNTHETIC"
        }
    }


def main():
    MATLAB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MATLAB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[MATLAB DATA EXPORT] Exporting Phase 1-6 Artifacts to matlab/data/ ...")

    # 1. Camera Geometry
    cam_data = export_camera_geometry()
    sio.savemat(str(MATLAB_DATA_DIR / "camera_geometry.mat"), cam_data)
    print("  [OK] camera_geometry.mat saved")

    # 2. Scenarios and Costmaps
    raw_maps, scenarios = export_scenarios_and_costmaps()
    sio.savemat(str(MATLAB_DATA_DIR / "costmap_scenarios.mat"), scenarios)
    print("  [OK] costmap_scenarios.mat saved")

    # 3. Plans and Trajectories
    plans = export_plans_and_trajectories(raw_maps)
    sio.savemat(str(MATLAB_DATA_DIR / "planner_trajectories.mat"), plans)
    print("  [OK] planner_trajectories.mat saved")

    # 4. Synthetic SLAM Trajectories
    slam_data = export_synthetic_slam_trajectories()
    sio.savemat(str(MATLAB_DATA_DIR / "synthetic_slam_trajectories.mat"), slam_data)
    print("  [OK] synthetic_slam_trajectories.mat saved")

    # 5. Export combined manifest JSON
    manifest = {
        "generated_by": "scripts/export_matlab_data.py",
        "description": "Turnkey data export of Phase 1-6 UGV artifacts for MATLAB validation",
        "files": [
            "camera_geometry.mat",
            "costmap_scenarios.mat",
            "planner_trajectories.mat",
            "synthetic_slam_trajectories.mat"
        ],
        "camera_intrinsics": {
            "width": cam_data["width"],
            "height": cam_data["height"],
            "fx": cam_data["fx"],
            "fy": cam_data["fy"],
            "cx": cam_data["cx"],
            "cy": cam_data["cy"]
        },
        "costmap_dimensions": {
            "forward_extent_m": [0.2, 5.8],
            "lateral_extent_m": [-3.0, 3.0],
            "resolution_m": 0.10,
            "grid_shape": [58, 60]
        }
    }
    with open(MATLAB_DATA_DIR / "data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("  [OK] data_manifest.json saved")
    print("\n[SUCCESS] MATLAB data export complete.\n")


if __name__ == "__main__":
    main()
