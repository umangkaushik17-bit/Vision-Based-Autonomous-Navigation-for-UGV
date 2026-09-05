"""Generate comprehensive diagnostic visualizations for Phase 9 Spatial Fusion."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from ai.costmap.base import CostmapMetadata
from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.localization.base import TrackingState
from ai.localization.result import LocalizationEstimate
from ai.spatial_fusion.fusion_engine import SpatialFusionEngine
from ai.spatial_fusion.transforms import create_se3_matrix, euler_to_rotation_matrix
from ai.spatial_fusion.world_map import WorldTerrainMap


def generate_all_visualizations():
    out_dir = Path("outputs/spatial_fusion/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup world map and engine
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 15.0), y_bounds_m=(-10.0, 10.0))
    engine = SpatialFusionEngine(world_map=wmap, degraded_confidence_scale=0.50)

    # Local BEV observation generator
    def get_bev(t, obs_body=(2.5, 0.5), haz_body=(4.0, -0.8)):
        H, W = (58, 60)
        cost = np.full((H, W), 0.15, dtype=np.float32)
        trav = np.full((H, W), 0.85, dtype=np.float32)
        obst = np.zeros((H, W), dtype=bool)
        hazard = np.zeros((H, W), dtype=np.float32)
        unknown = np.zeros((H, W), dtype=bool)
        conf = np.full((H, W), 0.90, dtype=np.float32)
        meta = CostmapMetadata(resolution_m=0.10, shape=(H, W), x_range_m=(0.2, 6.0), y_range_m=(-3.0, 3.0), timestamp=t)
        bev = BEVTerrainCostmap(cost, trav, obst, hazard, unknown, conf, np.zeros((H, W), dtype=np.float32),
                                np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32),
                                np.ones((H, W), dtype=np.int32), meta)
        # Add rock
        ro, co = bev.meter_to_grid(obs_body[0], obs_body[1])
        bev.obstacle[max(0, ro-2):min(H, ro+3), max(0, co-2):min(W, co+3)] = True
        bev.cost[max(0, ro-2):min(H, ro+3), max(0, co-2):min(W, co+3)] = 1.0
        # Add creek / water hazard
        rh, ch = bev.meter_to_grid(haz_body[0], haz_body[1])
        bev.hazard[max(0, rh-2):min(H, rh+3), max(0, ch-2):min(W, ch+3)] = 0.90
        bev.cost[max(0, rh-2):min(H, rh+3), max(0, ch-2):min(W, ch+3)] = 1.0
        return bev

    # --- Vis 1: Local BEV Observation ---
    bev_step1 = get_bev(1.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(bev_step1.cost, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Local BEV Cost Layer")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(bev_step1.obstacle, cmap="Reds")
    axes[1].set_title("Local BEV Obstacle Layer")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(bev_step1.hazard, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Local BEV Hazard Layer (0.90 Critical)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    vis1_path = out_dir / "vis1_local_bev.png"
    plt.savefig(vis1_path, dpi=150)
    plt.close()
    print(f"Saved: {vis1_path}")

    # Fuse step 1 (Origin, yaw=0)
    T1 = np.eye(4)
    engine.fuse_local_costmap(bev_step1, direct_pose=T1)

    # --- Vis 3: World Map After One Observation ---
    fig, ax = plt.subplots(figsize=(8, 8))
    # Mask unobserved cells
    disp_cost = wmap.cost.copy()
    disp_cost[wmap.unknown] = np.nan
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")
    im = ax.imshow(disp_cost, cmap=cmap, vmin=0, vmax=1,
                   extent=[wmap.x_bounds_m[0], wmap.x_bounds_m[1], wmap.y_bounds_m[0], wmap.y_bounds_m[1]])
    ax.plot(0, 0, "ro", markersize=8, label="Robot Pose (0,0)")
    ax.set_title("Vis 3: World Terrain Map (Single Observation at Origin)")
    ax.set_xlabel("World X (East / Forward) [m]")
    ax.set_ylabel("World Y (North / Left) [m]")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="Cost (0=Free, 1=Blocked, Gray=Unknown)")
    vis3_path = out_dir / "vis3_world_map_single_obs.png"
    plt.savefig(vis3_path, dpi=150)
    plt.close()
    print(f"Saved: {vis3_path}")

    # Fuse step 2 (x=2.0m, yaw=20 deg)
    R2 = euler_to_rotation_matrix(0, 0, np.radians(20))
    T2 = create_se3_matrix(R2, [2.0, 0.5, 0.0])
    # Body-relative obstacle and hazard positions:
    p_obs_b2 = R2.T @ (np.array([2.5, 0.5, 0]) - np.array([2.0, 0.5, 0]))
    p_haz_b2 = R2.T @ (np.array([4.0, -0.8, 0]) - np.array([2.0, 0.5, 0]))
    bev2 = get_bev(2.0, p_obs_b2[:2], p_haz_b2[:2])
    engine.fuse_local_costmap(bev2, direct_pose=T2)

    # Fuse step 3 (x=4.0m, y=1.5m, yaw=45 deg, DEGRADED)
    R3 = euler_to_rotation_matrix(0, 0, np.radians(45))
    T3 = create_se3_matrix(R3, [4.0, 1.5, 0.0])
    p_obs_b3 = R3.T @ (np.array([2.5, 0.5, 0]) - np.array([4.0, 1.5, 0]))
    p_haz_b3 = R3.T @ (np.array([4.0, -0.8, 0]) - np.array([4.0, 1.5, 0]))
    bev3 = get_bev(3.0, p_obs_b3[:2], p_haz_b3[:2])
    engine.fuse_local_costmap(bev3, direct_pose=T3, direct_state=TrackingState.DEGRADED, direct_confidence=0.50)

    # --- Vis 2: Robot Trajectory and Poses ---
    fig, ax = plt.subplots(figsize=(8, 8))
    traj_x = [0.0, 2.0, 4.0]
    traj_y = [0.0, 0.5, 1.5]
    yaws = [0.0, np.radians(20), np.radians(45)]
    ax.plot(traj_x, traj_y, "b--o", label="Robot Path")
    for x, y, yaw in zip(traj_x, traj_y, yaws):
        dx = 0.8 * np.cos(yaw)
        dy = 0.8 * np.sin(yaw)
        ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc="red", ec="red")
    ax.scatter([2.5], [0.5], marker="s", color="black", s=100, label="True Obstacle (2.5, 0.5)")
    ax.scatter([4.0], [-0.8], marker="^", color="darkorange", s=100, label="True Hazard (4.0, -0.8)")
    ax.set_title("Vis 2: Multi-Step Robot Poses and Headings in World Frame")
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Y [m]")
    ax.grid(True)
    ax.legend(loc="upper left")
    vis2_path = out_dir / "vis2_robot_pose_trajectory.png"
    plt.savefig(vis2_path, dpi=150)
    plt.close()
    print(f"Saved: {vis2_path}")

    # --- Vis 4: World Map After Multiple Robot Poses ---
    fig, ax = plt.subplots(figsize=(8, 8))
    disp_cost = wmap.cost.copy()
    disp_cost[wmap.unknown] = np.nan
    im = ax.imshow(disp_cost, cmap=cmap, vmin=0, vmax=1,
                   extent=[wmap.x_bounds_m[0], wmap.x_bounds_m[1], wmap.y_bounds_m[0], wmap.y_bounds_m[1]])
    ax.plot(traj_x, traj_y, "r-o", markersize=6, label="Robot Trajectory")
    ax.set_title("Vis 4: World Terrain Map (Multi-Pose Accumulated Coverage)")
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Y [m]")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="Cost")
    vis4_path = out_dir / "vis4_world_map_multi_obs.png"
    plt.savefig(vis4_path, dpi=150)
    plt.close()
    print(f"Saved: {vis4_path}")

    # --- Vis 5: Hazard and Obstacle Preservation ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im_obst = axes[0].imshow(wmap.obstacle, cmap="Reds",
                             extent=[wmap.x_bounds_m[0], wmap.x_bounds_m[1], wmap.y_bounds_m[0], wmap.y_bounds_m[1]])
    axes[0].set_title("Vis 5A: World Obstacle Layer (Spatially Fixed at (2.5, 0.5))")
    axes[0].set_xlim(-1, 8)
    axes[0].set_ylim(-4, 5)
    plt.colorbar(im_obst, ax=axes[0], fraction=0.046)

    im_haz = axes[1].imshow(wmap.hazard, cmap="magma", vmin=0, vmax=1,
                            extent=[wmap.x_bounds_m[0], wmap.x_bounds_m[1], wmap.y_bounds_m[0], wmap.y_bounds_m[1]])
    axes[1].set_title("Vis 5B: World Hazard Layer (Critical 0.90 Preserved)")
    axes[1].set_xlim(-1, 8)
    axes[1].set_ylim(-4, 5)
    plt.colorbar(im_haz, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    vis5_path = out_dir / "vis5_hazard_obstacle_preservation.png"
    plt.savefig(vis5_path, dpi=150)
    plt.close()
    print(f"Saved: {vis5_path}")

    # --- Vis 6: Localization Uncertainty / Spatial Confidence ---
    fig, ax = plt.subplots(figsize=(8, 8))
    disp_conf = wmap.confidence.copy()
    disp_conf[wmap.unknown] = np.nan
    im_conf = ax.imshow(disp_conf, cmap="plasma", vmin=0, vmax=1,
                        extent=[wmap.x_bounds_m[0], wmap.x_bounds_m[1], wmap.y_bounds_m[0], wmap.y_bounds_m[1]])
    ax.set_title("Vis 6: World Spatial Confidence (TRACKING_OK vs DEGRADED Scaling)")
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Y [m]")
    plt.colorbar(im_conf, ax=ax, label="Confidence [0.0, 1.0]")
    vis6_path = out_dir / "vis6_localization_uncertainty.png"
    plt.savefig(vis6_path, dpi=150)
    plt.close()
    print(f"Saved: {vis6_path}")

    # --- Vis 7: Tracking LOST Protection ---
    fig, ax = plt.subplots(figsize=(8, 8))
    # Try fusing when LOST:
    bev_lost = get_bev(4.0)
    res_lost = engine.fuse_local_costmap(bev_lost, direct_pose=T3, direct_state=TrackingState.LOST, direct_confidence=0.10)
    ax.text(0.5, 0.6, "Tracking State: LOST\nZero-Trust Contract Enforced",
            horizontalalignment="center", verticalalignment="center", fontsize=16,
            color="darkred", bbox=dict(boxstyle="round,pad=1", facecolor="mistyrose", edgecolor="darkred", lw=2))
    ax.text(0.5, 0.35, f"Spatial Fusion Blocked: {not res_lost.success}\nReason: {res_lost.reason}\nZero Stale/Unanchored Poses Emitted",
            horizontalalignment="center", verticalalignment="center", fontsize=13,
            color="black", bbox=dict(boxstyle="square,pad=0.8", facecolor="white", edgecolor="gray"))
    ax.set_axis_off()
    ax.set_title("Vis 7: Zero-Trust Tracking LOST Protection in Phase 9")
    vis7_path = out_dir / "vis7_tracking_lost_protection.png"
    plt.savefig(vis7_path, dpi=150)
    plt.close()
    print(f"Saved: {vis7_path}")

    print("\n[OK] All 7 Visualizations generated successfully in outputs/spatial_fusion/visualizations/\n")


if __name__ == "__main__":
    generate_all_visualizations()
