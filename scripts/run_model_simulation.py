#!/usr/bin/env python3
"""Closed-Loop Perception & Navigation Simulation with Newly Installed RELLIS-3D Model.

Features:
1. Streams camera frames (from benchmark test sequences, synthetic camera, or user directory)
2. Executes real-time traversability segmentation using LightweightTraversabilityNet / DenseTraversabilityPredictor
3. Projects predicted traversability into local BEV terrain costmap
4. Drives the simulated 4WD UGV using closed-loop A* and DWA planners
5. Renders and saves an interactive 4-panel HUD telemetry dashboard
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Ensure root directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import GoalState, RobotState
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.visualization import PlanningVisualizer
from ai.traversability.dense_predictor import DenseTraversabilityPredictor


def create_hud_dashboard(
    rgb_frame: np.ndarray,
    overlay_frame: np.ndarray,
    costmap_vis: np.ndarray,
    step: int,
    robot_v: float,
    robot_w: float,
    inference_ms: float,
    fps: float,
    model_name: str,
    device: str
) -> np.ndarray:
    """Compose a 4-panel visual dashboard displaying real-time perception and autonomy state."""
    target_h, target_w = 360, 480

    p1 = cv2.resize(rgb_frame, (target_w, target_h))
    p2 = cv2.resize(overlay_frame, (target_w, target_h))
    p3 = cv2.resize(costmap_vis, (target_w, target_h))

    # Panel 4: Telemetry Dashboard
    p4 = np.full((target_h, target_w, 3), 25, dtype=np.uint8)
    cv2.putText(p4, "UGV AUTONOMY TELEMETRY", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 220, 255), 2)
    cv2.line(p4, (20, 45), (target_w - 20, 45), (80, 80, 80), 1)

    lines = [
        f"Model: {model_name[:25]}",
        f"Compute Device: {device}",
        f"Inference Latency: {inference_ms:.2f} ms",
        f"Throughput: {fps:.1f} FPS",
        f"Sim Step: {step:03d}",
        f"Forward Speed v: {robot_v:.2f} m/s",
        f"Angular Velocity w: {robot_w:.2f} rad/s",
        "Costmap: BEV 0.10m Resolution",
        "Traversability: 3-Class Safety Fused",
        "Planner: Global A* + Local DWA"
    ]

    y_pos = 75
    for line in lines:
        color = (255, 255, 255)
        if "Inference" in line or "Throughput" in line:
            color = (50, 255, 50)
        elif "Forward Speed" in line or "Angular" in line:
            color = (0, 215, 255)
        cv2.putText(p4, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y_pos += 26

    # Labels for top panels
    cv2.putText(p1, "1. Camera Input (RGB)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(p2, "2. Traversability Segmentation (Safe=Green, Unsafe=Red)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(p3, "3. Local BEV Costmap & Trajectory", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Combine into 2x2 grid
    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    canvas = np.vstack([top_row, bot_row])

    # Global title bar
    bar = np.zeros((40, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, "SIH2026 UGV AUTONOMOUS NAVIGATION SIMULATION — RELLIS-3D PERCEPTION IN THE LOOP",
                (30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

    return np.vstack([bar, canvas])


def run_simulation(
    checkpoint: str = "outputs/rellis3d/checkpoints/best.pt",
    device: str = "cpu",
    frames_dir: str = "data/benchmark_test/public_rugd/rgb",
    output_dir: str = "outputs/simulation/perception_closed_loop",
    max_steps: int = 25,
    save_video: bool = True
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("      CLOSED-LOOP UGV SIMULATION WITH NEWLY TRAINED RELLIS-3D MODEL       ")
    print("=" * 75)

    # 1. Initialize Predictor
    print(f"[*] Initializing DenseTraversabilityPredictor on {device}...")
    predictor = DenseTraversabilityPredictor(checkpoint=checkpoint, device=device)
    print(f"[+] Model: {predictor.model_name} (Checkpoint: {checkpoint})")

    # 2. Gather Test Frames
    frame_paths = sorted(list(Path(frames_dir).glob("*.png")) + list(Path(frames_dir).glob("*.jpg")))
    if not frame_paths:
        # Fallback to synthetic or any available test frame
        frame_paths = sorted(Path("data").rglob("*.png"))[:10]
        if not frame_paths:
            print("[-] No test frames found. Generating synthetic outdoor terrain frames...")
            frame_paths = []

    print(f"[+] Discovered {len(frame_paths)} input frames from {frames_dir}")

    # 3. Setup Navigation Planners
    global_planner = AStarPlanner()
    local_planner = DWAPlanner()
    viz = PlanningVisualizer()

    # Robot Start and Target Goal
    current_state = RobotState(x=0.5, y=0.0, yaw=0.0, v=0.0, omega=0.0)
    goal = GoalState(x=4.5, y=0.0, xy_tolerance_m=0.20)

    # Setup Video Writer if requested
    video_writer = None
    video_out_path = out_path / "ugv_simulation_run.mp4"

    sim_frames = []
    print("\n[*] Starting closed-loop simulation loop...")

    for step in range(max_steps):
        t_step_start = time.perf_counter()

        # A. Acquire Current Camera Frame
        if frame_paths:
            frame_file = frame_paths[step % len(frame_paths)]
            bgr = cv2.imread(str(frame_file))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            # Synthetic camera frame with road and obstacles
            rgb = np.full((400, 640, 3), 110, dtype=np.uint8)
            rgb[200:, 180:460] = [170, 160, 140]  # Drivable path

        # B. Model Forward Inference
        t_inf_0 = time.perf_counter()
        trav_res = predictor.predict(rgb)
        inf_ms = (time.perf_counter() - t_inf_0) * 1000.0

        # C. Construct BEV Costmap from Traversability Predictions
        costmap = BEVTerrainCostmap.create_empty(
            x_range_m=(0.2, 5.0), y_range_m=(-2.5, 2.5),
            resolution_m=0.10, default_cost=0.10
        )
        costmap.unknown.fill(False)

        # Convert predicted drivability map to costmap obstacle mask
        h_grid, w_grid = costmap.shape
        drivability_grid = cv2.resize(trav_res.mask, (w_grid, h_grid), interpolation=cv2.INTER_LINEAR)
        # Terrain with safe prob < 0.35 marked as lethal obstacle
        lethal_mask = drivability_grid < 0.35
        costmap.obstacle[lethal_mask] = True
        costmap.cost[lethal_mask] = 1.00
        costmap.cost[~lethal_mask] = np.clip(1.0 - drivability_grid[~lethal_mask], 0.05, 0.90)

        # D. Planning Cycle (Global + Local DWA)
        global_path, g_status, _ = global_planner.plan(costmap, current_state, goal)
        if global_path is None:
            # Fallback path straight to goal
            from ai.planning.path import Path as GlobalPath, PathPoint
            global_path = GlobalPath([
                PathPoint(x=current_state.x, y=current_state.y, yaw=0.0),
                PathPoint(x=goal.x, y=goal.y, yaw=0.0)
            ])

        traj, cmd_v, cmd_w, status, diag = local_planner.compute_trajectory(
            current_state, global_path, costmap
        )

        # Render BEV Costmap View
        bev_hud = viz.render_plan(
            costmap, current_state, goal,
            global_path=global_path,
            local_trajectory=traj,
            title=f"Step {step:02d} | v={cmd_v:.2f}m/s w={cmd_w:.2f}rad/s"
        )

        # E. Propagate Robot Kinematics (dt = 0.10 s)
        dt = 0.10
        new_yaw = current_state.yaw + cmd_w * dt
        new_yaw = (new_yaw + np.pi) % (2.0 * np.pi) - np.pi
        new_x = current_state.x + cmd_v * np.cos(new_yaw) * dt
        new_y = current_state.y + cmd_v * np.sin(new_yaw) * dt

        current_state = RobotState(
            x=new_x, y=new_y, yaw=new_yaw, v=cmd_v, omega=cmd_w,
            timestamp=current_state.timestamp + dt
        )

        fps = 1.0 / max(time.perf_counter() - t_step_start, 1e-4)

        # F. Compose Multi-Panel Dashboard
        dashboard = create_hud_dashboard(
            rgb_frame=rgb,
            overlay_frame=trav_res.overlay,
            costmap_vis=bev_hud,
            step=step,
            robot_v=cmd_v,
            robot_w=cmd_w,
            inference_ms=inf_ms,
            fps=fps,
            model_name=predictor.model_name,
            device=predictor.device
        )

        # Save HUD frame
        frame_save_path = out_path / f"sim_step_{step:03d}.png"
        cv2.imwrite(str(frame_save_path), cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR))

        if video_writer is None and save_video:
            h_vid, w_vid = dashboard.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(video_out_path), fourcc, 10.0, (w_vid, h_vid))

        if video_writer is not None:
            video_writer.write(cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR))

        print(f"  Step {step:02d}/{max_steps}: v={cmd_v:.2f} m/s, w={cmd_w:.2f} rad/s | Latency: {inf_ms:.1f} ms | Saved: {frame_save_path.name}")

        if goal.is_reached(current_state):
            print(f"[+] Target goal reached at step {step}!")
            break

    if video_writer is not None:
        video_writer.release()
        print(f"\n[+] Rendered full simulation video: {video_out_path}")

    print(f"[+] All simulation artifacts saved in: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Closed-Loop Simulation with RELLIS-3D Model")
    parser.add_argument("--checkpoint", type=str, default="outputs/rellis3d/checkpoints/best.pt", help="Path to trained model weights")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda:0')")
    parser.add_argument("--frames-dir", type=str, default="data/benchmark_test/public_rugd/rgb", help="Directory of camera frames")
    parser.add_argument("--output-dir", type=str, default="outputs/simulation/perception_closed_loop", help="Output directory")
    parser.add_argument("--steps", type=int, default=15, help="Number of simulation steps")
    args = parser.parse_args()

    run_simulation(
        checkpoint=args.checkpoint,
        device=args.device,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        max_steps=args.steps
    )


if __name__ == "__main__":
    main()
