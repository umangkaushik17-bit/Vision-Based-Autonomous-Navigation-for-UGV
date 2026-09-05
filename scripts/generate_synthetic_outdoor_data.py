"""Synthetic Outdoor Sensor Data Generator for SIH26126 Autonomous UGV.

Generates realistic outdoor camera frames, semantic traversability ground truth,
depth maps, and simulated IMU readings. This empowers immediate offline development
and deterministic unit testing for perception, localization, and planning without
requiring heavy 3D simulation engines.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

def generate_synthetic_frame(
    step: int,
    total_steps: int,
    width: int = 640,
    height: int = 480
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Generate a single synchronized multi-modal sensor packet.
    
    Returns:
        rgb_image: (H, W, 3) uint8 BGR
        traversability_gt: (H, W) uint8 (0=SAFE, 1=UNKNOWN, 2=UNSAFE)
        depth_map: (H, W) float32 in meters
        imu_pose_data: dict of IMU readings and ground-truth pose
    """
    # Initialize blank images
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    trav_gt = np.ones((height, width), dtype=np.uint8)  # Default UNKNOWN
    depth = np.full((height, width), 50.0, dtype=np.float32)

    horizon_y = int(height * 0.35)

    # 1. Sky Rendering (Blue gradient with slight cloud noise)
    for y in range(horizon_y):
        factor = y / horizon_y
        b = int(220 + 35 * factor)
        g = int(180 + 40 * factor)
        r = int(130 + 50 * factor)
        rgb[y, :] = (b, g, r)
        trav_gt[y, :] = 2 # Sky is non-traversable

    # 2. Ground plane base (Rough grass / soil texture)
    # Add Perlin-like texture using Gaussian blurs
    base_terrain = np.random.randint(40, 110, (height - horizon_y, width, 3), dtype=np.uint8)
    base_terrain = cv2.GaussianBlur(base_terrain, (25, 25), 0)
    # Give green/brown outdoor hue
    base_terrain[:, :, 0] = (base_terrain[:, :, 0] * 0.4).astype(np.uint8)  # Blue low
    base_terrain[:, :, 1] = (base_terrain[:, :, 1] * 0.8 + 40).clip(0, 255).astype(np.uint8)  # Green med
    base_terrain[:, :, 2] = (base_terrain[:, :, 2] * 0.6 + 30).clip(0, 255).astype(np.uint8)  # Red/brown
    rgb[horizon_y:, :] = base_terrain
    trav_gt[horizon_y:, :] = 1 # Rough grass is UNKNOWN by default

    # 3. Ground plane depth (from ~1.0m at bottom to 30.0m at horizon)
    for y in range(horizon_y, height):
        normalized_y = (y - horizon_y) / (height - horizon_y)  # 0 at horizon, 1 at bottom
        d = 1.2 / max(0.02, normalized_y)
        depth[y, :] = np.clip(d, 0.5, 35.0)

    # 4. Winding Trail (Safe Path)
    # Path center shifts gently as the robot moves
    path_offset = math.sin(step * 0.15) * 50.0
    trail_points_left = []
    trail_points_right = []
    
    for y in range(horizon_y, height):
        norm = (y - horizon_y) / (height - horizon_y)
        center_x = (width / 2.0) + path_offset * (1.0 - norm * 0.5)
        # Perspective trail width: narrow at horizon, wide at bottom
        half_width = 15.0 + norm * 140.0
        trail_points_left.append([center_x - half_width, y])
        trail_points_right.append([center_x + half_width, y])

    trail_poly = np.array(trail_points_left + trail_points_right[::-1], dtype=np.int32)
    # Draw dirt trail
    trail_color = (60, 100, 140)  # BGR for dry dirt path
    cv2.fillPoly(rgb, [trail_poly], trail_color)
    cv2.fillPoly(trav_gt, [trail_poly], 0)  # 0 = SAFE

    # Add noise/grain to trail for realism
    noise = np.random.normal(0, 8, (height, width, 3)).astype(np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 5. Place Obstacles (Boulders & Tree Trunks)
    # Place a rock hazard near the path on specific frames
    rock_pos_y = int(height * 0.65)
    rock_pos_x = int((width / 2.0) + path_offset + math.cos(step * 0.2) * 90.0)
    rock_radius_x = int(25 + 10 * (rock_pos_y / height))
    rock_radius_y = int(18 + 8 * (rock_pos_y / height))

    if horizon_y < rock_pos_y < height - 20:
        # Draw 3D-shaded boulder
        cv2.ellipse(rgb, (rock_pos_x, rock_pos_y), (rock_radius_x, rock_radius_y), 0, 0, 360, (70, 75, 80), -1)
        # Highlight and shadow
        cv2.ellipse(rgb, (rock_pos_x - 5, rock_pos_y - 4), (rock_radius_x // 2, rock_radius_y // 2), 0, 0, 360, (110, 115, 120), -1)
        # Mark as UNSAFE (2)
        cv2.ellipse(trav_gt, (rock_pos_x, rock_pos_y), (rock_radius_x, rock_radius_y), 0, 0, 360, 2, -1)
        # Depth adjustment: rock is closer than ground
        rock_depth = depth[rock_pos_y, min(width - 1, max(0, rock_pos_x))] * 0.85
        cv2.ellipse(depth, (rock_pos_x, rock_pos_y), (rock_radius_x, rock_radius_y), 0, 0, 360, float(rock_depth), -1)

    # 6. Simulated Lighting Variation (Sunlight shift & shadows)
    sunlight_intensity = 0.85 + 0.15 * math.sin(step * 0.1)
    rgb = np.clip(rgb.astype(np.float32) * sunlight_intensity, 0, 255).astype(np.uint8)

    # 7. Simulated IMU & Ground Truth Pose
    dt = 0.1 # 10 Hz
    robot_speed = 0.8 # m/s
    yaw = math.sin(step * 0.1) * 0.2
    x = step * robot_speed * dt * math.cos(yaw)
    y = step * robot_speed * dt * math.sin(yaw)

    imu_pose_data = {
        "sequence_id": step,
        "timestamp_sec": round(step * dt, 3),
        "ground_truth_pose": {
            "x_m": round(x, 3),
            "y_m": round(y, 3),
            "yaw_rad": round(yaw, 4)
        },
        "imu": {
            "linear_acceleration_mps2": [
                round(np.random.normal(0.0, 0.05), 3),
                round(np.random.normal(0.0, 0.05), 3),
                round(9.81 + np.random.normal(0.0, 0.1), 3)
            ],
            "angular_velocity_radps": [
                round(np.random.normal(0.0, 0.01), 4),
                round(np.random.normal(0.0, 0.01), 4),
                round(yaw * 0.1 + np.random.normal(0.0, 0.02), 4)
            ]
        }
    }

    return rgb, trav_gt, depth, imu_pose_data

def generate_dataset(output_dir: str, num_frames: int = 15, width: int = 640, height: int = 480):
    """Generate and save an outdoor synthetic dataset sequence."""
    out_path = Path(output_dir)
    rgb_dir = out_path / "rgb"
    gt_dir = out_path / "traversability_gt"
    depth_dir = out_path / "depth"
    
    rgb_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    print(f"[Synthetic Generator] Generating {num_frames} frames into {out_path}...")

    for i in range(num_frames):
        rgb, gt, depth, meta = generate_synthetic_frame(i, num_frames, width, height)
        
        frame_name = f"frame_{i:04d}"
        rgb_file = rgb_dir / f"{frame_name}.png"
        gt_file = gt_dir / f"{frame_name}.png"
        depth_file = depth_dir / f"{frame_name}.npy"

        cv2.imwrite(str(rgb_file), rgb)
        # Scale GT for easy visual inspection: 0->0 (black/safe), 1->127 (gray/unknown), 2->255 (white/unsafe)
        cv2.imwrite(str(gt_file), gt * 127)
        np.save(str(depth_file), depth)

        meta["files"] = {
            "rgb": str(rgb_file.relative_to(out_path)),
            "gt": str(gt_file.relative_to(out_path)),
            "depth": str(depth_file.relative_to(out_path))
        }
        manifest.append(meta)

    manifest_file = out_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Synthetic Generator] Successfully created dataset with {num_frames} frames at {out_path}")
    print(f"[Synthetic Generator] Manifest saved to {manifest_file}")

def main():
    parser = argparse.ArgumentParser(description="Synthetic Outdoor UGV Sensor Data Generator")
    parser.add_argument("--output", type=str, default="data/synthetic_sample", help="Output directory")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to generate")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    args = parser.parse_args()

    generate_dataset(args.output, args.frames, args.width, args.height)

if __name__ == "__main__":
    main()
