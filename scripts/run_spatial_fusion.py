"""CLI tool to execute Phase 9 Spatial Fusion across dataset sequences or synthetic streams."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from ai.costmap.base import CostmapMetadata
from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.localization.base import CameraIntrinsics, TrackingState
from ai.localization.datasets import SyntheticStereoSequenceGenerator
from ai.localization.stereo_vo import StereoVisualOdometry
from ai.spatial_fusion.fusion_engine import SpatialFusionEngine
from ai.spatial_fusion.world_map import WorldTerrainMap


def main():
    parser = argparse.ArgumentParser(description="Run Phase 9 Perception + Localization Spatial Fusion")
    parser.add_argument("--num-frames", type=int, default=25, help="Number of frames to process")
    parser.add_argument("--resolution", type=float, default=0.10, help="World grid resolution in m/cell")
    parser.add_argument("--extent-x", type=float, default=30.0, help="World map X span (m)")
    parser.add_argument("--extent-y", type=float, default=30.0, help="World map Y span (m)")
    parser.add_argument("--out-dir", type=str, default="outputs/spatial_fusion/runs", help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    camera = CameraIntrinsics(fx=462.14, fy=462.14, cx=320.0, cy=240.0, width=640, height=480, baseline_m=0.05)
    vo = StereoVisualOdometry(camera)
    wmap = WorldTerrainMap(
        resolution_m=args.resolution,
        x_bounds_m=(-args.extent_x / 2.0, args.extent_x / 2.0),
        y_bounds_m=(-args.extent_y / 2.0, args.extent_y / 2.0)
    )
    engine = SpatialFusionEngine(world_map=wmap)

    gen = SyntheticStereoSequenceGenerator(num_frames=args.num_frames, baseline_m=camera.baseline_m, motion_type="straight")

    print(f"\nRunning Spatial Fusion on {args.num_frames} frames...")
    print(f"World Map Grid: {wmap.shape} ({wmap.resolution_m} m/cell)")

    for i, frame in enumerate(gen):
        est = vo.process_frame(frame.image_left, frame.timestamp, frame.image_right)
        engine.add_localization(est)

        # Generate local BEV costmap
        H, W = (58, 60)
        bev = BEVTerrainCostmap(
            cost=np.full((H, W), 0.15, dtype=np.float32),
            traversability=np.full((H, W), 0.85, dtype=np.float32),
            obstacle=np.zeros((H, W), dtype=bool),
            hazard=np.zeros((H, W), dtype=np.float32),
            unknown=np.zeros((H, W), dtype=bool),
            confidence=np.full((H, W), 0.90, dtype=np.float32),
            slope_deg=np.zeros((H, W), dtype=np.float32),
            roughness=np.zeros((H, W), dtype=np.float32),
            height_var=np.zeros((H, W), dtype=np.float32),
            observation_count=np.ones((H, W), dtype=np.int32),
            metadata=CostmapMetadata(resolution_m=0.10, shape=(H, W), x_range_m=(0.2, 6.0), y_range_m=(-3.0, 3.0), timestamp=frame.timestamp)
        )

        res = engine.fuse_local_costmap(bev, timestamp=frame.timestamp)
        if i % 5 == 0 or i == args.num_frames - 1:
            print(f"  Frame {i+1:<3}: State={est.tracking_state.value:<11} | Fused={res.success} | Fused Cells={res.fused_cells_count}")

    stats = wmap.summary_stats()
    print("\nRun Complete. Summary Stats:")
    print(f"  Observed Cells: {stats['observed_cells']} / {stats['total_cells']} ({stats['coverage_pct']:.1f}%)")
    print(f"  Mean Confidence: {stats['mean_confidence']:.2f}")
    print(f"  Memory: {stats['memory_mb']:.2f} MB\n")


if __name__ == "__main__":
    main()
