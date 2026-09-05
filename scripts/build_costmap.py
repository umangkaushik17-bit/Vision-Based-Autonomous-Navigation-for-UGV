"""CLI utility to construct a planner-ready BEV Terrain Costmap from a camera frame."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import cv2
import numpy as np

from ai.costmap import (
    BEVCostmapFuser,
    CostmapMetadata,
    CostmapSerializer,
    CostmapVisualizer,
    LayerGenerator,
    ObstacleInflator,
    TemporalCostmapAccumulator,
)
from ai.obstacles import UnifiedHazardDetector
from ai.perception_fusion import SafetyPerceptionFusion
from ai.traversability.rugd_model import RUGDTraversabilityPredictor
from ai.traversability.sam_tp import SAMTPTraversabilityPredictor


def main():
    parser = argparse.ArgumentParser(description="Build a planner-ready BEV costmap from input frame")
    parser.add_argument("--image", type=str, default="frame.jpg", help="Path to input RGB frame")
    parser.add_argument("--output-prefix", type=str, default="outputs/costmap/exports/costmap_sample", help="Output file prefix")
    parser.add_argument("--inflate", action="store_true", default=True, help="Apply vehicle footprint obstacle inflation")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print("\n" + "=" * 70)
    print("         SIH26126 PHASE 5: BEV TERRAIN COSTMAP GENERATION")
    print("=" * 70)

    # 1. Perception
    print("[1/4] Running Perception Pipeline...")
    t0 = time.perf_counter()
    pred_a = SAMTPTraversabilityPredictor()
    pred_b = RUGDTraversabilityPredictor()
    hazard_detector = UnifiedHazardDetector()
    fusion = SafetyPerceptionFusion()

    res_a = pred_a.predict(rgb)
    res_b = pred_b.predict(rgb)
    dets = hazard_detector.detect(rgb)
    fused_res = fusion.fuse(rgb, res_a, res_b, dets)
    print(f"      Perception completed in {(time.perf_counter() - t0)*1000:.1f} ms")

    # 2. Layer Generation & BEV Projection
    print("[2/4] Generating 8-Layer BEV Raster Grid...")
    t1 = time.perf_counter()
    layer_gen = LayerGenerator()
    layers = layer_gen.generate_layers(fused_res, rgb_image=rgb)
    metadata = CostmapMetadata(
        resolution_m=layer_gen.resolution_m,
        shape=(layer_gen.n_x, layer_gen.n_y),
        x_range_m=layer_gen.x_range_m,
        y_range_m=layer_gen.y_range_m,
        timestamp=time.time()
    )
    print(f"      BEV Layer generation completed in {(time.perf_counter() - t1)*1000:.1f} ms")

    # 3. Multi-Layer Cost Fusion & Inflation
    print("[3/4] Fusing Costmap with Safety Overrides & Inflation...")
    fuser = BEVCostmapFuser()
    costmap = fuser.fuse_layers(layers, metadata)

    if args.inflate:
        inflator = ObstacleInflator(resolution_m=layer_gen.resolution_m)
        final_costmap = inflator.inflate_costmap(costmap)
    else:
        final_costmap = costmap

    stats = final_costmap.summary_stats()
    print(f"      Grid Dimensions : {stats['grid_dimensions']} ({stats['total_cells']} cells)")
    print(f"      Safe Area Ratio : {stats['safe_cell_ratio']*100:.1f}%")
    print(f"      Blocked Ratio   : {stats['blocked_cell_ratio']*100:.1f}%")
    print(f"      Unknown Ratio   : {stats['unknown_cell_ratio']*100:.1f}%")

    # 4. Serialization & Visualization
    print("[4/4] Serializing and Rendering Diagnostic HUD...")
    npz_path, json_path = CostmapSerializer.save(final_costmap, args.output_prefix)
    print(f"      Saved NPZ: {npz_path}")
    print(f"      Saved JSON: {json_path}")

    viz = CostmapVisualizer()
    hud = viz.create_9_view_figure(rgb, fused_res.overlay, costmap, final_costmap)
    hud_path = Path(args.output_prefix).with_suffix(".png")
    cv2.imwrite(str(hud_path), cv2.cvtColor(hud, cv2.COLOR_RGB2BGR))
    print(f"      Saved 9-View Diagnostic HUD: {hud_path}")

    print("\n[OK] Costmap generation successfully completed.")


if __name__ == "__main__":
    main()
