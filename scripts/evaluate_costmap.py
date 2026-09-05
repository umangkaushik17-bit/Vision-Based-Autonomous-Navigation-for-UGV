"""Evaluation and ablation harness for BEV costmap generation and temporal stability."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from ai.costmap import (
    BEVCostmapFuser,
    CostmapMetadata,
    LayerGenerator,
    ObstacleInflator,
    TemporalCostmapAccumulator,
)
from ai.perception_fusion import SafetyAwarePerceptionResult

OUTPUT_DIR = Path("outputs/costmap/comparisons")


def run_ablation_study(layers: Dict[str, np.ndarray], metadata: CostmapMetadata) -> Dict[str, Any]:
    """Execute Ablations A through D to isolate the contribution of each layer."""
    print("\n[ABLATION] Running Multi-Layer Contribution Study...")

    # Ablation A: Learned Perception Only
    fuser_a = BEVCostmapFuser(
        w_traversability=1.0, w_hazard=0.0, w_slope=0.0, w_roughness=0.0, w_height=0.0, w_unknown=0.0
    )
    map_a = fuser_a.fuse_layers(layers, metadata)

    # Ablation B: Learned Perception + Explicit Hazards
    fuser_b = BEVCostmapFuser(
        w_traversability=0.60, w_hazard=0.40, w_slope=0.0, w_roughness=0.0, w_height=0.0, w_unknown=0.0
    )
    map_b = fuser_b.fuse_layers(layers, metadata)

    # Ablation C: Learned Perception + 3D Terrain Geometry
    fuser_c = BEVCostmapFuser(
        w_traversability=0.60, w_hazard=0.0, w_slope=0.25, w_roughness=0.15, w_height=0.0, w_unknown=0.0
    )
    map_c = fuser_c.fuse_layers(layers, metadata)

    # Ablation D: Full Fusion Model (Default with all layers + hard overrides)
    fuser_d = BEVCostmapFuser()
    map_d = fuser_d.fuse_layers(layers, metadata)

    return {
        "ablation_a_learned_only": map_a.summary_stats(),
        "ablation_b_learned_plus_hazards": map_b.summary_stats(),
        "ablation_c_learned_plus_geometry": map_c.summary_stats(),
        "ablation_d_full_fusion": map_d.summary_stats()
    }


def measure_temporal_stability(metadata: CostmapMetadata, num_frames: int = 20) -> Dict[str, Any]:
    """Measure cell jitter, occupancy flicker, and state transitions with and without temporal fusion."""
    print(f"\n[TEMPORAL] Measuring Temporal Stability over {num_frames} frames...")

    temporal = TemporalCostmapAccumulator(metadata)
    fuser = BEVCostmapFuser()
    inflator = ObstacleInflator(resolution_m=metadata.resolution_m)

    # Simulate moving forward at 0.5 m/s (0.05m per frame at 10 Hz) with random perceptual noise
    np.random.seed(42)
    single_frame_flicker = 0
    temporal_frame_flicker = 0

    prev_single_obs = None
    prev_temporal_obs = None

    for i in range(num_frames):
        # Base traversability with intermittent sensor noise
        raw_trav = np.full(metadata.shape, 0.85, dtype=np.float32)
        noise = np.random.randn(*metadata.shape).astype(np.float32) * 0.15
        trav_noisy = np.clip(raw_trav + noise, 0.0, 1.0)

        # Persistent obstacle at row 20-25, col 25-35, with occasional 1-frame sensor dropout
        haz_mask = np.zeros(metadata.shape, dtype=np.float32)
        if i not in [5, 12]:  # Artificial dropouts at frame 5 and 12
            haz_mask[20:25, 25:35] = 1.0

        mock_layers = {
            "traversability": trav_noisy,
            "hazard": haz_mask,
            "obstacle": haz_mask > 0.5,
            "unknown": np.zeros(metadata.shape, dtype=bool),
            "confidence": np.full(metadata.shape, 0.90, dtype=np.float32),
            "slope_deg": np.zeros(metadata.shape, dtype=np.float32),
            "roughness": np.zeros(metadata.shape, dtype=np.float32),
            "height_var": np.zeros(metadata.shape, dtype=np.float32),
            "observation_count": np.ones(metadata.shape, dtype=np.int32)
        }

        # Single frame costmap
        costmap_single = fuser.fuse_layers(mock_layers, metadata)
        obs_single = costmap_single.obstacle

        # Temporal costmap with forward delta dx=0.05m
        dx = 0.05 if i > 0 else 0.0
        costmap_temp = temporal.update(costmap_single, pose_delta=(dx, 0.0, 0.0))
        obs_temp = costmap_temp.obstacle

        if prev_single_obs is not None:
            single_frame_flicker += int(np.sum(obs_single != prev_single_obs))
            temporal_frame_flicker += int(np.sum(obs_temp != prev_temporal_obs))

        prev_single_obs = obs_single
        prev_temporal_obs = obs_temp

    flicker_reduction_pct = (1.0 - temporal_frame_flicker / max(single_frame_flicker, 1)) * 100.0

    print(f"  Single Frame Obstacle State Transitions : {single_frame_flicker}")
    print(f"  Temporal Fused Obstacle State Transitions: {temporal_frame_flicker}")
    print(f"  Flicker & Jitter Reduction               : {flicker_reduction_pct:.1f}%")

    return {
        "frames_evaluated": num_frames,
        "single_frame_cell_transitions": single_frame_flicker,
        "temporal_cell_transitions": temporal_frame_flicker,
        "flicker_reduction_percent": round(flicker_reduction_pct, 2)
    }


def evaluate_bev_false_safe_preservation() -> Dict[str, Any]:
    """Verify that BEV projection preserves Phase 4 safety and assigns lethal cost to water/obstacles."""
    print("\n[SAFETY] Verifying False-Safe Preservation after BEV Conversion...")

    # Scenario: Creek water reflection patch in camera view
    h, w = 480, 640
    rgb = np.full((h, w, 3), 110, dtype=np.uint8)
    rgb[300:400, 200:440] = [245, 245, 255]

    # Ground truth: water region is lethal obstacle
    gt_water_pixels = np.zeros((h, w), dtype=bool)
    gt_water_pixels[300:400, 200:440] = True

    # Phase 4 perception output: water detected and overridden to 0.0 traversability
    trav = np.full((h, w), 0.85, dtype=np.float32)
    trav[gt_water_pixels] = 0.0
    hazard = np.zeros((h, w), dtype=np.float32)
    hazard[gt_water_pixels] = 1.0

    perception_res = SafetyAwarePerceptionResult(
        final_traversability=trav,
        hazard_mask=gt_water_pixels,
        hazard_classes=["WATER"],
        risk_map=hazard,
        unknown_mask=np.zeros((h, w), dtype=bool),
        confidence_map=np.full((h, w), 0.95, dtype=np.float32),
        detections=[],
        overlay=rgb,
        latency_ms=10.0
    )

    layer_gen = LayerGenerator()
    layers = layer_gen.generate_layers(perception_res, rgb_image=rgb)
    metadata = CostmapMetadata(
        resolution_m=layer_gen.resolution_m,
        shape=(layer_gen.n_x, layer_gen.n_y),
        x_range_m=layer_gen.x_range_m,
        y_range_m=layer_gen.y_range_m
    )

    fuser = BEVCostmapFuser()
    costmap = fuser.fuse_layers(layers, metadata)

    # Check BEV cells where hazard was projected
    bev_hazard_cells = (costmap.hazard >= 0.85)
    false_safe_bev_cells = (costmap.cost[bev_hazard_cells] < 0.50).sum()

    print(f"  BEV Hazard Cells Identified            : {int(bev_hazard_cells.sum())}")
    print(f"  False-Safe BEV Cells (Cost < 0.50)     : {int(false_safe_bev_cells)}")
    print(f"  Mean Lethal Cost in Hazard Zone        : {float(costmap.cost[bev_hazard_cells].mean()):.4f}")

    return {
        "bev_hazard_cells_count": int(bev_hazard_cells.sum()),
        "false_safe_bev_cells_count": int(false_safe_bev_cells),
        "mean_hazard_cost": round(float(costmap.cost[bev_hazard_cells].mean()), 4),
        "safety_preservation_pass": bool(false_safe_bev_cells == 0)
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("      SIH26126 PHASE 5: COSTMAP EVALUATION & ABLATION HARNESS")
    print("=" * 70)

    # Setup standard test scenario
    h, w = 480, 640
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    trav = np.full((h, w), 0.80, dtype=np.float32)
    haz_mask = np.zeros((h, w), dtype=bool)
    haz_mask[320:380, 250:350] = True
    risk = np.zeros((h, w), dtype=np.float32)
    risk[haz_mask] = 1.0

    perception_res = SafetyAwarePerceptionResult(
        final_traversability=trav,
        hazard_mask=haz_mask,
        hazard_classes=["ROCK"],
        risk_map=risk,
        unknown_mask=np.zeros((h, w), dtype=bool),
        confidence_map=np.full((h, w), 0.90, dtype=np.float32),
        detections=[],
        overlay=rgb,
        latency_ms=10.0
    )

    layer_gen = LayerGenerator()
    layers = layer_gen.generate_layers(perception_res, rgb_image=rgb)
    metadata = CostmapMetadata(
        resolution_m=layer_gen.resolution_m,
        shape=(layer_gen.n_x, layer_gen.n_y),
        x_range_m=layer_gen.x_range_m,
        y_range_m=layer_gen.y_range_m
    )

    # 1. Ablation Study
    ablation_results = run_ablation_study(layers, metadata)

    # 2. Temporal Stability Metrics
    temporal_results = measure_temporal_stability(metadata)

    # 3. False-Safe Preservation Verification
    false_safe_results = evaluate_bev_false_safe_preservation()

    master_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ablation_study": ablation_results,
        "temporal_stability": temporal_results,
        "false_safe_preservation": false_safe_results
    }

    report_path = OUTPUT_DIR / "costmap_evaluation_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[OK] Master evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
