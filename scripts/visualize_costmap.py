"""Visualizer generating comprehensive 9-panel diagnostic HUDs across test scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

from ai.costmap import (
    BEVCostmapFuser,
    CostmapMetadata,
    CostmapVisualizer,
    LayerGenerator,
    ObstacleInflator,
    TemporalCostmapAccumulator,
)
from ai.obstacles.base import HazardType, ObstacleDetection, RiskLevel
from ai.perception_fusion import SafetyAwarePerceptionResult

OUTPUT_DIR = Path("outputs/costmap/scenarios")


def generate_scenario_costmap(
    scenario_name: str,
    rgb_frame: np.ndarray,
    traversability: np.ndarray,
    hazard_mask: np.ndarray,
    slope_degrees: float = 0.0,
    is_unknown: bool = False
) -> Path:
    """Run costmap generation on a scenario and save 9-view HUD."""
    h, w = rgb_frame.shape[:2]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    risk_map = np.zeros((h, w), dtype=np.float32)
    risk_map[hazard_mask] = 1.0
    risk_map[~hazard_mask] = 1.0 - traversability[~hazard_mask]

    unk_mask = np.full((h, w), is_unknown, dtype=bool)
    conf_map = np.full((h, w), 0.90 if not is_unknown else 0.20, dtype=np.float32)

    safety_overlay = rgb_frame.copy()
    safety_overlay[hazard_mask] = [0, 0, 220]  # Red in BGR

    perception_res = SafetyAwarePerceptionResult(
        final_traversability=traversability,
        hazard_mask=hazard_mask,
        hazard_classes=["HAZARD"] if hazard_mask.any() else [],
        risk_map=risk_map,
        unknown_mask=unk_mask,
        confidence_map=conf_map,
        detections=[],
        overlay=safety_overlay,
        latency_ms=5.0
    )

    layer_gen = LayerGenerator()
    layers = layer_gen.generate_layers(perception_res, rgb_image=rgb_frame)

    # Inject geometric slope if specified
    if slope_degrees > 0.0:
        layers["slope_deg"].fill(slope_degrees)

    metadata = CostmapMetadata(
        resolution_m=layer_gen.resolution_m,
        shape=(layer_gen.n_x, layer_gen.n_y),
        x_range_m=layer_gen.x_range_m,
        y_range_m=layer_gen.y_range_m
    )

    fuser = BEVCostmapFuser()
    costmap = fuser.fuse_layers(layers, metadata)

    inflator = ObstacleInflator(resolution_m=layer_gen.resolution_m)
    inflated = inflator.inflate_costmap(costmap)

    temporal = TemporalCostmapAccumulator(metadata)
    accum = temporal.update(inflated)

    viz = CostmapVisualizer()
    hud = viz.create_9_view_figure(rgb_frame, safety_overlay, costmap, accum)

    out_path = OUTPUT_DIR / f"{scenario_name}_9view.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(hud, cv2.COLOR_RGB2BGR))
    print(f"  [SAVED] {scenario_name}: {out_path}")
    return out_path


def main():
    print("\n" + "=" * 70)
    print("      SIH26126 PHASE 5: COSTMAP SCENARIO VISUALIZATION SUITE")
    print("=" * 70)

    h, w = 480, 640

    # 1. Nominal Safe Trail
    rgb_safe = np.full((h, w, 3), 130, dtype=np.uint8)
    trav_safe = np.full((h, w), 0.92, dtype=np.float32)
    haz_safe = np.zeros((h, w), dtype=bool)
    generate_scenario_costmap("nominal_safe_trail", rgb_safe, trav_safe, haz_safe)

    # 2. Creek Water Reflection (Phase 3/4 Failure Case Mitigated)
    rgb_creek = np.full((h, w, 3), 110, dtype=np.uint8)
    rgb_creek[300:400, 180:460] = [245, 245, 255]
    trav_creek = np.full((h, w), 0.85, dtype=np.float32)
    haz_creek = np.zeros((h, w), dtype=bool)
    haz_creek[300:400, 180:460] = True  # Water hazard detected
    generate_scenario_costmap("creek_water_reflection", rgb_creek, trav_creek, haz_creek)

    # 3. Protruding Dark Rock Obstacle
    rgb_rock = np.full((h, w, 3), 180, dtype=np.uint8)
    rgb_rock[320:380, 260:340] = [30, 30, 30]
    trav_rock = np.full((h, w), 0.88, dtype=np.float32)
    haz_rock = np.zeros((h, w), dtype=bool)
    haz_rock[320:380, 260:340] = True
    generate_scenario_costmap("protruding_dark_rock", rgb_rock, trav_rock, haz_rock)

    # 4. Critical Rollover Slope (30 degrees)
    rgb_slope = np.full((h, w, 3), 140, dtype=np.uint8)
    trav_slope = np.full((h, w), 0.70, dtype=np.float32)
    haz_slope = np.zeros((h, w), dtype=bool)
    generate_scenario_costmap("critical_rollover_slope", rgb_slope, trav_slope, haz_slope, slope_degrees=30.0)

    # 5. Model Disagreement / Sensor Dropout (Unknown)
    rgb_unk = np.full((h, w, 3), 100, dtype=np.uint8)
    trav_unk = np.full((h, w), 0.50, dtype=np.float32)
    haz_unk = np.zeros((h, w), dtype=bool)
    generate_scenario_costmap("model_disagreement_unknown", rgb_unk, trav_unk, haz_unk, is_unknown=True)

    print("\n[OK] All scenario diagnostic HUDs generated successfully.")


if __name__ == "__main__":
    main()
