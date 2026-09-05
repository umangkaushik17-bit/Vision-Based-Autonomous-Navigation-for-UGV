"""Costmap visualization utilities producing 9-view comprehensive diagnostic HUDs."""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import matplotlib.cm as cm
import numpy as np

from ai.costmap.base import VehicleFootprint
from ai.costmap.bev_costmap import BEVTerrainCostmap


class CostmapVisualizer:
    """Renders high-definition diagnostic panels and HUDs for BEV costmaps."""

    def __init__(self, footprint: Optional[VehicleFootprint] = None) -> None:
        self.footprint = footprint or VehicleFootprint()

    def render_cost_layer(
        self,
        cost_array: np.ndarray,
        unknown_mask: Optional[np.ndarray] = None,
        colormap: str = "RdYlGn_r"
    ) -> np.ndarray:
        """Render a normalized cost array [0, 1] into an RGB heatmap (Green=Safe, Red=Lethal, Gray=Unknown)."""
        norm_cost = np.clip(cost_array, 0.0, 1.0)
        cmap = cm.get_cmap(colormap)
        rgba = cmap(norm_cost)
        rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)

        if unknown_mask is not None:
            rgb[unknown_mask] = [128, 128, 128]  # Neutral gray for unknown cells

        return rgb

    def draw_vehicle_footprint(
        self,
        bev_image: np.ndarray,
        costmap: BEVTerrainCostmap
    ) -> np.ndarray:
        """Draw the vehicle footprint polygon at the robot origin (0, 0) on the BEV map."""
        vis = bev_image.copy()
        # Origin (0, 0) in grid space
        r_orig, c_orig = costmap.meter_to_grid(0.0, 0.0)

        # Footprint vertices
        corners_m = self.footprint.footprint_polygon
        corner_pts = []
        for x_m, y_m in corners_m:
            r, c = costmap.meter_to_grid(x_m, y_m)
            corner_pts.append([c, r])

        pts = np.array(corner_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # Draw origin crosshair
        if costmap.is_grid_in_bounds(r_orig, c_orig):
            cv2.drawMarker(vis, (c_orig, r_orig), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=2)

        return vis

    def create_9_view_figure(
        self,
        rgb_frame: np.ndarray,
        safety_overlay: np.ndarray,
        costmap: BEVTerrainCostmap,
        temporal_costmap: Optional[BEVTerrainCostmap] = None
    ) -> np.ndarray:
        """Compose a 3x3 diagnostic panel (9 views) illustrating the entire perception-to-costmap pipeline.

        Row 1:
          1. RGB Camera Input
          2. Phase 4 Safety Perception Overlay
          3. BEV Learned Traversability Map
        Row 2:
          4. BEV Hazard Layer
          5. BEV Terrain Slope Map (Degrees)
          6. BEV Terrain Roughness Map
        Row 3:
          7. BEV Unknown & Confidence Map
          8. Instantaneous Fused Costmap
          9. Temporal Accumulated & Inflated Costmap (with Robot Footprint)
        """
        target_size = (320, 240)

        # 1. RGB
        v1 = cv2.resize(rgb_frame, target_size)
        cv2.putText(v1, "1. Camera RGB Input", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 2. Phase 4 Safety Overlay
        v2 = cv2.resize(safety_overlay, target_size)
        cv2.putText(v2, "2. Phase 4 Safety Perception", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # 3. BEV Traversability
        trav_rgb = self.render_cost_layer(1.0 - costmap.traversability, costmap.unknown, "RdYlGn")
        v3 = cv2.resize(trav_rgb, target_size)
        cv2.putText(v3, f"3. BEV Traversability (Mean: {costmap.traversability.mean():.2f})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

        # 4. BEV Hazard Layer
        hazard_rgb = self.render_cost_layer(costmap.hazard, costmap.unknown, "Reds")
        v4 = cv2.resize(hazard_rgb, target_size)
        cv2.putText(v4, f"4. BEV Hazards (Max: {costmap.hazard.max():.2f})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

        # 5. BEV Slope
        slope_norm = np.clip(costmap.slope_deg / 30.0, 0.0, 1.0)
        slope_rgb = self.render_cost_layer(slope_norm, costmap.unknown, "YlOrRd")
        v5 = cv2.resize(slope_rgb, target_size)
        cv2.putText(v5, f"5. Terrain Slope (Max: {costmap.slope_deg.max():.1f} deg)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

        # 6. BEV Roughness
        rough_rgb = self.render_cost_layer(costmap.roughness, costmap.unknown, "inferno")
        v6 = cv2.resize(rough_rgb, target_size)
        cv2.putText(v6, f"6. Roughness / Coplanarity", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

        # 7. BEV Unknown & Confidence
        conf_rgb = self.render_cost_layer(1.0 - costmap.confidence, None, "Greys")
        conf_rgb[costmap.unknown] = [200, 150, 0]  # Highlight unknown in orange/blue
        v7 = cv2.resize(conf_rgb, target_size)
        cv2.putText(v7, f"7. Unknown: {costmap.unknown.mean()*100:.1f}% | Conf: {costmap.confidence.mean():.2f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

        # 8. Instantaneous Costmap
        inst_cost_rgb = self.render_cost_layer(costmap.cost, costmap.unknown, "RdYlGn_r")
        v8 = cv2.resize(inst_cost_rgb, target_size)
        cv2.putText(v8, "8. Instantaneous Costmap", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

        # 9. Temporal & Inflated Costmap
        active_map = temporal_costmap or costmap
        temp_cost_rgb = self.render_cost_layer(active_map.cost, active_map.unknown, "RdYlGn_r")
        temp_with_footprint = self.draw_vehicle_footprint(temp_cost_rgb, active_map)
        v9 = cv2.resize(temp_with_footprint, target_size)
        cv2.putText(v9, "9. Final Inflated Costmap (Rover Body)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

        # Assemble 3x3 grid (960 x 720)
        row1 = np.hstack([v1, v2, v3])
        row2 = np.hstack([v4, v5, v6])
        row3 = np.hstack([v7, v8, v9])
        grid = np.vstack([row1, row2, row3])
        return grid
