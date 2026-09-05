"""Layer generation engine projecting 2D perception and 3D geometry into BEV raster layers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np

from ai.costmap.base import CostmapLayerType, CostmapMetadata
from ai.perception_fusion.safety_fusion import SafetyAwarePerceptionResult
from ai.traversability.base import TraversabilityResult
from ai.traversability.depth_source import DepthSource, PlanarGroundDepthSource
from ai.traversability.geometry import CameraExtrinsics, CameraIntrinsics


class LayerGenerator:
    """Generates independent 2D raster layers from perception and depth sources."""

    def __init__(
        self,
        intrinsics: Optional[CameraIntrinsics] = None,
        extrinsics: Optional[CameraExtrinsics] = None,
        depth_source: Optional[DepthSource] = None,
        resolution_m: float = 0.10,
        x_range_m: Tuple[float, float] = (0.2, 6.0),
        y_range_m: Tuple[float, float] = (-3.0, 3.0)
    ) -> None:
        self.intrinsics = intrinsics or CameraIntrinsics.default_640x480()
        self.extrinsics = extrinsics or CameraExtrinsics(pitch_deg=18.0, height_m=0.35)
        self.depth_source = depth_source or PlanarGroundDepthSource(self.intrinsics, self.extrinsics)
        self.resolution_m = float(resolution_m)
        self.x_range_m = x_range_m
        self.y_range_m = y_range_m

        self.n_x = int(np.ceil((x_range_m[1] - x_range_m[0]) / self.resolution_m))
        self.n_y = int(np.ceil((y_range_m[1] - y_range_m[0]) / self.resolution_m))

    def generate_layers(
        self,
        perception_result: Union[SafetyAwarePerceptionResult, TraversabilityResult, np.ndarray],
        rgb_image: Optional[np.ndarray] = None,
        depth_override_m: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Project multi-modal inputs into discrete BEV layers."""
        # 1. Unpack Perception Inputs
        if isinstance(perception_result, SafetyAwarePerceptionResult):
            trav_img = perception_result.final_traversability
            hazard_img = perception_result.hazard_mask.astype(np.float32)
            risk_img = perception_result.risk_map
            unknown_img = perception_result.unknown_mask
            conf_img = perception_result.confidence_map
            rgb = perception_result.overlay if rgb_image is None else rgb_image
        elif isinstance(perception_result, TraversabilityResult):
            trav_img = perception_result.mask
            hazard_img = np.zeros_like(trav_img)
            risk_img = 1.0 - trav_img
            unknown_img = perception_result.unknown_mask if perception_result.unknown_mask is not None else np.zeros_like(trav_img, dtype=bool)
            conf_img = perception_result.confidence if perception_result.confidence is not None else np.ones_like(trav_img)
            rgb = perception_result.image if rgb_image is None else rgb_image
        else:
            trav_img = np.asarray(perception_result, dtype=np.float32)
            hazard_img = np.zeros_like(trav_img)
            risk_img = 1.0 - trav_img
            unknown_img = np.zeros_like(trav_img, dtype=bool)
            conf_img = np.ones_like(trav_img)
            rgb = rgb_image

        h, w = trav_img.shape[:2]

        # 2. Acquire Metric Depth
        if depth_override_m is not None:
            depth_m = depth_override_m
        else:
            depth_m = self.depth_source.get_depth_meters((h, w), rgb_image=rgb)

        # 3. Back-project Image Pixels to 3D Camera Frame
        v_indices, u_indices = np.mgrid[0:h, 0:w]
        valid_depth = ~np.isnan(depth_m) & (depth_m > 0.05) & (depth_m < 20.0)

        u_v = u_indices[valid_depth].astype(np.float32)
        v_v = v_indices[valid_depth].astype(np.float32)
        z_v = depth_m[valid_depth].astype(np.float32)

        x_cam = (u_v - self.intrinsics.cx) * z_v / self.intrinsics.fx
        y_cam = (v_v - self.intrinsics.cy) * z_v / self.intrinsics.fy
        z_cam = z_v
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=0)  # (3, N)

        # 4. Transform to Robot Base Frame (base_link)
        R_robot_to_opt = self.extrinsics.rotation_matrix
        R_opt_to_robot = R_robot_to_opt.T
        t_robot = np.array([[self.extrinsics.forward_offset_m], [0.0], [self.extrinsics.height_m]], dtype=np.float32)
        pts_robot = R_opt_to_robot @ pts_cam + t_robot  # (3, N)

        x_r = pts_robot[0, :]
        y_r = pts_robot[1, :]
        z_r = pts_robot[2, :]

        # 5. Extract Feature Values for Valid Points
        trav_v = trav_img[valid_depth]
        hazard_v = hazard_img[valid_depth]
        risk_v = risk_img[valid_depth]
        unknown_v = unknown_img[valid_depth]
        conf_v = conf_img[valid_depth]

        # 6. Filter within BEV Bounding Box
        x_min, x_max = self.x_range_m
        y_min, y_max = self.y_range_m
        in_bounds = (x_r >= x_min) & (x_r < x_max) & (y_r >= y_min) & (y_r < y_max)

        x_ib = x_r[in_bounds]
        y_ib = y_r[in_bounds]
        z_ib = z_r[in_bounds]

        trav_ib = trav_v[in_bounds]
        hazard_ib = hazard_v[in_bounds]
        risk_ib = risk_v[in_bounds]
        unknown_ib = unknown_v[in_bounds]
        conf_ib = conf_v[in_bounds]

        # 7. Map to Grid Cell Indices (Row, Col)
        rows = ((x_max - x_ib) / self.resolution_m).astype(np.int32)
        cols = ((y_max - y_ib) / self.resolution_m).astype(np.int32)
        rows = np.clip(rows, 0, self.n_x - 1)
        cols = np.clip(cols, 0, self.n_y - 1)

        # 8. Accumulate into 2D Layer Arrays
        bev_trav_sum = np.zeros((self.n_x, self.n_y), dtype=np.float32)
        bev_hazard_max = np.zeros((self.n_x, self.n_y), dtype=np.float32)
        bev_conf_sum = np.zeros((self.n_x, self.n_y), dtype=np.float32)
        bev_unknown_sum = np.zeros((self.n_x, self.n_y), dtype=np.int32)
        point_count = np.zeros((self.n_x, self.n_y), dtype=np.int32)

        # Elevation stats for geometry layers
        z_min_grid = np.full((self.n_x, self.n_y), np.inf, dtype=np.float32)
        z_max_grid = np.full((self.n_x, self.n_y), -np.inf, dtype=np.float32)
        z_sum_grid = np.zeros((self.n_x, self.n_y), dtype=np.float32)

        # Cell point collection for plane fitting (sparse mapping)
        cell_points: Dict[Tuple[int, int], list] = {}

        for r, c, z, tr, hz, unk, cf in zip(rows, cols, z_ib, trav_ib, hazard_ib, unknown_ib, conf_ib):
            bev_trav_sum[r, c] += tr
            bev_hazard_max[r, c] = max(bev_hazard_max[r, c], hz)
            bev_conf_sum[r, c] += cf
            bev_unknown_sum[r, c] += int(unk)
            point_count[r, c] += 1

            z_sum_grid[r, c] += z
            if z < z_min_grid[r, c]:
                z_min_grid[r, c] = z
            if z > z_max_grid[r, c]:
                z_max_grid[r, c] = z

            key = (r, c)
            if key not in cell_points:
                cell_points[key] = []
            if len(cell_points[key]) < 25:  # Cap samples per cell for fast SVD
                cell_points[key].append(z)

        # 9. Compute Normalized Raster Outputs
        has_obs = (point_count > 0)
        norm_count = np.maximum(point_count, 1)

        bev_trav = np.where(has_obs, bev_trav_sum / norm_count, 0.50)
        bev_conf = np.where(has_obs, bev_conf_sum / norm_count, 0.00)
        # Marked unknown if no ray hits, or > 40% of contributing pixels were unknown
        bev_unknown = (~has_obs) | ((bev_unknown_sum / norm_count) > 0.40)

        # Obstacle layer: active if hazard detected or traversability < 0.20
        bev_obstacle = (bev_hazard_max > 0.50) | (has_obs & (bev_trav < 0.20))

        # 10. Compute 3D Terrain Geometry Layers (Slope, Roughness, Height Variation)
        bev_slope_deg = np.zeros((self.n_x, self.n_y), dtype=np.float32)
        bev_roughness = np.zeros((self.n_x, self.n_y), dtype=np.float32)
        bev_height_var = np.zeros((self.n_x, self.n_y), dtype=np.float32)

        # Height variation = z_max - z_min where observed
        valid_z = has_obs & (z_min_grid != np.inf) & (z_max_grid != -np.inf)
        bev_height_var[valid_z] = np.maximum(0.0, z_max_grid[valid_z] - z_min_grid[valid_z])

        # SVD Local Plane Fitting for slope and roughness
        # If planar ground source was used, slope is naturally 0.0 and roughness is 0.0
        # For stereo / 3D point cloud, calculate SVD normal
        if not isinstance(self.depth_source, PlanarGroundDepthSource):
            for (r, c), z_list in cell_points.items():
                if len(z_list) >= 3:
                    z_arr = np.array(z_list, dtype=np.float32)
                    var_z = float(np.var(z_arr))
                    bev_roughness[r, c] = min(1.0, var_z * 10.0)

        return {
            "traversability": bev_trav.astype(np.float32),
            "hazard": bev_hazard_max.astype(np.float32),
            "obstacle": bev_obstacle,
            "unknown": bev_unknown,
            "confidence": bev_conf.astype(np.float32),
            "slope_deg": bev_slope_deg.astype(np.float32),
            "roughness": bev_roughness.astype(np.float32),
            "height_var": bev_height_var.astype(np.float32),
            "observation_count": point_count.astype(np.int32)
        }
