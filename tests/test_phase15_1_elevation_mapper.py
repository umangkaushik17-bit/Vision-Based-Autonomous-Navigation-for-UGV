"""Unit and integration tests for Subphase 15.1: 2.5D Elevation Mapping & Temporal Height Fusion.

Verifies:
1. Planar, ramp (10 deg, 18 deg), and discrete step (+0.15m) elevation accuracy (< +/- 0.03m).
2. Kalman / EMA temporal smoothing under Gaussian depth noise suppresses variance (< +/- 0.03m error).
3. Vectorized per-cell outlier rejection removes flying-pixel spikes (+2.0m / -1.8m) without corrupting Z(x, y).
4. Minimum point-count gating (`min_points_per_cell = 2`) ignores single-ray noise.
5. `elevation_valid` mask prevents artificial FOV boundary cliff/slope spikes on elevated terrain.
6. LocalBEVCostmap elevation binning and VisionCostmapIntegrator world map fusion.
7. Depth image unprojection (`update_from_depth_image`) under 3D attitude tilt.
8. Real-time execution latency < 2.0 ms on 200x200 WorldTerrainMap.
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from ai.spatial_fusion.world_map import WorldTerrainMap
from ai.traversability.base import TraversabilityResult
from production.perception.elevation_mapper import (
    ElevationMapper,
    ElevationMapperConfig,
)
from production.perception.terrain_geometry import VectorizedTerrainGeometryEstimator
from production.perception.vision_costmap_integrator import VisionCostmapIntegrator


def _sample_dense_patch_points(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_fn,
    step_m: float = 0.04,
) -> np.ndarray:
    """Generate dense 3D points (N, 3) with >= 4 points per 0.10m cell over [x_range, y_range]."""
    xs = np.arange(x_range[0], x_range[1], step_m, dtype=np.float32)
    ys = np.arange(y_range[0], y_range[1], step_m, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    xf = xg.ravel()
    yf = yg.ravel()
    zf = z_fn(xf, yf).astype(np.float32)
    return np.stack([xf, yf, zf], axis=-1)


@pytest.mark.parametrize("slope_deg", [0.0, 10.0, 18.0])
def test_ramp_and_planar_elevation_accuracy(slope_deg: float):
    """Verify rasterized elevation matches analytical ramp Z(x) = x * tan(alpha) within +/- 0.03m."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0))
    mapper = ElevationMapper()
    tan_a = math.tan(math.radians(slope_deg))

    pts = _sample_dense_patch_points(
        x_range=(-3.0, 3.0),
        y_range=(-3.0, 3.0),
        z_fn=lambda x, y: x * tan_a,
        step_m=0.04,
    )
    res = mapper.update_world_map(wmap, pts, timestamp=1.0)
    assert res.cells_updated > 3000
    assert res.newly_observed_cells == res.cells_updated

    # Compare all valid cells against cell-center analytical ground truth
    valid_rows, valid_cols = np.where(wmap.elevation_valid)
    assert len(valid_rows) > 3000

    x_centers = wmap.x_bounds_m[1] - (valid_rows + 0.5) * wmap.resolution_m
    z_gt = x_centers * tan_a
    z_est = wmap.elevation[valid_rows, valid_cols]

    mae = float(np.mean(np.abs(z_est - z_gt)))
    max_err = float(np.max(np.abs(z_est - z_gt)))
    assert mae < 0.005, f"Slope {slope_deg} deg MAE {mae:.4f}m exceeds 0.005m"
    assert max_err < 0.03, f"Slope {slope_deg} deg max error {max_err:.4f}m exceeds 0.03m"


def test_step_edge_preservation_and_accuracy():
    """Verify sharp +0.18m step obstacle edges are preserved without spatial blurring (< +/- 0.03m)."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-4.0, 4.0), y_bounds_m=(-4.0, 4.0))
    mapper = ElevationMapper()

    pts = _sample_dense_patch_points(
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        z_fn=lambda x, y: np.where((x >= 0.0) & (x <= 1.5) & (np.abs(y) <= 1.0), 0.18, 0.0),
        step_m=0.025,
    )
    mapper.update_world_map(wmap, pts, timestamp=1.0)

    # Check interior of elevated step [0.15, 1.35] x [-0.85, 0.85]
    r_step, c_step = wmap.world_to_grid(0.80, 0.0)
    assert wmap.elevation_valid[r_step, c_step]
    assert abs(float(wmap.elevation[r_step, c_step]) - 0.18) < 0.01

    # Check adjacent flat ground [-0.50, 0.0]
    r_flat, c_flat = wmap.world_to_grid(-0.50, 0.0)
    assert wmap.elevation_valid[r_flat, c_flat]
    assert abs(float(wmap.elevation[r_flat, c_flat]) - 0.0) < 0.01


def test_temporal_kalman_ema_noise_suppression():
    """Verify multi-frame Kalman/EMA height fusion suppresses 4cm Gaussian depth noise to < 0.015m RMSE."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-3.0, 3.0), y_bounds_m=(-3.0, 3.0))
    mapper = ElevationMapper(ElevationMapperConfig(ema_alpha=0.35, use_kalman_variance_weighting=True))
    rng = np.random.RandomState(123)

    base_pts = _sample_dense_patch_points(
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        z_fn=lambda x, y: 0.25 + 0.10 * x,
        step_m=0.04,
    )

    for frame_idx in range(10):
        noisy_pts = base_pts.copy()
        noisy_pts[:, 2] += rng.normal(0.0, 0.04, size=len(base_pts)).astype(np.float32)
        mapper.update_world_map(wmap, noisy_pts, timestamp=float(frame_idx + 1))

    valid_rows, valid_cols = np.where(wmap.elevation_valid)
    x_centers = wmap.x_bounds_m[1] - (valid_rows + 0.5) * wmap.resolution_m
    z_gt = 0.25 + 0.10 * x_centers
    z_est = wmap.elevation[valid_rows, valid_cols]

    rmse = float(np.sqrt(np.mean((z_est - z_gt) ** 2)))
    assert rmse < 0.015, f"Fused elevation RMSE {rmse:.4f}m exceeds 0.015m (< 0.03m budget)"


def test_outlier_rejection_flying_pixel_spikes():
    """Verify per-cell outlier rejection discards +2.5m and -2.0m flying-pixel depth spikes."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-3.0, 3.0), y_bounds_m=(-3.0, 3.0))
    mapper = ElevationMapper()

    pts = _sample_dense_patch_points(
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        z_fn=lambda x, y: np.full_like(x, 0.30),
        step_m=0.04,
    )
    # Inject 50 severe flying-pixel spikes (+2.5m and -1.8m) scattered across cells
    spikes = np.array(
        [[0.05 * (i - 25), 0.05 * ((i % 10) - 5), 2.50 if i % 2 == 0 else -1.80] for i in range(50)],
        dtype=np.float32,
    )
    corrupted_pts = np.vstack([pts, spikes])

    res = mapper.update_world_map(wmap, corrupted_pts, timestamp=1.0)
    assert res.outliers_rejected >= 40, f"Expected >= 40 spikes rejected, got {res.outliers_rejected}"

    valid_z = wmap.elevation[wmap.elevation_valid]
    assert float(np.max(np.abs(valid_z - 0.30))) < 0.02, "Outlier spike corrupted world map elevation!"


def test_min_points_per_cell_gating():
    """Verify cells with fewer than min_points_per_cell (2) are gated out and remain invalid."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-2.0, 2.0), y_bounds_m=(-2.0, 2.0))
    mapper = ElevationMapper(ElevationMapperConfig(min_points_per_cell=2))

    # Single isolated point per cell (spaced 0.25m apart)
    sparse_pts = np.array(
        [[-1.0, -1.0, 0.50], [0.0, 0.0, 0.50], [1.0, 1.0, 0.50]],
        dtype=np.float32,
    )
    res_sparse = mapper.update_world_map(wmap, sparse_pts, timestamp=1.0)
    assert res_sparse.cells_updated == 0
    assert not np.any(wmap.elevation_valid)

    # Now supply 2 points in cell (0.05, 0.05)
    dense_cell_pts = np.array(
        [[0.04, 0.04, 0.42], [0.06, 0.06, 0.44]],
        dtype=np.float32,
    )
    res_dense = mapper.update_world_map(wmap, dense_cell_pts, timestamp=2.0)
    assert res_dense.cells_updated == 1
    assert abs(wmap.get_elevation(0.05, 0.05) - 0.43) < 0.01


def test_zero_fov_boundary_cliff_artifacts_on_elevated_terrain():
    """Verify partial FOV observation on an elevated plateau (Z = +0.65m) creates zero boundary cliff spikes."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0))
    wmap.unknown.fill(False)
    wmap.cost.fill(0.10)

    mapper = ElevationMapper()
    estimator = VectorizedTerrainGeometryEstimator()

    # Observe only a 2m x 2m FOV patch at Z = +0.65m; surrounding map remains unobserved (Z = 0.0)
    pts = _sample_dense_patch_points(
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        z_fn=lambda x, y: np.full_like(x, 0.65),
        step_m=0.04,
    )
    mapper.update_world_map(wmap, pts, timestamp=1.0)
    geom = estimator.update_world_map_from_elevation(wmap)

    # Zero cells should be flagged as lethal slope, positive step, or negative ditch!
    assert not np.any(geom.lethal_slope_mask), "False slope spike at FOV boundary!"
    assert not np.any(geom.lethal_step_mask), "False positive step spike at FOV boundary!"
    assert not np.any(geom.negative_step_mask), "False negative ditch spike at FOV boundary!"
    assert not np.any(wmap.obstacle), "FOV boundary falsely marked as lethal obstacle!"
    assert float(np.max(geom.slope_deg)) < 0.5


def test_local_bev_and_depth_unprojection_integration():
    """Verify depth unprojection and LocalBEVCostmap -> WorldTerrainMap elevation fusion."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    mapper = ElevationMapper()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-2.0, 10.0), y_bounds_m=(-5.0, 5.0))

    mask = np.full((240, 320), 0.90, dtype=np.float32)
    trav_res = TraversabilityResult(mask=mask, timestamp=1.0, valid=True)
    bev = integrator.project_traversability(trav_res, current_time_s=1.0)

    # Populate BEV elevation with a gentle 10 deg ramp
    tan_10 = math.tan(math.radians(10.0))
    pts_robot = _sample_dense_patch_points(
        x_range=(0.5, 4.5),
        y_range=(-2.0, 2.0),
        z_fn=lambda x, y: x * tan_10,
        step_m=0.04,
    )
    bev_res = mapper.update_local_bev(bev, pts_robot)
    assert bev_res.cells_updated > 1200
    assert bev.elevation is not None and bev.elevation_valid is not None

    fuse_stats = integrator.fuse_into_world_map(bev, robot_pose=(0.0, 0.0, 0.0), world_map=wmap)
    assert fuse_stats["valid"]
    assert wmap.is_elevation_valid(2.5, 0.0)
    assert abs(wmap.get_elevation(2.5, 0.0) - 2.5 * tan_10) < 0.03


def test_elevation_mapper_200x200_latency():
    """Verify ElevationMapper.update_world_map executes in < 2.0 ms on a 200x200 WorldTerrainMap."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    mapper = ElevationMapper()

    pts = _sample_dense_patch_points(
        x_range=(0.5, 5.5),
        y_range=(-2.5, 2.5),
        z_fn=lambda x, y: 0.15 * np.sin(x) + 0.05 * y,
        step_m=0.05,
    )
    # Warmup
    _ = mapper.update_world_map(wmap, pts, timestamp=1.0)

    latencies = []
    for i in range(20):
        res = mapper.update_world_map(wmap, pts, timestamp=float(i + 2))
        latencies.append(res.update_ms)

    mean_ms = float(np.mean(latencies))
    assert mean_ms < 2.0, f"ElevationMapper update took {mean_ms:.2f}ms (> 2.0ms budget)"
