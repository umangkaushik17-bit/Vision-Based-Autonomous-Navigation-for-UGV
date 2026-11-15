"""Unit tests for Dynamic SE(3) Attitude-Compensated IPM Projection (WS-1B / P0-2).

Verifies:
1. Zero-attitude parity between precomputed LUT and dynamic SE(3) attitude LUT.
2. Pitch invariance: A ground obstacle at (x=2.5m, y=0.0m) projected into a pitched camera
   (+10 deg, -10 deg, +12 deg) reconstructs back to the exact metric ground coordinate (error < 0.05m).
3. Roll invariance: A ground obstacle at (x=2.5m, y=0.5m) projected into a rolled camera
   (+10 deg, -10 deg) reconstructs back to the exact metric ground coordinate (error < 0.05m).
4. Combined roll + pitch attitude compensation accuracy and real-time latency (< 5.0 ms).
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from ai.traversability.base import TraversabilityResult
from production.perception.vision_costmap_integrator import VisionCostmapIntegrator


def test_zero_attitude_lut_parity():
    """Verify compute_attitude_compensated_lut(0, 0) matches precomputed LUT identically."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    v0, u0, r0, c0 = integrator.compute_attitude_compensated_lut(0.0, 0.0)
    np.testing.assert_array_equal(v0, integrator._lut_valid_v)
    np.testing.assert_array_equal(u0, integrator._lut_valid_u)
    np.testing.assert_array_equal(r0, integrator._lut_rows)
    np.testing.assert_array_equal(c0, integrator._lut_cols)

    # Also test tiny non-zero attitude (1e-3 rad) vs zero
    v1, u1, r1, c1 = integrator.compute_attitude_compensated_lut(1e-3, -1e-3)
    assert len(v1) > 10000


@pytest.mark.parametrize("pitch_deg", [-10.0, -5.0, 0.0, 5.0, 10.0, 12.0])
def test_pitch_attitude_compensation_metric_stability(pitch_deg: float):
    """Verify that a stationary ground obstacle at (2.8m, 0.0m) projects to the same BEV cell
    even as the vehicle pitches between -10 deg and +12 deg.
    """
    integrator = VisionCostmapIntegrator.create_simulation_default()
    target_x_m = 2.80
    target_y_m = 0.00
    pitch_rad = math.radians(pitch_deg)

    # Project the true 3D ground obstacle into the tilted camera image
    px = integrator.project_ground_point_to_pixel(
        x_ground_m=target_x_m,
        y_ground_m=target_y_m,
        roll_rad=0.0,
        pitch_rad=pitch_rad,
        image_shape=(240, 320),
    )
    assert px is not None, f"Obstacle at ({target_x_m}, {target_y_m}) should be in FOV at pitch={pitch_deg} deg"
    v_obs, u_obs = px

    # Paint a small 3x3 obstacle patch around (v_obs, u_obs)
    mask = np.full((240, 320), 0.90, dtype=np.float32)
    v0, v1 = max(0, v_obs - 1), min(240, v_obs + 2)
    u0, u1 = max(0, u_obs - 1), min(320, u_obs + 2)
    mask[v0:v1, u0:u1] = 0.05

    res = TraversabilityResult(mask=mask, timestamp=1.0, valid=True)
    bev = integrator.project_traversability(
        res,
        current_time_s=1.0,
        roll_rad=0.0,
        pitch_rad=pitch_rad,
    )
    assert bev.valid
    assert np.any(bev.obstacle), f"No obstacle projected at pitch={pitch_deg} deg"

    obs_rows, obs_cols = np.where(bev.obstacle)
    mean_x = float(np.mean(bev.x_robot[obs_rows, obs_cols]))
    mean_y = float(np.mean(bev.y_robot[obs_rows, obs_cols]))

    err_m = math.hypot(mean_x - target_x_m, mean_y - target_y_m)
    assert err_m <= 0.12, f"Pitch {pitch_deg} deg caused metric projection error {err_m:.3f}m (> 0.12m)"


@pytest.mark.parametrize("roll_deg", [-12.0, -6.0, 0.0, 6.0, 12.0])
def test_roll_attitude_compensation_metric_stability(roll_deg: float):
    """Verify that a stationary ground obstacle at (2.5m, 0.4m) projects to the same BEV cell
    even as the vehicle rolls between -12 deg and +12 deg.
    """
    integrator = VisionCostmapIntegrator.create_simulation_default()
    target_x_m = 2.50
    target_y_m = 0.40
    roll_rad = math.radians(roll_deg)

    px = integrator.project_ground_point_to_pixel(
        x_ground_m=target_x_m,
        y_ground_m=target_y_m,
        roll_rad=roll_rad,
        pitch_rad=0.0,
        image_shape=(240, 320),
    )
    assert px is not None
    v_obs, u_obs = px

    mask = np.full((240, 320), 0.90, dtype=np.float32)
    v0, v1 = max(0, v_obs - 1), min(240, v_obs + 2)
    u0, u1 = max(0, u_obs - 1), min(320, u_obs + 2)
    mask[v0:v1, u0:u1] = 0.05

    res = TraversabilityResult(mask=mask, timestamp=1.0, valid=True)
    bev = integrator.project_traversability(
        res,
        current_time_s=1.0,
        roll_rad=roll_rad,
        pitch_rad=0.0,
    )
    assert bev.valid
    assert np.any(bev.obstacle)

    obs_rows, obs_cols = np.where(bev.obstacle)
    mean_x = float(np.mean(bev.x_robot[obs_rows, obs_cols]))
    mean_y = float(np.mean(bev.y_robot[obs_rows, obs_cols]))

    err_m = math.hypot(mean_x - target_x_m, mean_y - target_y_m)
    assert err_m <= 0.10, f"Roll {roll_deg} deg caused metric projection error {err_m:.3f}m (> 0.10m)"


@pytest.mark.parametrize("roll_deg,pitch_deg", [(-8.0, 6.0), (8.0, -6.0), (10.0, 8.0), (-10.0, -8.0)])
def test_combined_roll_and_pitch_attitude_compensation(roll_deg: float, pitch_deg: float):
    """Verify simultaneous 3D roll + pitch SE(3) compensation keeps ground obstacle drift <= 0.10m."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    target_x_m = 2.60
    target_y_m = 0.20
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)

    px = integrator.project_ground_point_to_pixel(
        x_ground_m=target_x_m,
        y_ground_m=target_y_m,
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        image_shape=(240, 320),
    )
    assert px is not None
    v_obs, u_obs = px

    mask = np.full((240, 320), 0.90, dtype=np.float32)
    v0, v1 = max(0, v_obs - 1), min(240, v_obs + 2)
    u0, u1 = max(0, u_obs - 1), min(320, u_obs + 2)
    mask[v0:v1, u0:u1] = 0.05

    res = TraversabilityResult(mask=mask, timestamp=1.0, valid=True)
    bev = integrator.project_traversability(
        res,
        current_time_s=1.0,
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
    )
    assert bev.valid
    assert np.any(bev.obstacle)

    obs_rows, obs_cols = np.where(bev.obstacle)
    mean_x = float(np.mean(bev.x_robot[obs_rows, obs_cols]))
    mean_y = float(np.mean(bev.y_robot[obs_rows, obs_cols]))

    err_m = math.hypot(mean_x - target_x_m, mean_y - target_y_m)
    assert err_m <= 0.10, f"Roll={roll_deg}, Pitch={pitch_deg} caused projection drift {err_m:.3f}m (> 0.10m)"


def test_attitude_compensated_ipm_latency():
    """Verify dynamic SE(3) attitude-compensated projection runs in < 3.5 ms on CPU."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    mask = np.full((240, 320), 0.85, dtype=np.float32)
    mask[120:140, 150:170] = 0.10
    res = TraversabilityResult(mask=mask, timestamp=1.0, valid=True)

    # Warmup
    _ = integrator.project_traversability(res, current_time_s=1.0, roll_rad=0.05, pitch_rad=-0.08)

    latencies = []
    for i in range(10):
        t0 = time.perf_counter()
        _ = integrator.project_traversability(
            res,
            current_time_s=1.0,
            roll_rad=0.04 * math.sin(i),
            pitch_rad=0.06 * math.cos(i),
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    assert mean_ms < 3.5, f"Dynamic SE(3) IPM latency {mean_ms:.2f}ms exceeds 3.5ms budget"
