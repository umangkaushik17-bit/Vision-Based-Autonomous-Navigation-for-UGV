"""Automated tests for Subphase 16.1: Adverse Illumination, Glare, Shadow & Sensor-Dropout Resilience.

Verifies:
1. `PhotometricResilienceProcessor` (`production/perception/photometric_resilience.py`) converts incoming
   RGB frames to CIE-LAB, applies adaptive LAB-CLAHE normalization when frame brightness L_bar not in [65, 190]
   or contrast sigma_L < 28, and restores normalized RGB in `< 1.8 ms` on CPU.
2. Specular Glare & Deep Shadow Confidence Map `C_photo(u, v) in [0.1, 1.0]` accurately detects saturated
   specular glare (`L > 245, S < 20`) and cast shadow bands, achieving `< 2%` traversability false-obstacle
   rate across `+60%` glare, `-60%` low-light, and sharp diagonal shadow bands while preserving true obstacles.
3. `VisionCostmapIntegrator` attenuates low-confidence glare/shadow ray dips so zero permanent ghost obstacles
   are latched into `WorldTerrainMap` from sunlight glare or tree shadows.
4. During a 3.0-second Stereo VIO + Camera dropout, `MultiSensorEKF` transitions seamlessly into
   wheel-odometry + 6-axis IMU gravity-stabilized dead-reckoning with `< 0.05 m` drift over 3.0s, while
   `SafetyGovernor` enforces conservative velocity clamping (`degraded_max_v = 0.35 m/s`) with zero NaNs or collisions.
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from ai.planning.safety_governor import SafetyGovernor, SafetyGovernorConfig
from ai.spatial_fusion.world_map import WorldTerrainMap
from ai.traversability.base import TraversabilityResult
from production.localization.ekf_fusion import MultiSensorEKF
from production.localization.types import (
    IMUMeasurement,
    StereoVIOMeasurement,
    WheelOdometryMeasurement,
)
from production.perception.photometric_resilience import (
    PhotometricResilienceConfig,
    PhotometricResilienceProcessor,
)
from production.perception.vision_costmap_integrator import VisionCostmapIntegrator


def _build_synthetic_outdoor_trail_rgb(h: int = 240, w: int = 320) -> np.ndarray:
    """Create a realistic textured outdoor dirt/grass trail RGB frame in (H, W, 3) uint8."""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:h, 0:w]
    # Textured dirt/grass trail base with natural spatial gradient and micro-texture
    r = np.clip(110 + 18 * np.sin(xx * 0.08) + 14 * np.cos(yy * 0.06) + rng.normal(0, 8, (h, w)), 60, 175)
    g = np.clip(132 + 20 * np.sin(xx * 0.08) + 16 * np.cos(yy * 0.06) + rng.normal(0, 8, (h, w)), 70, 195)
    b = np.clip(88 + 15 * np.sin(xx * 0.08) + 12 * np.cos(yy * 0.06) + rng.normal(0, 8, (h, w)), 45, 155)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def test_adaptive_lab_clahe_normalization_and_latency_under_1_8ms():
    """Verify adaptive LAB-CLAHE triggers on +/-60% lighting shifts and executes in < 1.8 ms."""
    processor = PhotometricResilienceProcessor()
    base_rgb = _build_synthetic_outdoor_trail_rgb(240, 320)

    # 1. +60% harsh midday glare shift (mean L > 190)
    glare_rgb = np.clip(base_rgb.astype(np.float32) * 1.60 + 55.0, 0, 255).astype(np.uint8)
    res_glare = processor.process(glare_rgb, timestamp_s=1.0)
    assert res_glare.mean_l_before > 190.0
    assert res_glare.clahe_applied is True
    assert 65.0 <= res_glare.mean_l_after <= 190.0

    # 2. -60% dusk low-light shift (mean L < 65)
    lowlight_rgb = np.clip(base_rgb.astype(np.float32) * 0.40, 0, 255).astype(np.uint8)
    res_low = processor.process(lowlight_rgb, timestamp_s=1.1)
    assert res_low.mean_l_before < 65.0
    assert res_low.clahe_applied is True
    assert 65.0 <= res_low.mean_l_after <= 190.0
    assert res_low.contrast_l_after > res_low.contrast_l_before

    # 3. Low-contrast fog/haze frame (sigma_L < 28)
    haze_rgb = np.clip(128.0 + (base_rgb.astype(np.float32) - 128.0) * 0.20, 0, 255).astype(np.uint8)
    res_haze = processor.process(haze_rgb, timestamp_s=1.2)
    assert res_haze.contrast_l_before < 28.0
    assert res_haze.clahe_applied is True
    assert res_haze.contrast_l_after >= 25.0

    # 4. Latency benchmark (< 1.8 ms mean budget on 240x320 RGB)
    for _ in range(5):
        processor.process(glare_rgb, timestamp_s=2.0)

    latencies_ms = []
    for i in range(40):
        t0 = time.perf_counter()
        processor.process(glare_rgb, timestamp_s=2.1 + i * 0.03)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    print(f"\n[PHASE 16.1 BENCHMARK] PhotometricResilienceProcessor.process(): mean={mean_ms:.2f} ms, P95={p95_ms:.2f} ms")
    assert mean_ms < 1.8, f"Expected mean LAB-CLAHE latency < 1.8 ms, got {mean_ms:.2f} ms"


def test_specular_glare_and_shadow_confidence_and_false_obstacle_rate_under_2pct():
    """Verify < 2% false-obstacle rate under +60% glare, -60% low-light, and sharp diagonal shadow bands."""
    processor = PhotometricResilienceProcessor()
    base_rgb = _build_synthetic_outdoor_trail_rgb(240, 320)
    h, w = base_rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]

    # Case A: +60% glare + saturated specular glare patch (L > 245, saturation S < 20)
    glare_rgb = np.clip(base_rgb.astype(np.float32) * 1.60 + 42.0, 0, 255).astype(np.uint8)
    spec_patch = (yy >= 120) & (yy <= 170) & (xx >= 110) & (xx <= 210)
    glare_rgb[spec_patch] = np.array([252, 252, 250], dtype=np.uint8)

    trav_glare, photo_glare = processor.evaluate_outdoor_traversability(glare_rgb, timestamp_s=1.0)
    assert np.any(photo_glare.glare_mask[spec_patch]), "Must detect saturated specular glare pixels"
    assert float(np.min(photo_glare.confidence_map[spec_patch])) == pytest.approx(0.10, abs=0.06)
    false_obs_glare = float(np.mean(trav_glare.mask < 0.30))
    assert false_obs_glare < 0.02, f"Glare false-obstacle rate {false_obs_glare:.4f} >= 2%"

    # Case B: -60% low-light across entire trail
    low_rgb = np.clip(base_rgb.astype(np.float32) * 0.40, 0, 255).astype(np.uint8)
    trav_low, photo_low = processor.evaluate_outdoor_traversability(low_rgb, timestamp_s=1.1)
    false_obs_low = float(np.mean(trav_low.mask < 0.30))
    assert false_obs_low < 0.02, f"Low-light false-obstacle rate {false_obs_low:.4f} >= 2%"

    # Case C: Sharp diagonal tree shadow band across the trail (luminance attenuated by 65%, same chromaticity)
    shadow_rgb = base_rgb.copy()
    diag_band = np.abs((xx - 0.8 * yy) - 60.0) < 22.0
    shadow_rgb[diag_band] = (shadow_rgb[diag_band].astype(np.float32) * 0.32).astype(np.uint8)

    trav_shadow, photo_shadow = processor.evaluate_outdoor_traversability(shadow_rgb, timestamp_s=1.2)
    assert np.any(photo_shadow.shadow_mask[diag_band]), "Must detect cast diagonal shadow band"
    assert float(np.min(photo_shadow.confidence_map[diag_band])) <= 0.20
    false_obs_shadow = float(np.mean(trav_shadow.mask < 0.30))
    assert false_obs_shadow < 0.02, f"Shadow band false-obstacle rate {false_obs_shadow:.4f} >= 2%"

    # Verify a real physical obstacle (bright red hazard crate) is still 100% detected as an obstacle
    obs_rgb = shadow_rgb.copy()
    crate_mask = (yy >= 140) & (yy <= 185) & (xx >= 140) & (xx <= 180)
    obs_rgb[crate_mask] = np.array([220, 25, 25], dtype=np.uint8)
    trav_with_obs, _ = processor.evaluate_outdoor_traversability(obs_rgb, timestamp_s=1.3)
    assert float(np.mean(trav_with_obs.mask[crate_mask] < 0.30)) > 0.90, "Real obstacle must remain detected"


def test_vision_costmap_integrator_suppresses_glare_and_shadow_ghost_obstacles():
    """Verify VisionCostmapIntegrator does not latch ghost obstacles from low-confidence glare/shadow bands."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-2.0, 10.0), y_bounds_m=(-5.0, 5.0))
    wmap.unknown.fill(False)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)

    # Raw traversability has a false-obstacle dip (tau = 0.12 < 0.30) along a shadow band at v in [130, 160],
    # but photometric confidence map marks that band as low-confidence shadow (C_photo = 0.15).
    raw_mask = np.full((240, 320), 0.88, dtype=np.float32)
    raw_mask[130:160, 80:240] = 0.12
    conf_map = np.ones((240, 320), dtype=np.float32)
    conf_map[130:160, 80:240] = 0.15

    trav_res = TraversabilityResult(
        mask=raw_mask,
        confidence=conf_map,
        timestamp=1.0,
        valid=True,
    )

    bev = integrator.project_traversability(trav_res, current_time_s=1.0)
    fusion_stats = integrator.fuse_into_world_map(bev, robot_pose=(0.0, 0.0, 0.0), world_map=wmap)

    assert fusion_stats["obstacles_marked"] == 0, "Glare/shadow band must not mark ghost obstacles in WorldTerrainMap"
    assert int(np.sum(wmap.obstacle)) == 0


def test_ekf_3s_vio_outage_dead_reckoning_and_camera_dropout_speed_clamping():
    """Verify < 0.05 m EKF drift over a 3.0s VIO outage and 0.35 m/s SafetyGovernor clamping on camera dropout."""
    ekf = MultiSensorEKF()
    ekf.reset(initial_pose=(0.0, 0.0, 0.0))
    governor = SafetyGovernor(SafetyGovernorConfig(degraded_max_v_mps=0.35, camera_stale_timeout_s=0.50))
    photo_proc = PhotometricResilienceProcessor()

    dt = 0.02  # 50 Hz
    t = 0.0
    true_x = 0.0
    true_y = 0.0
    true_yaw = 0.0
    v_cmd = 0.40
    w_cmd = 0.05

    # Phase 1: 1.0s nominal multi-sensor operation (VIO + Wheel + 6-axis IMU active)
    for _ in range(50):
        t += dt
        true_yaw += w_cmd * dt
        true_x += v_cmd * math.cos(true_yaw) * dt
        true_y += v_cmd * math.sin(true_yaw) * dt

        ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=v_cmd, vyaw=w_cmd))
        ekf.update_imu(
            IMUMeasurement(
                timestamp=t,
                angular_velocity=np.array([0.0, 0.0, w_cmd], dtype=np.float64),
                linear_acceleration=np.array([0.0, 0.0, 9.80665], dtype=np.float64),
            )
        )
        ekf.update_stereo_vio(
            StereoVIOMeasurement(
                timestamp=t,
                x=true_x,
                y=true_y,
                z=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=true_yaw,
                vx=v_cmd * math.cos(true_yaw),
                vy=v_cmd * math.sin(true_yaw),
                vz=0.0,
                inliers=110,
                tracking_state="TRACKING_OK",
                valid=True,
            )
        )

    state_pre = ekf.get_state(t)
    assert state_pre.tracking_state == "FUSED_TRACKING_OK"
    last_cam_ts = t

    # Phase 2: 3.0-second Stereo VIO + Camera dropout (150 steps at 50 Hz; only Wheel + IMU active)
    v_dropout = 0.35
    w_dropout = 0.0
    for _ in range(150):
        t += dt
        true_x += v_dropout * math.cos(true_yaw) * dt
        true_y += v_dropout * math.sin(true_yaw) * dt

        ekf.update_wheel_odom(WheelOdometryMeasurement(timestamp=t, vx=v_dropout, vyaw=w_dropout))
        ekf.update_imu(
            IMUMeasurement(
                timestamp=t,
                angular_velocity=np.array([0.0, 0.0, w_dropout], dtype=np.float64),
                linear_acceleration=np.array([0.0, 0.0, 9.80665], dtype=np.float64),
            )
        )
        # VIO reports LOST
        ekf.update_stereo_vio(
            StereoVIOMeasurement(
                timestamp=t,
                x=0.0,
                y=0.0,
                z=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                tracking_state="LOST",
                valid=False,
            )
        )

    state_dropout = ekf.get_state(t)
    assert state_dropout.tracking_state == "COASTING_IMU_WHEEL"
    assert not np.isnan(state_dropout.covariance).any()
    assert not np.isinf(state_dropout.covariance).any()

    pos_drift_m = math.hypot(state_dropout.x - true_x, state_dropout.y - true_y)
    assert pos_drift_m < 0.05, f"Expected < 0.05 m EKF dead-reckoning drift over 3.0s VIO outage, got {pos_drift_m:.4f} m"

    # Verify camera dropout watchdog and SafetyGovernor degraded velocity clamping (0.35 m/s)
    is_cam_drop, cam_age_s = photo_proc.check_camera_dropout(current_time_s=t, frame_timestamp_s=last_cam_ts)
    assert is_cam_drop is True
    assert cam_age_s == pytest.approx(3.0, abs=0.05)

    gov_v, gov_w, gov_dec, gov_diag = governor.evaluate(
        requested_v=0.65,
        requested_w=0.10,
        robot_pose=(state_dropout.x, state_dropout.y, state_dropout.yaw),
        hazards=[],
        current_time_s=t,
        camera_frame_age_s=cam_age_s,
        camera_dropout=is_cam_drop,
    )
    assert gov_v == pytest.approx(0.35)
    assert "CAMERA_DROPOUT_DEGRADED_SPEED" in gov_dec
    assert gov_diag["camera_stale_or_dropout"] is True
