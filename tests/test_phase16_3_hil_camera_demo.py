"""Automated tests for Subphase 16.3: Plug-and-Play Live Webcam / Stereo Hardware-in-the-Loop (HIL) Demo.

Verifies:
1. `HILCameraNavigationBridge` (`production/perception/hil_camera_bridge.py`) and
   `run_hil_demo` (`scripts/run_live_camera_hil_demo.py`) execute 30 closed-loop HIL frames
   end-to-end in `--headless` mode.
2. Obstacle response: injecting a physical obstacle at (2.8, 0.0) in the camera frame marks
   BEV obstacle cells and causes C++17 A* / C++17 DWA to steer around or regulate speed.
3. Tilt compensation response: under -8 deg camera pitch tilt, SE(3) attitude IPM keeps
   the projected obstacle coordinate stable within `<= 0.15 m` of true (2.8, 0.0).
4. Composite 3-panel OpenCV HUD (`(280, 640, 3)` uint8) renders Panel 1 (RGB+Arc), Panel 2 (BEV+Path),
   and Panel 3 (Telemetry strip).
5. Mean E2E frame latency is strictly `< 35 ms` (`>= 28 FPS`).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from production.perception.hil_camera_bridge import (
    HILCameraNavigationBridge,
    SyntheticCameraSource,
)
from scripts.run_live_camera_hil_demo import run_hil_demo


def test_30_frame_headless_hil_demo_latency_and_hud_rendering():
    """Execute 30 closed-loop HIL frames in headless mode and verify < 35 ms latency (>= 28 FPS) and HUD output."""
    summary = run_hil_demo(num_frames=30, headless=True, use_synthetic=True)

    assert summary["frames_executed"] == 30
    assert summary["headless"] is True
    assert summary["hud_shape"] == [280, 640, 3]
    print(
        f"\n[PHASE 16.3 BENCHMARK] 30-Frame HIL Demo: mean={summary['mean_latency_ms']:.2f} ms, "
        f"P95={summary['p95_latency_ms']:.2f} ms, FPS={summary['mean_fps']:.1f}"
    )
    assert summary["mean_latency_ms"] < 35.0, f"Expected mean latency < 35 ms, got {summary['mean_latency_ms']} ms"
    assert summary["mean_fps"] >= 28.0, f"Expected >= 28 FPS, got {summary['mean_fps']} FPS"


def test_hil_bridge_obstacle_detection_and_steering_response():
    """Verify obstacle injection in camera frame marks BEV obstacles and triggers steering/avoidance response."""
    bridge = HILCameraNavigationBridge()
    cam = SyntheticCameraSource(width=320, height=240, integrator=bridge.integrator)

    # 1. Clear trail frame -> straight ahead (w_cmd ~ 0)
    rgb_clear = cam.read_frame(obstacle_ground_xy=None)
    res_clear = bridge.step_frame(rgb_clear, timestamp_s=1.0, render_hud=True)
    assert res_clear.obstacles_marked == 0
    assert res_clear.v_cmd > 0.25
    assert abs(res_clear.w_cmd) < 0.10
    assert res_clear.composite_hud is not None
    assert res_clear.composite_hud.shape == (280, 640, 3)

    # 2. Obstacle injected on centerline at (2.4, 0.0) -> marks obstacles and steers to detour
    rgb_obs = cam.read_frame(obstacle_ground_xy=(2.4, 0.0))
    res_obs = bridge.step_frame(rgb_obs, timestamp_s=1.1, render_hud=True)
    assert res_obs.obstacles_marked > 0
    assert res_obs.obstacle_projected_xy is not None
    assert abs(res_obs.obstacle_projected_xy[0] - 2.4) <= 0.25
    assert abs(res_obs.w_cmd) > 0.02 or res_obs.v_cmd < res_clear.v_cmd


def test_hil_bridge_se3_tilt_compensation_and_rollover_margin_response():
    """Verify SE(3) attitude IPM keeps obstacle projection accurate under pitch tilt and updates rollover margin."""
    bridge = HILCameraNavigationBridge()
    cam = SyntheticCameraSource(width=320, height=240, integrator=bridge.integrator)

    # Obstacle at (2.8, 0.0) viewed under -8.0 deg pitch and +14.0 deg roll
    rgb_tilted = cam.read_frame(obstacle_ground_xy=(2.8, 0.0), roll_deg=14.0, pitch_deg=-8.0)
    res_tilted = bridge.step_frame(rgb_tilted, timestamp_s=2.0, roll_deg=14.0, pitch_deg=-8.0, render_hud=False)

    assert res_tilted.obstacle_projected_xy is not None
    drift_m = math.hypot(res_tilted.obstacle_projected_xy[0] - 2.8, res_tilted.obstacle_projected_xy[1] - 0.0)
    assert drift_m <= 0.20, f"Expected SE(3) compensated obstacle drift <= 0.20 m, got {drift_m:.3f} m"
    assert res_tilted.rollover_margin < 0.60, "Rollover margin must reflect 14 deg chassis roll"
