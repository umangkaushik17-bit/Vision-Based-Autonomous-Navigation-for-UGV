"""Unit tests for Subphase 15.0: Metric Depth Foundation & Uneven Terrain Generator.

Verifies:
1. `AnalyticalTerrainDepthSource` (`ai/traversability/depth_source.py`) and
   `DepthUnprojector` (`production/perception/depth_unprojector.py`) reconstruct 3D terrain
   elevation Z(x, y) within `< +/-0.02 m` across `[0.5, 5.0] m` over flat, ramp, step, and ditch geometries.
2. Vectorized depth unprojection executes in `< 1.5 ms` on CPU.
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from ai.traversability.depth_source import AnalyticalTerrainDepthSource
from ai.traversability.geometry import CameraExtrinsics, CameraIntrinsics
from production.perception.depth_unprojector import DepthUnprojector


def test_depth_unprojection_accuracy_across_flat_ramp_step_and_ditch():
    """Verify reconstructed 3D points match analytical terrain Z(x, y) within < +/-0.02 m over 0.5-5.0 m."""
    intr = CameraIntrinsics(fx=381.3614, fy=381.3614, cx=320.0, cy=240.0, width=640, height=480)
    extr = CameraExtrinsics(pitch_deg=15.0, roll_deg=0.0, yaw_deg=0.0, height_m=0.525, forward_offset_m=0.15)
    unprojector = DepthUnprojector(intrinsics=intr, extrinsics=extr, min_depth_m=0.5, max_depth_m=5.0)

    geometries = {
        "flat": lambda x, y: np.zeros_like(x, dtype=np.float32),
        "ramp_10deg": lambda x, y: np.maximum(0.0, (x - 1.5) * math.tan(math.radians(10.0))).astype(np.float32),
        "step_18cm": lambda x, y: np.where(x >= 2.5, 0.18, 0.0).astype(np.float32),
        "ditch_15cm": lambda x, y: np.where((x >= 2.2) & (x <= 3.2), -0.15, 0.0).astype(np.float32),
    }

    for name, fn in geometries.items():
        src = AnalyticalTerrainDepthSource(intrinsics=intr, extrinsics=extr, terrain_fn=fn, roll_rad=0.04, pitch_rad=-0.05)
        depth_map = src.get_depth_meters((120, 160), roll_rad=0.04, pitch_rad=-0.05)
        pts_3d, _ = unprojector.unproject(depth_map, roll_rad=0.04, pitch_rad=-0.05, stride=1)

        in_range = (pts_3d[:, 0] >= 0.5) & (pts_3d[:, 0] <= 5.0)
        pts_r = pts_3d[in_range]
        expected_z = fn(pts_r[:, 0], pts_r[:, 1])
        max_err = float(np.max(np.abs(pts_r[:, 2] - expected_z)))
        assert max_err < 0.02, f"Geometry {name} max Z unprojection error {max_err:.4f} m >= 0.02 m"


def test_depth_unprojector_latency_under_1_5ms():
    """Verify DepthUnprojector.unproject executes in < 1.5 ms mean on CPU."""
    unprojector = DepthUnprojector()
    depth_map = np.full((120, 160), 2.5, dtype=np.float32)

    for _ in range(5):
        unprojector.unproject(depth_map, roll_rad=0.05, pitch_rad=-0.08)

    times_ms = []
    for _ in range(40):
        t0 = time.perf_counter()
        unprojector.unproject(depth_map, roll_rad=0.05, pitch_rad=-0.08)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    assert mean_ms < 1.5, f"Expected DepthUnprojector latency < 1.5 ms, got {mean_ms:.3f} ms"
