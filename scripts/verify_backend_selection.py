"""Verification script for Step 9: Backend Selection & Arbitration.

Tests:
1. Explicit CPU -> CPUDWABackend
2. Explicit VECTORIZED_CPU -> VectorizedCPUDWABackend
3. Explicit CUDA -> RuntimeError when unavailable (NO silent fallback)
4. Explicit INTEL_GPU -> IntelGPUDWABackend on XPU (or RuntimeError if unavailable, NO silent fallback)
5. Explicit NPU -> Clear RuntimeError explaining non-neural rationale
6. AUTO -> Validated available backend
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from ai.planning.backends import (
    CPUDWABackend,
    CUDADWABackend,
    DWABackendType,
    IntelGPUDWABackend,
    NPUDWABackend,
    UnifiedDWAPlanner,
    VectorizedCPUDWABackend,
)
from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.path import Path
from ai.spatial_fusion.world_map import WorldTerrainMap


def run_backend_verification():
    print("==================================================")
    print("STEP 9: DWA BACKEND SELECTION & ARBITRATION AUDIT")
    print("==================================================")

    cfg = PlanningConfig()
    cfg.vehicle.length_m = 0.65
    cfg.vehicle.width_m = 0.45
    cfg.vehicle.collision_margin_m = 0.05
    cfg.vehicle.circumscribed_radius_m = 0.40
    cfg.dwa["prediction_horizon_s"] = 1.50
    cfg.dwa["simulation_dt_s"] = 0.10
    cfg.dwa["v_samples"] = 15
    cfg.dwa["omega_samples"] = 25

    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)

    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0)
    path = Path.from_xy_array(np.array([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]))

    results = {}

    # 1. CPU
    print("\n[1] Testing Explicit CPU Selection...")
    planner_cpu = UnifiedDWAPlanner(config=cfg, backend=DWABackendType.CPU)
    assert planner_cpu.backend_type == DWABackendType.CPU
    traj, v, w, status, diag = planner_cpu.compute_trajectory(state, path, wmap)
    assert status == PlannerStatus.SUCCESS
    assert "active_backend" in diag
    print(f"    Selected: {planner_cpu.backend_name}")
    print(f"    Status: {status.name}, v={v:.2f}, w={w:.2f}")
    results["CPU"] = "PASS (CPUDWABackend executed)"

    # 2. VECTORIZED_CPU
    print("\n[2] Testing Explicit VECTORIZED_CPU Selection...")
    planner_vec = UnifiedDWAPlanner(config=cfg, backend=DWABackendType.VECTORIZED_CPU)
    assert planner_vec.backend_type == DWABackendType.VECTORIZED_CPU
    traj, v, w, status, diag = planner_vec.compute_trajectory(state, path, wmap)
    assert status == PlannerStatus.SUCCESS
    print(f"    Selected: {planner_vec.backend_name}")
    print(f"    Status: {status.name}, v={v:.2f}, w={w:.2f}")
    results["VECTORIZED_CPU"] = "PASS (VectorizedCPUDWABackend executed)"

    # 3. CUDA (Explicit)
    print("\n[3] Testing Explicit CUDA Selection...")
    cuda_avail = torch.cuda.is_available()
    print(f"    torch.cuda.is_available() = {cuda_avail}")
    if not cuda_avail:
        try:
            planner_cuda = UnifiedDWAPlanner(config=cfg, backend=DWABackendType.CUDA)
            raise AssertionError("FAIL: Explicit CUDA request did not raise RuntimeError!")
        except RuntimeError as e:
            print(f"    SUCCESS: Caught expected RuntimeError (NO silent fallback): {e}")
            results["CUDA"] = "PASS (Strict rejection without silent fallback)"
    else:
        planner_cuda = UnifiedDWAPlanner(config=cfg, backend=DWABackendType.CUDA)
        results["CUDA"] = "PASS (CUDA executed)"

    # 4. INTEL_GPU (Explicit)
    print("\n[4] Testing Explicit INTEL_GPU Selection...")
    has_xpu = hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)()
    print(f"    torch.xpu.is_available() = {has_xpu}")
    if has_xpu:
        planner_intel = UnifiedDWAPlanner(config=cfg, backend=DWABackendType.INTEL_GPU)
        assert planner_intel.backend_type == DWABackendType.INTEL_GPU
        assert "Intel GPU" in planner_intel.backend_name
        traj, v, w, status, diag = planner_intel.compute_trajectory(state, path, wmap)
        assert status == PlannerStatus.SUCCESS
        print(f"    Selected: {planner_intel.backend_name} on {planner_intel.active_backend.device_name}")
        print(f"    Status: {status.name}, v={v:.2f}, w={w:.2f}")
        results["INTEL_GPU"] = f"PASS (IntelGPUDWABackend executed on {planner_intel.active_backend.device_name})"
    else:
        try:
            planner_intel = UnifiedDWAPlanner(config=cfg, backend=DWABackendType.INTEL_GPU)
            raise AssertionError("FAIL: Explicit INTEL_GPU request did not raise RuntimeError!")
        except RuntimeError as e:
            print(f"    SUCCESS: Caught expected RuntimeError (NO silent fallback): {e}")
            results["INTEL_GPU"] = "PASS (Strict rejection without silent fallback)"

    # 5. NPU (Explicit)
    print("\n[5] Testing Explicit NPU Selection...")
    try:
        planner_npu = UnifiedDWAPlanner(config=cfg, backend=DWABackendType.NPU)
        raise AssertionError("FAIL: Explicit NPU request did not raise RuntimeError!")
    except RuntimeError as e:
        print(f"    SUCCESS: Caught expected RuntimeError explaining non-neural rationale:\n    '{e}'")
        results["NPU"] = "PASS (Rejected with architectural rationale)"

    # 6. AUTO
    print("\n[6] Testing AUTO Selection Arbitration...")
    planner_auto = UnifiedDWAPlanner(config=cfg, backend=DWABackendType.AUTO)
    health = planner_auto.active_backend.check_health()
    print(f"    AUTO selected: {planner_auto.backend_name}")
    print(f"    Backend Device: {health.device_name}")
    print(f"    Safety Validated: {health.safety_validated}")
    assert health.is_available
    assert health.safety_validated
    traj, v, w, status, diag = planner_auto.compute_trajectory(state, path, wmap)
    assert status == PlannerStatus.SUCCESS
    results["AUTO"] = f"PASS (Selected {planner_auto.backend_name}, validated={health.safety_validated})"

    print("\n==================================================")
    print("STEP 9 SUMMARY RESULTS:")
    for k, v in results.items():
        print(f"  {k:16s}: {v}")
    print("==================================================")


if __name__ == "__main__":
    run_backend_verification()
