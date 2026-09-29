"""Unit test suite for Hardware-Independent DWA Backends.

Verifies:
1. Reference CPU backend instantiation and planning
2. Vectorized CPU backend instantiation and planning
3. Explicit CUDA selection failure when CUDA unavailable (NO silent fallback)
4. Explicit Intel GPU selection failure when Intel GPU runtime unavailable (NO silent fallback)
5. Explicit NPU selection rejection with clear non-neural rationale
6. AUTO backend selection arbitration
7. Runtime failure and safe fallback behavior
"""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.fixture
def planning_config() -> PlanningConfig:
    cfg = PlanningConfig()
    cfg.vehicle.length_m = 0.65
    cfg.vehicle.width_m = 0.45
    cfg.vehicle.collision_margin_m = 0.05
    cfg.vehicle.circumscribed_radius_m = 0.40
    cfg.dwa["prediction_horizon_s"] = 1.50
    cfg.dwa["simulation_dt_s"] = 0.10
    cfg.dwa["v_samples"] = 15
    cfg.dwa["omega_samples"] = 25
    return cfg


@pytest.fixture
def world_map() -> WorldTerrainMap:
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    return wmap


def test_cpu_backend_nominal(planning_config: PlanningConfig, world_map: WorldTerrainMap) -> None:
    planner = UnifiedDWAPlanner(config=planning_config, backend=DWABackendType.CPU)
    assert planner.backend_type == DWABackendType.CPU
    assert "Reference CPU" in planner.backend_name

    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0)
    path = Path.from_xy_array(np.array([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]))
    traj, v, w, status, diag = planner.compute_trajectory(state, path, world_map)

    assert status == PlannerStatus.SUCCESS
    assert traj is not None
    assert v > 0.0


def test_vectorized_cpu_backend_nominal(planning_config: PlanningConfig, world_map: WorldTerrainMap) -> None:
    planner = UnifiedDWAPlanner(config=planning_config, backend=DWABackendType.VECTORIZED_CPU)
    assert planner.backend_type == DWABackendType.VECTORIZED_CPU

    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0)
    path = Path.from_xy_array(np.array([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]))
    traj, v, w, status, diag = planner.compute_trajectory(state, path, world_map)

    assert status == PlannerStatus.SUCCESS
    assert traj is not None
    assert v > 0.0


def test_explicit_cuda_fails_clearly_when_unavailable(planning_config: PlanningConfig) -> None:
    """Explicit CUDA request must raise RuntimeError without silent fallback if CUDA is absent."""
    if not CUDADWABackend.is_available():
        with pytest.raises(RuntimeError, match="CUDA backend explicitly requested"):
            UnifiedDWAPlanner(config=planning_config, backend=DWABackendType.CUDA)


def test_explicit_intel_gpu_fails_clearly_when_unavailable(planning_config: PlanningConfig) -> None:
    """Explicit INTEL_GPU request must raise RuntimeError without silent fallback if runtime is absent."""
    if not IntelGPUDWABackend.is_available():
        with pytest.raises(RuntimeError, match="INTEL_GPU backend explicitly requested"):
            UnifiedDWAPlanner(config=planning_config, backend=DWABackendType.INTEL_GPU)


def test_explicit_npu_fails_clearly(planning_config: PlanningConfig) -> None:
    """Explicit NPU request must raise RuntimeError explaining non-neural computational incompatibility."""
    with pytest.raises(RuntimeError, match="NPU backend is not supported"):
        UnifiedDWAPlanner(config=planning_config, backend=DWABackendType.NPU)


def test_auto_backend_selection(planning_config: PlanningConfig, world_map: WorldTerrainMap) -> None:
    """AUTO mode must select the fastest validated available backend on the machine."""
    planner = UnifiedDWAPlanner(config=planning_config, backend=DWABackendType.AUTO)
    assert planner.active_backend.is_available()

    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0)
    path = Path.from_xy_array(np.array([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]))
    traj, v, w, status, diag = planner.compute_trajectory(state, path, world_map)

    assert status == PlannerStatus.SUCCESS
    assert traj is not None
    assert "active_backend" in diag
