"""Unit tests for Phase 7 MATLAB data interchange and validation modules."""

from pathlib import Path
import numpy as np
import pytest
import scipy.io as sio

from scripts.verify_matlab_math import (
    test_camera_extrinsics_math as check_camera_extrinsics,
    test_camera_intrinsics_math as check_camera_intrinsics,
    test_bev_geometry_math as check_bev_geometry,
    test_terrain_slope_math as check_terrain_slope,
    test_ate_rpe_math as check_ate_rpe,
)


def test_matlab_exported_data_integrity():
    """Verify that all native .mat files exist, are readable, and non-empty."""
    data_dir = Path("matlab/data")
    assert (data_dir / "camera_geometry.mat").exists()
    assert (data_dir / "costmap_scenarios.mat").exists()
    assert (data_dir / "planner_trajectories.mat").exists()
    assert (data_dir / "synthetic_slam_trajectories.mat").exists()
    assert (data_dir / "data_manifest.json").exists()

    # Load camera geometry
    cam = sio.loadmat(str(data_dir / "camera_geometry.mat"))
    assert "K" in cam and cam["K"].shape == (3, 3)
    assert "R_robot_to_camera" in cam and cam["R_robot_to_camera"].shape == (3, 3)

    # Load costmap scenarios
    cm = sio.loadmat(str(data_dir / "costmap_scenarios.mat"))
    assert "nominal_flat" in cm and "terrain_detour" in cm

    # Load SLAM trajectories
    slam = sio.loadmat(str(data_dir / "synthetic_slam_trajectories.mat"))
    assert "ground_truth" in slam and slam["ground_truth"].shape[1] == 4


def test_matlab_mathematical_parity():
    """Run cross-language mathematical parity tests."""
    assert check_camera_intrinsics()
    assert check_camera_extrinsics()
    assert check_bev_geometry()
    assert check_terrain_slope()
    assert check_ate_rpe()


def test_matlab_scripts_structure():
    """Verify presence of all required MATLAB modules and scripts."""
    required_scripts = [
        "matlab/utils/check_matlab_environment.m",
        "matlab/utils/define_tolerances.m",
        "matlab/audit/audit_project_artifacts.m",
        "matlab/scripts/audit_project_artifacts.m",
        "matlab/calibration/validate_intrinsics.m",
        "matlab/calibration/validate_extrinsics.m",
        "matlab/calibration/analyze_distortion.m",
        "matlab/geometry/check_coordinate_conventions.m",
        "matlab/geometry/transform_robot_to_camera.m",
        "matlab/geometry/transform_camera_to_robot.m",
        "matlab/bev/validate_bev_geometry.m",
        "matlab/bev/validate_bev_resolution.m",
        "matlab/bev/compute_bev_transform_error.m",
        "matlab/terrain/validate_plane_fitting.m",
        "matlab/terrain/validate_slope_estimation.m",
        "matlab/terrain/validate_step_height.m",
        "matlab/costmap/cross_validate_costmap.m",
        "matlab/costmap/compute_costmap_statistics.m",
        "matlab/costmap/validate_obstacle_inflation.m",
        "matlab/costmap/validate_temporal_fusion.m",
        "matlab/costmap/validate_motion_compensation.m",
        "matlab/planning/analyze_global_paths.m",
        "matlab/planning/validate_terrain_detour.m",
        "matlab/planning/analyze_dynamic_obstacle.m",
        "matlab/planning/audit_dwa_scoring.m",
        "matlab/slamming/align_trajectories.m",
        "matlab/slamming/compute_ate.m",
        "matlab/slamming/compute_rpe.m",
        "matlab/slamming/test_synthetic_slam_metrics.m",
        "matlab/metrics/compare_python_matlab_metrics.m",
        "matlab/metrics/compute_tracking_error.m",
        "matlab/tests/run_matlab_test_suite.m",
        "matlab/scripts/run_all_validation.m"
    ]
    for script_rel in required_scripts:
        p = Path(script_rel)
        assert p.exists(), f"Missing required MATLAB script: {script_rel}"
        assert p.stat().st_size > 0, f"MATLAB script is empty: {script_rel}"
