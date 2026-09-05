"""Unit tests for synthetic outdoor sensor generator."""

import numpy as np
import pytest
from scripts.generate_synthetic_outdoor_data import generate_synthetic_frame

def test_generate_synthetic_frame():
    width = 320
    height = 240
    rgb, gt, depth, imu_meta = generate_synthetic_frame(
        step=0, total_steps=10, width=width, height=height
    )

    # RGB verification
    assert rgb.shape == (height, width, 3)
    assert rgb.dtype == np.uint8

    # Traversability GT verification
    assert gt.shape == (height, width)
    assert gt.dtype == np.uint8
    unique_classes = np.unique(gt)
    # Must only contain 0 (SAFE), 1 (UNKNOWN), or 2 (UNSAFE)
    assert set(unique_classes).issubset({0, 1, 2})
    assert 0 in unique_classes, "Must have safe trail points"

    # Depth verification
    assert depth.shape == (height, width)
    assert depth.dtype == np.float32
    assert np.all(depth > 0.0), "All depth values must be strictly positive"

    # IMU & metadata verification
    assert "sequence_id" in imu_meta
    assert "ground_truth_pose" in imu_meta
    assert "imu" in imu_meta
    assert len(imu_meta["imu"]["linear_acceleration_mps2"]) == 3
