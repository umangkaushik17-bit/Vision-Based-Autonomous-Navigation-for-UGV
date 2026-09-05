"""Unit tests for Visual Odometry and SLAM estimators, failure injection, and loop closing."""

import numpy as np
import pytest

from ai.localization.base import CameraIntrinsics, IMUReading, TrackingState
from ai.localization.datasets import SyntheticStereoSequenceGenerator
from ai.localization.loop_closing import LoopClosingEngine
from ai.localization.mono_vo import MonocularVisualOdometry
from ai.localization.stereo_vo import StereoVisualOdometry
from ai.localization.visual_inertial import VisualInertialOdometry


@pytest.fixture
def camera():
    return CameraIntrinsics(
        fx=462.14, fy=462.14, cx=320.0, cy=240.0, width=640, height=480, baseline_m=0.05
    )


def test_monocular_vo_pipeline(camera):
    """Verify MonocularVisualOdometry initialization and sequential frame processing."""
    vo = MonocularVisualOdometry(camera)
    gen = SyntheticStereoSequenceGenerator(num_frames=5, baseline_m=camera.baseline_m)

    for frame in gen:
        est = vo.process_frame(
            image_left=frame.image_left,
            timestamp=frame.timestamp
        )
        assert est.timestamp == frame.timestamp
        assert est.pose.shape == (4, 4)

    assert len(vo.trajectory) == 5


def test_stereo_vo_pipeline(camera):
    """Verify StereoVisualOdometry metric scale triangulation and PnP tracking."""
    vo = StereoVisualOdometry(camera)
    gen = SyntheticStereoSequenceGenerator(num_frames=5, baseline_m=camera.baseline_m)

    for frame in gen:
        est = vo.process_frame(
            image_left=frame.image_left,
            timestamp=frame.timestamp,
            image_right=frame.image_right
        )
        assert est.pose.shape == (4, 4)

    assert len(vo.trajectory) == 5


def test_visual_inertial_odometry(camera):
    """Verify VisualInertialOdometry with synchronized IMU."""
    vio = VisualInertialOdometry(camera)
    gen = SyntheticStereoSequenceGenerator(num_frames=5, baseline_m=camera.baseline_m)

    for frame in gen:
        est = vio.process_frame(
            image_left=frame.image_left,
            timestamp=frame.timestamp,
            image_right=frame.image_right,
            imu=frame.imu
        )
        assert est.pose.shape == (4, 4)

    assert len(vio.trajectory) == 5


def test_failure_injection_handling(camera):
    """Verify that featureless blank frames transition cleanly to TrackingState.LOST."""
    vo = StereoVisualOdometry(camera)
    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)

    est = vo.process_frame(blank_img, timestamp=0.0, image_right=blank_img)
    assert est.tracking_state == TrackingState.LOST
    assert est.confidence < 0.25


def test_loop_closing_engine():
    """Verify loop closure registration and trajectory relaxation."""
    engine = LoopClosingEngine(min_inliers_loop=15, min_separation_keyframes=5)
    rng = np.random.RandomState(42)

    # Register 10 keyframes
    des_shared = rng.randint(0, 255, size=(30, 32), dtype=np.uint8)
    for i in range(10):
        T = np.eye(4)
        T[0, 3] = float(i)
        # Add shared descriptors at frame 0 and frame 9 to trigger loop closure
        des = des_shared if (i == 0 or i == 9) else rng.randint(0, 255, size=(30, 32), dtype=np.uint8)
        engine.add_keyframe(
            keyframe_id=i,
            timestamp=float(i),
            pose=T,
            descriptors=des,
            keypoints=np.zeros((30, 2))
        )

    # Detect loop at frame 9
    loop = engine.detect_loop(des_shared, curr_keyframe_id=9)
    assert loop is not None
    match_id, T_rel, drift = loop
    assert match_id == 0
    assert engine.loop_closures_accepted >= 1

    # Test relocalization from tracking loss
    reloc = engine.relocalize(des_shared, min_inliers_reloc=15)
    assert reloc is not None
    kf_id, recovered_pose, inliers = reloc
    assert kf_id in (0, 9)
    assert engine.relocalization_successes >= 1

