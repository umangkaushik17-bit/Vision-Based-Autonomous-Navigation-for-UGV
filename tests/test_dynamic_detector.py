"""Unit tests for Milestone 13.2 — Detection Interface.

Validates:
1. Base detector interface constraints and parameter validation
2. Empty input handling (returns empty list)
3. Single detection generation and packaging
4. Multiple detections handling with ephemeral IDs
5. Confidence propagation and minimum confidence filtering
6. Timestamp propagation
7. Bounding geometry accuracy (centers, sizes, yaws, OBB corners)
8. Coordinate / frame metadata preservation
9. Invalid parameter rejections (negative timestamps, invalid confidences)
10. Deterministic repeated execution under fixed random seeds
11. Direct ingestion into Phase 13.1 data model methods (corners, AABB, transform_to_frame, serialization)
12. Ground-truth isolation audit (no GT poses, future velocities, or GT IDs in runtime detect())
13. Sensor dropout simulation
14. Measurement noise simulation
15. Vision detector extensibility stub conformity
"""

import inspect
import math
import numpy as np
import pytest

from ai.planning.dynamic.data_model import DynamicObstacleDetection
from ai.planning.dynamic.detector_interface import (
    BaseDynamicObstacleDetector,
    SyntheticDynamicObstacleDetector,
    SyntheticObstacleScript,
    VisionDynamicObstacleDetector,
)


class TestBaseDetectorInterface:
    def test_cannot_instantiate_abstract_base_class(self):
        with pytest.raises(TypeError):
            BaseDynamicObstacleDetector()

    def test_base_parameter_validation(self):
        class DummyDetector(BaseDynamicObstacleDetector):
            def detect(self, frame, timestamp_s):
                return []

        # Empty frame_id
        with pytest.raises(ValueError, match="frame_id"):
            DummyDetector(frame_id="")

        # Confidence out of bounds
        with pytest.raises(ValueError, match="min_confidence"):
            DummyDetector(min_confidence=-0.1)
        with pytest.raises(ValueError, match="min_confidence"):
            DummyDetector(min_confidence=1.1)

        det = DummyDetector(frame_id="base_link", min_confidence=0.4)
        assert det.frame_id == "base_link"
        assert det.min_confidence == 0.4


class TestSyntheticDetectorExecution:
    def test_empty_detections(self):
        detector = SyntheticDynamicObstacleDetector(frame_id="base_link")
        # No scripts registered -> returns empty list
        detections = detector.detect(frame=None, timestamp_s=0.5)
        assert isinstance(detections, list)
        assert len(detections) == 0

    def test_single_detection_packaging(self):
        detector = SyntheticDynamicObstacleDetector(frame_id="base_link")
        script = SyntheticObstacleScript(
            obstacle_id="pedestrian_1",
            size=(0.6, 0.5),
            position_fn=lambda t: (2.0 + 0.5 * t, 1.0),
            yaw_fn=lambda t: 0.0,
            confidence=0.92,
            active_time_range=(0.0, 5.0),
        )
        detector.add_obstacle_script(script)

        # Detect at t = 2.0 s -> x = 2.0 + 0.5 * 2.0 = 3.0, y = 1.0
        dets = detector.detect(frame=None, timestamp_s=2.0)
        assert len(dets) == 1
        d = dets[0]

        assert isinstance(d, DynamicObstacleDetection)
        assert d.timestamp == 2.0
        assert d.frame_id == "base_link"
        assert math.isclose(d.bbox_center[0], 3.0)
        assert math.isclose(d.bbox_center[1], 1.0)
        assert d.bbox_size == (0.6, 0.5)
        assert math.isclose(d.bbox_yaw, 0.0)
        assert math.isclose(d.confidence, 0.92)

    def test_multiple_detections_with_ephemeral_ids(self):
        detector = SyntheticDynamicObstacleDetector(frame_id="base_link")
        s1 = SyntheticObstacleScript(
            obstacle_id="obs_A",
            size=(1.0, 0.8),
            position_fn=lambda t: (3.0, 0.0),
            confidence=0.85,
        )
        s2 = SyntheticObstacleScript(
            obstacle_id="obs_B",
            size=(0.5, 0.5),
            position_fn=lambda t: (5.0, -1.0),
            confidence=0.90,
        )
        detector.add_obstacle_script(s1)
        detector.add_obstacle_script(s2)

        dets = detector.detect(frame=None, timestamp_s=1.0)
        assert len(dets) == 2
        # Verify IDs are unique and ephemeral (do not match obstacle_id script names)
        assert dets[0].detection_id != dets[1].detection_id
        assert "obs_A" not in dets[0].detection_id
        assert "obs_B" not in dets[1].detection_id

    def test_confidence_propagation_and_filtering(self):
        detector = SyntheticDynamicObstacleDetector(min_confidence=0.60)
        s_low = SyntheticObstacleScript(
            obstacle_id="low_conf",
            size=(1.0, 1.0),
            position_fn=lambda t: (2.0, 2.0),
            confidence=0.45,  # below 0.60
        )
        s_high = SyntheticObstacleScript(
            obstacle_id="high_conf",
            size=(1.0, 1.0),
            position_fn=lambda t: (4.0, 4.0),
            confidence=0.80,  # above 0.60
        )
        detector.add_obstacle_script(s_low)
        detector.add_obstacle_script(s_high)

        dets = detector.detect(frame=None, timestamp_s=1.0)
        assert len(dets) == 1
        assert math.isclose(dets[0].confidence, 0.80)

    def test_active_time_range_window(self):
        detector = SyntheticDynamicObstacleDetector()
        script = SyntheticObstacleScript(
            obstacle_id="timed",
            size=(1.0, 1.0),
            position_fn=lambda t: (1.0, 1.0),
            active_time_range=(2.0, 4.0),
        )
        detector.add_obstacle_script(script)

        assert len(detector.detect(frame=None, timestamp_s=1.0)) == 0
        assert len(detector.detect(frame=None, timestamp_s=2.5)) == 1
        assert len(detector.detect(frame=None, timestamp_s=4.5)) == 0

    def test_timestamp_propagation(self):
        detector = SyntheticDynamicObstacleDetector()
        detector.add_obstacle_script(
            SyntheticObstacleScript("s", (1, 1), lambda t: (1, 1))
        )
        for t in [0.0, 0.125, 1.5, 10.75]:
            dets = detector.detect(frame=None, timestamp_s=t)
            assert len(dets) == 1
            assert math.isclose(dets[0].timestamp, t)

    def test_invalid_detection_timestamp_rejection(self):
        detector = SyntheticDynamicObstacleDetector()
        with pytest.raises(ValueError, match="timestamp_s"):
            detector.detect(frame=None, timestamp_s=-0.01)

    def test_deterministic_repeated_execution(self):
        # With noise, identical seed must produce identical outputs
        script = SyntheticObstacleScript(
            obstacle_id="noisy_obs",
            size=(1.0, 1.0),
            position_fn=lambda t: (3.0, 2.0),
            measurement_noise_std=0.15,
            dropout_probability=0.20,
        )

        det1 = SyntheticDynamicObstacleDetector(random_seed=12345)
        det1.add_obstacle_script(script)

        det2 = SyntheticDynamicObstacleDetector(random_seed=12345)
        det2.add_obstacle_script(script)

        timestamps = [0.1 * i for i in range(15)]
        results1 = [det1.detect(None, t) for t in timestamps]
        results2 = [det2.detect(None, t) for t in timestamps]

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert len(r1) == len(r2)
            if len(r1) > 0:
                assert math.isclose(r1[0].bbox_center[0], r2[0].bbox_center[0])
                assert math.isclose(r1[0].bbox_center[1], r2[0].bbox_center[1])

    def test_sensor_dropout_simulation(self):
        # 100% dropout should emit zero detections
        detector = SyntheticDynamicObstacleDetector()
        script = SyntheticObstacleScript(
            obstacle_id="dropping",
            size=(1.0, 1.0),
            position_fn=lambda t: (1.0, 1.0),
            dropout_probability=1.0,
        )
        detector.add_obstacle_script(script)
        assert len(detector.detect(None, 1.0)) == 0


class TestDataModelCompatibility:
    def test_detector_output_usable_by_data_model_methods(self):
        detector = SyntheticDynamicObstacleDetector(frame_id="base_link")
        script = SyntheticObstacleScript(
            obstacle_id="car",
            size=(2.0, 1.0),
            position_fn=lambda t: (4.0, 0.0),
            yaw_fn=lambda t: math.pi / 4.0,
            generate_cluster_points=True,
            cluster_point_count=8,
        )
        detector.add_obstacle_script(script)

        dets = detector.detect(frame=None, timestamp_s=0.5)
        assert len(dets) == 1
        det = dets[0]

        # 1. Bounding corners
        corners = det.get_corners(margin_m=0.1)
        assert corners.shape == (4, 2)

        # 2. AABB bounds
        x_min, y_min, x_max, y_max = det.get_axis_aligned_bbox(margin_m=0.0)
        assert x_min < x_max
        assert y_min < y_max

        # 3. Serialization round-trip
        serialized = det.to_dict()
        recovered = DynamicObstacleDetection.from_dict(serialized)
        assert recovered.detection_id == det.detection_id
        assert math.isclose(recovered.bbox_yaw, det.bbox_yaw)
        assert len(recovered.raw_cluster_points) == 8

        # 4. Coordinate frame transformation (base_link -> odom)
        robot_pose = (10.0, 5.0, 0.0)
        det_odom = det.transform_to_frame("odom", robot_pose)
        assert det_odom.frame_id == "odom"
        assert math.isclose(det_odom.bbox_center[0], 14.0)
        assert math.isclose(det_odom.bbox_center[1], 5.0)


class TestGroundTruthIsolationAudit:
    def test_detect_method_signature_strict_isolation(self):
        # Verify detect() signature accepts ONLY (self, frame, timestamp_s)
        sig = inspect.signature(BaseDynamicObstacleDetector.detect)
        param_names = list(sig.parameters.keys())
        assert param_names == ["self", "frame", "timestamp_s"], (
            f"Ground-truth violation: Base detect() signature has unauthorized parameters: {param_names}"
        )

        sig_synth = inspect.signature(SyntheticDynamicObstacleDetector.detect)
        synth_params = list(sig_synth.parameters.keys())
        assert synth_params == ["self", "frame", "timestamp_s"], (
            f"Ground-truth violation: Synthetic detect() signature has unauthorized parameters: {param_params}"
        )

    def test_no_ground_truth_leakage_in_emitted_detections(self):
        detector = SyntheticDynamicObstacleDetector()
        detector.add_obstacle_script(
            SyntheticObstacleScript(
                obstacle_id="secret_gt_agent_99",
                size=(1.0, 0.5),
                position_fn=lambda t: (1.0 + t, 0.0),
            )
        )
        dets = detector.detect(frame=None, timestamp_s=1.0)
        d = dets[0]

        # Ensure no GT velocity, GT ID, or GT registry object is attached to DynamicObstacleDetection
        assert not hasattr(d, "velocity")
        assert not hasattr(d, "gt_track_id")
        assert not hasattr(d, "ground_truth_pose")
        assert "secret_gt_agent_99" not in d.detection_id


class TestVisionExtensionStub:
    def test_vision_detector_conforms_to_interface(self):
        vision_det = VisionDynamicObstacleDetector(
            model_weights_path=None,
            frame_id="base_link",
            min_confidence=0.50,
        )
        assert isinstance(vision_det, BaseDynamicObstacleDetector)
        # Empty frame returns empty detections list
        assert vision_det.detect(frame=None, timestamp_s=1.0) == []
