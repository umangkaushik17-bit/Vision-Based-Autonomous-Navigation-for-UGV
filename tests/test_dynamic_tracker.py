"""Unit tests for Milestone 13.3 — Multi-Frame Tracking.

Validates:
A. Empty detection stream handling
B. First detection creates NEW track
C. Confirmed detection transitions to TRACKING
D. Same object across multiple frames retains persistent track ID
E. Two distinct objects retain separate persistent track IDs
F. Moving objects at different positions correctly associated
G. One missed frame keeps track alive (no premature drop)
H. Repeated misses cause transition to DEGRADED
I. Prolonged misses cause transition to LOST and EXPIRED
J. Reacquisition behavior: DEGRADED/LOST returns to TRACKING upon fresh detection
K. Confidence handling: detections below min_detection_confidence filtered out
L. Timestamp handling: non-negative enforcement and update
M. Coordinate frame handling: requires robot_pose if input frame != target_frame
N. Deterministic repeated execution
O. Multiple simultaneous tracks (> 2 concurrent targets)
P. Crossing objects association stress test
Q. Association spatial gating: detections outside gate create new tracks
R. Ground-truth leakage audit: tracker does not read GT poses or simulation IDs
S. Config parameter validation
"""

import inspect
import math
import numpy as np
import pytest

from ai.planning.dynamic.association import HungarianAssociation
from ai.planning.dynamic.data_model import DynamicObstacleDetection, TrackedObstacleArray
from ai.planning.dynamic.detector_interface import (
    SyntheticDynamicObstacleDetector,
    SyntheticObstacleScript,
)
from ai.planning.dynamic.tracker import (
    MultiFrameTracker,
    TrackStatus,
    TrackerConfig,
)


@pytest.fixture
def default_config():
    return TrackerConfig(
        max_association_distance_m=1.50,
        hits_to_confirm=3,
        misses_to_degrade=1,
        misses_to_lost=3,
        misses_to_expire=5,
        max_coast_duration_s=2.0,
        min_detection_confidence=0.30,
        target_frame="odom",
    )


class TestTrackerLifecycleAndAssociation:
    def test_A_empty_detection_stream(self, default_config):
        tracker = MultiFrameTracker(default_config)
        arr = tracker.update([], timestamp_s=0.1)
        assert isinstance(arr, TrackedObstacleArray)
        assert len(arr) == 0
        assert tracker.active_track_count == 0

    def test_B_first_detection_creates_new_track(self, default_config):
        tracker = MultiFrameTracker(default_config)
        det = DynamicObstacleDetection("det_0", 0.1, "odom", (2.0, 1.0), (1.0, 0.5), 0.0, 0.9)
        arr = tracker.update([det], timestamp_s=0.1)

        assert len(arr) == 1
        track = arr[0]
        assert track.track_id == 0
        assert track.status == TrackStatus.NEW.value
        assert track.is_confirmed is False
        assert track.track_age == 1
        assert track.coast_cycles == 0
        assert math.isclose(track.x, 2.0)
        assert math.isclose(track.y, 1.0)

    def test_C_confirmation_to_tracking(self, default_config):
        # hits_to_confirm = 3
        tracker = MultiFrameTracker(default_config)

        # Frame 1: Hit 1 -> NEW
        det1 = DynamicObstacleDetection("d1", 0.1, "odom", (2.0, 1.0), (1.0, 0.5), 0.0, 0.9)
        arr1 = tracker.update([det1], timestamp_s=0.1)
        assert arr1[0].status == TrackStatus.NEW.value
        assert arr1[0].is_confirmed is False

        # Frame 2: Hit 2 -> Still NEW
        det2 = DynamicObstacleDetection("d2", 0.2, "odom", (2.05, 1.0), (1.0, 0.5), 0.0, 0.9)
        arr2 = tracker.update([det2], timestamp_s=0.2)
        assert arr2[0].status == TrackStatus.NEW.value
        assert arr2[0].is_confirmed is False

        # Frame 3: Hit 3 -> TRACKING (Confirmed!)
        det3 = DynamicObstacleDetection("d3", 0.3, "odom", (2.10, 1.0), (1.0, 0.5), 0.0, 0.9)
        arr3 = tracker.update([det3], timestamp_s=0.3)
        assert arr3[0].status == TrackStatus.TRACKING.value
        assert arr3[0].is_confirmed is True
        assert arr3[0].track_id == 0

    def test_D_same_object_retains_persistent_id(self, default_config):
        tracker = MultiFrameTracker(default_config)
        track_ids = []
        for i in range(10):
            t = 0.1 * (i + 1)
            # Moving along X at 0.5 m/s
            x = 2.0 + 0.5 * t
            det = DynamicObstacleDetection(f"d_{i}", t, "odom", (x, 0.0), (1.0, 0.5), 0.0, 0.95)
            arr = tracker.update([det], timestamp_s=t)
            assert len(arr) == 1
            track_ids.append(arr[0].track_id)

        # All frames must have the exact same ID (0)
        assert len(set(track_ids)) == 1
        assert track_ids[0] == 0

    def test_E_two_objects_retain_separate_ids(self, default_config):
        tracker = MultiFrameTracker(default_config)
        # Two distinct objects separated by 4 meters
        for i in range(5):
            t = 0.1 * (i + 1)
            d1 = DynamicObstacleDetection(f"d1_{i}", t, "odom", (2.0, 2.0), (0.8, 0.8), 0.0, 0.9)
            d2 = DynamicObstacleDetection(f"d2_{i}", t, "odom", (2.0, -2.0), (0.8, 0.8), 0.0, 0.9)
            arr = tracker.update([d1, d2], timestamp_s=t)
            assert len(arr) == 2
            ids = {arr[0].track_id, arr[1].track_id}
            assert ids == {0, 1}

    def test_F_moving_objects_correctly_associated(self, default_config):
        tracker = MultiFrameTracker(default_config)
        # Target starts at (1, 1) and moves to (1.5, 1.0) then (2.0, 1.0)
        tracker.update([DynamicObstacleDetection("d1", 0.1, "odom", (1.0, 1.0), (1, 1), 0, 0.9)], 0.1)
        tracker.update([DynamicObstacleDetection("d2", 0.2, "odom", (1.5, 1.0), (1, 1), 0, 0.9)], 0.2)
        arr = tracker.update([DynamicObstacleDetection("d3", 0.3, "odom", (2.0, 1.0), (1, 1), 0, 0.9)], 0.3)

        assert len(arr) == 1
        assert arr[0].track_id == 0
        assert math.isclose(arr[0].x, 2.0)
        assert len(arr[0].history) == 3

    def test_G_one_missed_frame_keeps_track_alive(self, default_config):
        tracker = MultiFrameTracker(default_config)
        # Confirm track (3 frames)
        for i in range(3):
            t = 0.1 * (i + 1)
            det = DynamicObstacleDetection(f"d_{i}", t, "odom", (2.0, 1.0), (1, 1), 0, 0.9)
            tracker.update([det], timestamp_s=t)

        # 1 missed frame at t = 0.4s
        arr_missed = tracker.update([], timestamp_s=0.4)
        assert len(arr_missed) == 1
        assert arr_missed[0].track_id == 0
        assert arr_missed[0].coast_cycles == 1

    def test_H_repeated_misses_cause_degraded_and_lost(self, default_config):
        # hits_to_confirm=3, misses_to_degrade=1, misses_to_lost=3, misses_to_expire=5
        tracker = MultiFrameTracker(default_config)
        for i in range(3):
            t = 0.1 * (i + 1)
            tracker.update([DynamicObstacleDetection(f"d_{i}", t, "odom", (2.0, 1.0), (1, 1), 0, 0.9)], t)

        # 1st miss (t = 0.4s) -> DEGRADED
        arr1 = tracker.update([], timestamp_s=0.4)
        assert arr1[0].status == TrackStatus.DEGRADED.value
        assert arr1[0].coast_cycles == 1

        # 2nd miss (t = 0.5s) -> DEGRADED
        arr2 = tracker.update([], timestamp_s=0.5)
        assert arr2[0].status == TrackStatus.DEGRADED.value
        assert arr2[0].coast_cycles == 2

        # 3rd miss (t = 0.6s) -> LOST (misses_to_lost = 3)
        arr3 = tracker.update([], timestamp_s=0.6)
        assert arr3[0].status == TrackStatus.LOST.value
        assert arr3[0].coast_cycles == 3

    def test_I_prolonged_misses_expire_and_prune(self, default_config):
        tracker = MultiFrameTracker(default_config)
        # Confirm track
        for i in range(3):
            t = 0.1 * (i + 1)
            tracker.update([DynamicObstacleDetection(f"d_{i}", t, "odom", (2.0, 1.0), (1, 1), 0, 0.9)], t)

        # Miss 5 consecutive frames (misses_to_expire = 5)
        for m in range(4):
            tracker.update([], timestamp_s=0.4 + 0.1 * m)

        # 5th miss at t = 0.8s -> EXPIRED and pruned from output
        arr_final = tracker.update([], timestamp_s=0.8)
        assert len(arr_final) == 0
        assert tracker.active_track_count == 0

    def test_J_reacquisition_returns_to_tracking(self, default_config):
        tracker = MultiFrameTracker(default_config)
        for i in range(3):
            t = 0.1 * (i + 1)
            tracker.update([DynamicObstacleDetection(f"d_{i}", t, "odom", (2.0, 1.0), (1, 1), 0, 0.9)], t)

        # 2 misses -> DEGRADED
        tracker.update([], timestamp_s=0.4)
        arr_deg = tracker.update([], timestamp_s=0.5)
        assert arr_deg[0].status == TrackStatus.DEGRADED.value

        # Reacquired detection at t = 0.6s
        det_reacquired = DynamicObstacleDetection("d_re", 0.6, "odom", (2.05, 1.0), (1, 1), 0, 0.9)
        arr_re = tracker.update([det_reacquired], timestamp_s=0.6)

        assert len(arr_re) == 1
        assert arr_re[0].track_id == 0
        assert arr_re[0].status == TrackStatus.TRACKING.value
        assert arr_re[0].coast_cycles == 0
        assert arr_re[0].is_confirmed is True

    def test_K_confidence_filtering(self, default_config):
        tracker = MultiFrameTracker(default_config)
        # Confidence 0.20 is below min_detection_confidence (0.30)
        low_conf_det = DynamicObstacleDetection("low", 0.1, "odom", (1.0, 1.0), (1, 1), 0, 0.20)
        arr = tracker.update([low_conf_det], timestamp_s=0.1)
        assert len(arr) == 0
        assert tracker.active_track_count == 0

    def test_L_timestamp_handling(self, default_config):
        tracker = MultiFrameTracker(default_config)
        with pytest.raises(ValueError, match="timestamp_s"):
            tracker.update([], timestamp_s=-0.05)

    def test_M_frame_transformation_with_robot_pose(self, default_config):
        tracker = MultiFrameTracker(default_config)
        # Detection is in 'base_link' at (2.0, 0.0). Robot pose in odom is (10.0, 5.0, 0.0)
        det_base = DynamicObstacleDetection("db", 0.1, "base_link", (2.0, 0.0), (1, 1), 0, 0.9)

        # Without robot_pose -> should raise ValueError
        with pytest.raises(ValueError, match="cannot be transformed"):
            tracker.update([det_base], timestamp_s=0.1, robot_pose=None)

        # With robot_pose -> should transform to odom (12.0, 5.0)
        arr = tracker.update([det_base], timestamp_s=0.1, robot_pose=(10.0, 5.0, 0.0))
        assert len(arr) == 1
        assert arr[0].frame_id == "odom"
        assert math.isclose(arr[0].x, 12.0)
        assert math.isclose(arr[0].y, 5.0)

    def test_N_deterministic_repeated_execution(self, default_config):
        def run_scenario():
            tr = MultiFrameTracker(default_config)
            results = []
            for i in range(6):
                t = 0.1 * (i + 1)
                d1 = DynamicObstacleDetection(f"d1_{i}", t, "odom", (1.0 + 0.2 * i, 0.0), (1, 1), 0, 0.9)
                d2 = DynamicObstacleDetection(f"d2_{i}", t, "odom", (0.0, 2.0 - 0.1 * i), (1, 1), 0, 0.9)
                res = tr.update([d1, d2], timestamp_s=t)
                results.append([(s.track_id, s.x, s.y, s.status) for s in res])
            return results

        run1 = run_scenario()
        run2 = run_scenario()
        assert run1 == run2

    def test_O_multiple_simultaneous_tracks(self, default_config):
        tracker = MultiFrameTracker(default_config)
        # 4 distinct targets
        targets = [(0.0, 3.0), (3.0, 0.0), (0.0, -3.0), (-3.0, 0.0)]
        for step in range(5):
            t = 0.1 * (step + 1)
            dets = [
                DynamicObstacleDetection(f"d_{idx}_{step}", t, "odom", pt, (0.8, 0.8), 0, 0.9)
                for idx, pt in enumerate(targets)
            ]
            arr = tracker.update(dets, timestamp_s=t)
            assert len(arr) == 4

        # Verify all 4 are confirmed and have persistent IDs [0, 1, 2, 3]
        confirmed = arr.confirmed_tracks()
        assert len(confirmed) == 4
        assert {t.track_id for t in confirmed} == {0, 1, 2, 3}

    def test_P_crossing_objects_stress_case(self, default_config):
        tracker = MultiFrameTracker(default_config)
        # Object 1 moves left-to-right along Y=0: x from -2.0 to +2.0
        # Object 2 moves bottom-to-top along X=0: y from -2.0 to +2.0
        # Step size 0.2m per frame; pass close at (0, 0)
        steps = 15
        xs = np.linspace(-1.5, 1.5, steps)
        ys = np.linspace(-1.5, 1.5, steps)

        track_0_history = []
        track_1_history = []

        for i in range(steps):
            t = 0.1 * (i + 1)
            d1 = DynamicObstacleDetection(f"d1_{i}", t, "odom", (float(xs[i]), 0.0), (0.5, 0.5), 0, 0.95)
            d2 = DynamicObstacleDetection(f"d2_{i}", t, "odom", (0.0, float(ys[i])), (0.5, 0.5), 0, 0.95)
            arr = tracker.update([d1, d2], timestamp_s=t)
            assert len(arr) == 2

            t0 = arr.get_track_by_id(0)
            t1 = arr.get_track_by_id(1)
            assert t0 is not None
            assert t1 is not None

            track_0_history.append((t0.x, t0.y))
            track_1_history.append((t1.x, t1.y))

        # Verify track 0 remained primarily on horizontal axis (y ~ 0)
        assert max(abs(pt[1]) for pt in track_0_history) < 0.20
        # Verify track 1 remained primarily on vertical axis (x ~ 0)
        assert max(abs(pt[0]) for pt in track_1_history) < 0.20

    def test_Q_association_spatial_gating(self, default_config):
        # max_association_distance_m = 1.50
        tracker = MultiFrameTracker(default_config)
        # Initialize Track 0 at (0.0, 0.0)
        tracker.update([DynamicObstacleDetection("d0", 0.1, "odom", (0.0, 0.0), (1, 1), 0, 0.9)], 0.1)

        # Detection at distance 2.5m (exceeds 1.5m gate)
        det_outside_gate = DynamicObstacleDetection("d_far", 0.2, "odom", (2.5, 0.0), (1, 1), 0, 0.9)
        arr = tracker.update([det_outside_gate], timestamp_s=0.2)

        # Track 0 was not matched (coasts), new Track 1 created for far detection
        assert len(arr) == 2
        t0 = arr.get_track_by_id(0)
        t1 = arr.get_track_by_id(1)
        assert t0 is not None and t0.coast_cycles == 1
        assert t1 is not None and t1.x == 2.5


class TestTrackerRobustnessAndAudit:
    def test_R_ground_truth_leakage_audit(self):
        tracker = MultiFrameTracker()
        # Verify update() signature has no GT parameters
        sig = inspect.signature(tracker.update)
        assert set(sig.parameters.keys()) == {"detections", "timestamp_s", "robot_pose"}

        # Verify TrackedObstacleState does not leak any GT attribute
        det = DynamicObstacleDetection("d", 0.1, "odom", (1, 1), (1, 1), 0, 0.9)
        arr = tracker.update([det], 0.1)
        state = arr[0]
        assert not hasattr(state, "ground_truth_pose")
        assert not hasattr(state, "gt_track_id")
        assert not hasattr(state, "future_velocity")

    def test_S_config_validation(self):
        with pytest.raises(ValueError, match="max_association_distance_m"):
            TrackerConfig(max_association_distance_m=-1.0)
        with pytest.raises(ValueError, match="hits_to_confirm"):
            TrackerConfig(hits_to_confirm=0)
        with pytest.raises(ValueError, match="misses_to_lost"):
            TrackerConfig(misses_to_degrade=2, misses_to_lost=2)  # lost must be > degrade
        with pytest.raises(ValueError, match="misses_to_expire"):
            TrackerConfig(misses_to_lost=4, misses_to_expire=3)  # expire must be > lost
        with pytest.raises(ValueError, match="min_detection_confidence"):
            TrackerConfig(min_detection_confidence=1.5)

    def test_synthetic_detector_pipeline_integration(self):
        # End-to-end integration: Synthetic Detector -> Tracker
        detector = SyntheticDynamicObstacleDetector(frame_id="odom", random_seed=42)
        detector.add_obstacle_script(
            SyntheticObstacleScript(
                obstacle_id="agent_1",
                size=(1.0, 0.6),
                position_fn=lambda t: (3.0 + 0.5 * t, 1.0),
                confidence=0.90,
            )
        )
        detector.add_obstacle_script(
            SyntheticObstacleScript(
                obstacle_id="agent_2",
                size=(0.8, 0.8),
                position_fn=lambda t: (1.0, 4.0 - 0.3 * t),
                confidence=0.85,
            )
        )

        tracker = MultiFrameTracker()
        for step in range(5):
            t = 0.1 * (step + 1)
            dets = detector.detect(frame=None, timestamp_s=t)
            arr = tracker.update(dets, timestamp_s=t)
            assert len(arr) == 2

        # Both tracks should now be confirmed
        assert len(arr.confirmed_tracks()) == 2
        assert {arr[0].track_id, arr[1].track_id} == {0, 1}
