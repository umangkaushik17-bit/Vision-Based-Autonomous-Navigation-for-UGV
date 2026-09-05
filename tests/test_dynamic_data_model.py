"""Unit tests for Milestone 13.1 — Dynamic Obstacle Data Model.

Tests:
1. Normal instantiation of DynamicObstacleDetection, TrackedObstacleState, TrajectoryWaypoint, PredictedObstacleTrajectory
2. Validation rules:
   - Non-negative timestamp
   - Positive bounding box dimensions
   - Confidence bounds [0, 1]
   - Non-empty frame_id
   - Positive semi-definite covariance check
   - Trajectory horizon, dt, and waypoint/polygon length equality
3. Geometric methods:
   - compute_obb_corners against expected positions
   - Margin expansions
   - Axis-aligned bounding box (AABB) bounds
   - SAT polygon intersection: overlapping vs separated boxes
   - Point-in-polygon verification
4. Coordinate frame transformations:
   - base_link to odom with translation and rotation
   - odom to base_link inverse transformation
   - raw cluster points transformation
5. Tracked state methods:
   - Speed, velocity heading, position, velocity
   - Position and velocity covariance submatrices
   - 3-sigma spatial uncertainty radius
6. Trajectory interpolation:
   - Interpolate state at offsets
   - Closest waypoint and polygon retrieval
7. Array containers:
   - DynamicObstacleDetectionArray, TrackedObstacleArray, PredictedTrajectoryArray
   - Filtering by track ID and confirmed status
8. Serialization / Deserialization:
   - Full round-trip fidelity (to_dict -> from_dict)
"""

import math
import numpy as np
import pytest

from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    DynamicObstacleDetectionArray,
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
    normalize_angle,
    point_in_polygon,
    polygon_intersection_sat,
)


class TestAngleAndGeometryUtilities:
    def test_normalize_angle(self):
        assert math.isclose(normalize_angle(0.0), 0.0)
        assert math.isclose(abs(normalize_angle(math.pi)), math.pi, abs_tol=1e-6)
        assert math.isclose(abs(normalize_angle(3 * math.pi)), math.pi, abs_tol=1e-6)
        assert math.isclose(abs(normalize_angle(-3 * math.pi)), math.pi, abs_tol=1e-6)
        assert math.isclose(normalize_angle(math.pi / 2.0), math.pi / 2.0)
        assert math.isclose(normalize_angle(-math.pi / 2.0), -math.pi / 2.0)
        assert math.isclose(normalize_angle(2.5 * math.pi), math.pi / 2.0, abs_tol=1e-6)

    def test_compute_obb_corners_unrotated(self):
        center = (2.0, 3.0)
        size = (1.0, 0.5)  # length=1.0 (hl=0.5), width=0.5 (hw=0.25)
        yaw = 0.0
        corners = compute_obb_corners(center, size, yaw)
        assert corners.shape == (4, 2)
        # Order: front-left, front-right, rear-right, rear-left
        np.testing.assert_allclose(corners[0], [2.5, 3.25])
        np.testing.assert_allclose(corners[1], [2.5, 2.75])
        np.testing.assert_allclose(corners[2], [1.5, 2.75])
        np.testing.assert_allclose(corners[3], [1.5, 3.25])

    def test_compute_obb_corners_rotated_90_deg(self):
        center = (0.0, 0.0)
        size = (2.0, 1.0)  # hl=1.0, hw=0.5
        yaw = math.pi / 2.0  # rotated 90 deg CCW
        corners = compute_obb_corners(center, size, yaw)
        # In body frame: front-left = [1.0, 0.5] -> rotated by 90 deg: [-0.5, 1.0]
        np.testing.assert_allclose(corners[0], [-0.5, 1.0], atol=1e-6)
        np.testing.assert_allclose(corners[1], [0.5, 1.0], atol=1e-6)
        np.testing.assert_allclose(corners[2], [0.5, -1.0], atol=1e-6)
        np.testing.assert_allclose(corners[3], [-0.5, -1.0], atol=1e-6)

    def test_compute_obb_corners_with_margin(self):
        center = (0.0, 0.0)
        size = (1.0, 1.0)  # hl=0.5, hw=0.5
        margin = 0.2
        corners = compute_obb_corners(center, size, 0.0, margin_m=margin)
        # Effective hl=0.7, hw=0.7
        np.testing.assert_allclose(corners[0], [0.7, 0.7])
        np.testing.assert_allclose(corners[2], [-0.7, -0.7])

    def test_polygon_intersection_sat(self):
        # Box 1: centered at (0, 0), size 2x2 -> [-1, 1] x [-1, 1]
        b1 = compute_obb_corners((0.0, 0.0), (2.0, 2.0), 0.0)
        # Box 2: overlapping at (1.5, 0), size 2x2 -> [0.5, 2.5] x [-1, 1]
        b2 = compute_obb_corners((1.5, 0.0), (2.0, 2.0), 0.0)
        assert polygon_intersection_sat(b1, b2) is True

        # Box 3: separated at (3.0, 0), size 2x2 -> [2.0, 4.0] x [-1, 1]
        b3 = compute_obb_corners((3.0, 0.0), (2.0, 2.0), 0.0)
        assert polygon_intersection_sat(b1, b3) is False

        # Box 4: diagonal separation
        b4 = compute_obb_corners((2.5, 2.5), (1.0, 1.0), 0.0)
        assert polygon_intersection_sat(b1, b4) is False

        # Box 5: rotated collision
        b5 = compute_obb_corners((1.2, 0.0), (1.0, 1.0), math.pi / 4.0)
        assert polygon_intersection_sat(b1, b5) is True

    def test_point_in_polygon(self):
        poly = compute_obb_corners((0.0, 0.0), (2.0, 2.0), 0.0)
        assert point_in_polygon((0.0, 0.0), poly) is True
        assert point_in_polygon((0.5, -0.5), poly) is True
        assert point_in_polygon((2.0, 0.0), poly) is False
        assert point_in_polygon((-1.5, 0.0), poly) is False


class TestDynamicObstacleDetection:
    def test_valid_creation(self):
        det = DynamicObstacleDetection(
            detection_id="det_001",
            timestamp=1.25,
            frame_id="base_link",
            bbox_center=(3.0, 0.5),
            bbox_size=(0.8, 0.6),
            bbox_yaw=0.1,
            confidence=0.95,
        )
        assert det.detection_id == "det_001"
        assert det.timestamp == 1.25
        assert det.frame_id == "base_link"
        assert det.bbox_center == (3.0, 0.5)
        assert det.bbox_size == (0.8, 0.6)
        assert math.isclose(det.bbox_yaw, 0.1)
        assert det.confidence == 0.95
        assert det.raw_cluster_points is None

    def test_validation_errors(self):
        # Negative timestamp
        with pytest.raises(ValueError, match="timestamp"):
            DynamicObstacleDetection("d1", -0.1, "base_link", (0, 0), (1, 1), 0, 0.5)

        # Non-positive size
        with pytest.raises(ValueError, match="bbox_size"):
            DynamicObstacleDetection("d1", 1.0, "base_link", (0, 0), (0, 1), 0, 0.5)
        with pytest.raises(ValueError, match="bbox_size"):
            DynamicObstacleDetection("d1", 1.0, "base_link", (0, 0), (1, -0.5), 0, 0.5)

        # Invalid confidence
        with pytest.raises(ValueError, match="confidence"):
            DynamicObstacleDetection("d1", 1.0, "base_link", (0, 0), (1, 1), 0, 1.2)
        with pytest.raises(ValueError, match="confidence"):
            DynamicObstacleDetection("d1", 1.0, "base_link", (0, 0), (1, 1), 0, -0.1)

        # Empty frame_id
        with pytest.raises(ValueError, match="frame_id"):
            DynamicObstacleDetection("d1", 1.0, "", (0, 0), (1, 1), 0, 0.5)

    def test_frame_transformation_base_to_odom(self):
        # Robot is at odom (10.0, 5.0) with yaw = 90 deg (math.pi/2)
        # Obstacle is 2.0m forward in base_link (x=2.0, y=0.0)
        det_base = DynamicObstacleDetection(
            detection_id="det_base_1",
            timestamp=10.0,
            frame_id="base_link",
            bbox_center=(2.0, 0.0),
            bbox_size=(1.0, 0.5),
            bbox_yaw=0.0,
            confidence=0.9,
            raw_cluster_points=np.array([[2.0, 0.0], [2.1, 0.1]]),
        )
        robot_pose = (10.0, 5.0, math.pi / 2.0)
        det_odom = det_base.transform_to_frame("odom", robot_pose)

        assert det_odom.frame_id == "odom"
        # 2m forward when facing +Y means center should be (10.0, 7.0)
        assert math.isclose(det_odom.bbox_center[0], 10.0, abs_tol=1e-6)
        assert math.isclose(det_odom.bbox_center[1], 7.0, abs_tol=1e-6)
        assert math.isclose(det_odom.bbox_yaw, math.pi / 2.0, abs_tol=1e-6)

        # Inverse transform back to base_link
        det_recovered = det_odom.transform_to_frame("base_link", robot_pose)
        assert det_recovered.frame_id == "base_link"
        assert math.isclose(det_recovered.bbox_center[0], 2.0, abs_tol=1e-6)
        assert math.isclose(det_recovered.bbox_center[1], 0.0, abs_tol=1e-6)
        assert math.isclose(det_recovered.bbox_yaw, 0.0, abs_tol=1e-6)

    def test_serialization_round_trip(self):
        det = DynamicObstacleDetection(
            detection_id="test_det",
            timestamp=5.5,
            frame_id="odom",
            bbox_center=(1.2, 3.4),
            bbox_size=(0.7, 0.4),
            bbox_yaw=-0.3,
            confidence=0.88,
            raw_cluster_points=np.array([[1.1, 3.3], [1.3, 3.5]]),
        )
        d = det.to_dict()
        det_reconstructed = DynamicObstacleDetection.from_dict(d)
        assert det_reconstructed.detection_id == det.detection_id
        assert det_reconstructed.timestamp == det.timestamp
        assert det_reconstructed.bbox_center == det.bbox_center
        assert det_reconstructed.bbox_size == det.bbox_size
        assert math.isclose(det_reconstructed.bbox_yaw, det.bbox_yaw)
        assert math.isclose(det_reconstructed.confidence, det.confidence)
        np.testing.assert_allclose(det_reconstructed.raw_cluster_points, det.raw_cluster_points)


class TestTrackedObstacleState:
    @pytest.fixture
    def valid_cov(self):
        return np.diag([0.04, 0.04, 0.09, 0.09])  # sigma_pos = 0.2m, sigma_vel = 0.3m/s

    def test_valid_creation(self, valid_cov):
        track = TrackedObstacleState(
            track_id=42,
            timestamp=15.0,
            x=5.0,
            y=2.0,
            vx=0.5,
            vy=-0.2,
            covariance=valid_cov,
            bbox_size=(1.2, 0.6),
            bbox_yaw=0.0,
            track_age=5,
            coast_cycles=0,
            is_confirmed=True,
        )
        assert track.track_id == 42
        assert track.frame_id == "odom"
        assert math.isclose(track.speed, math.hypot(0.5, -0.2))
        assert math.isclose(track.velocity_heading, math.atan2(-0.2, 0.5))
        assert track.position == (5.0, 2.0)
        assert track.velocity == (0.5, -0.2)
        assert track.is_confirmed is True

    def test_covariance_validation(self):
        # Non-square covariance
        with pytest.raises(ValueError, match="shape"):
            TrackedObstacleState(
                track_id=1,
                timestamp=1.0,
                x=0,
                y=0,
                vx=0,
                vy=0,
                covariance=np.eye(3),
                bbox_size=(1, 1),
                bbox_yaw=0,
            )

        # Negative eigenvalues (not positive semi-definite)
        bad_cov = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1],
        ])
        with pytest.raises(ValueError, match="positive semi-definite"):
            TrackedObstacleState(
                track_id=1,
                timestamp=1.0,
                x=0,
                y=0,
                vx=0,
                vy=0,
                covariance=bad_cov,
                bbox_size=(1, 1),
                bbox_yaw=0,
            )

    def test_position_uncertainty_radius(self, valid_cov):
        track = TrackedObstacleState(
            track_id=1,
            timestamp=1.0,
            x=0,
            y=0,
            vx=0,
            vy=0,
            covariance=valid_cov,  # var_x = 0.04 -> std = 0.2
            bbox_size=(1, 1),
            bbox_yaw=0,
        )
        # 3-sigma of 0.2 is 0.6 m
        assert math.isclose(track.position_uncertainty_radius(sigma_scale=3.0), 0.6, abs_tol=1e-6)

    def test_serialization_round_trip(self, valid_cov):
        track = TrackedObstacleState(
            track_id=7,
            timestamp=2.5,
            x=1.5,
            y=-3.2,
            vx=1.1,
            vy=0.4,
            covariance=valid_cov,
            bbox_size=(0.9, 0.5),
            bbox_yaw=0.75,
            track_age=12,
            coast_cycles=1,
            is_confirmed=True,
            history=[(1.0, 0.0, 0.0), (2.0, 1.0, 0.0), (2.5, 1.5, -3.2)],
        )
        d = track.to_dict()
        recovered = TrackedObstacleState.from_dict(d)
        assert recovered.track_id == 7
        assert recovered.x == track.x
        assert recovered.y == track.y
        assert recovered.vx == track.vx
        assert recovered.vy == track.vy
        np.testing.assert_allclose(recovered.covariance, valid_cov)
        assert recovered.bbox_size == (0.9, 0.5)
        assert recovered.is_confirmed is True
        assert len(recovered.history) == 3


class TestTrajectoryPredictionAndWaypoints:
    @pytest.fixture
    def sample_track(self):
        cov = np.diag([0.01, 0.01, 0.04, 0.04])
        return TrackedObstacleState(
            track_id=10,
            timestamp=0.0,
            x=0.0,
            y=0.0,
            vx=1.0,
            vy=0.0,
            covariance=cov,
            bbox_size=(1.0, 0.6),
            bbox_yaw=0.0,
        )

    def test_trajectory_interpolation(self, sample_track):
        horizon = 2.0
        dt = 0.5
        waypoints = []
        polygons = []

        for step in range(5):
            t_off = step * dt
            wp = TrajectoryWaypoint(
                time_offset=t_off,
                timestamp=t_off,
                x=t_off * 1.0,
                y=0.0,
                vx=1.0,
                vy=0.0,
                covariance_2d=np.diag([0.01 * (step + 1), 0.01 * (step + 1)]),
                uncertainty_radius=3.0 * math.sqrt(0.01 * (step + 1)),
            )
            poly = compute_obb_corners((wp.x, wp.y), (1.0, 0.6), 0.0)
            waypoints.append(wp)
            polygons.append(poly)

        traj = PredictedObstacleTrajectory(
            track_id=10,
            current_state=sample_track,
            horizon_s=horizon,
            dt=dt,
            waypoints=waypoints,
            bounding_polygon_per_step=polygons,
        )

        # Interpolate at exact waypoint offset 1.0
        state_1 = traj.interpolate_state_at(1.0)
        assert state_1 is not None
        assert math.isclose(state_1[0], 1.0)
        assert math.isclose(state_1[1], 0.0)

        # Interpolate halfway between 0.5 and 1.0 -> 0.75
        state_mid = traj.interpolate_state_at(0.75)
        assert state_mid is not None
        assert math.isclose(state_mid[0], 0.75)
        assert math.isclose(state_mid[2], 1.0)

        # Out-of-bounds interpolation returns None
        assert traj.interpolate_state_at(-0.1) is None
        assert traj.interpolate_state_at(2.5) is None

        # Nearest waypoint retrieval
        wp_nearest = traj.get_waypoint_at_offset(0.7)
        assert wp_nearest is not None
        assert math.isclose(wp_nearest.time_offset, 0.5)

    def test_predicted_trajectory_serialization(self, sample_track):
        wp = TrajectoryWaypoint(
            time_offset=0.5,
            timestamp=0.5,
            x=0.5,
            y=0.0,
            vx=1.0,
            vy=0.0,
            covariance_2d=np.diag([0.02, 0.02]),
            uncertainty_radius=0.42,
        )
        poly = compute_obb_corners((0.5, 0.0), (1.0, 0.6), 0.0)
        traj = PredictedObstacleTrajectory(
            track_id=10,
            current_state=sample_track,
            horizon_s=0.5,
            dt=0.5,
            waypoints=[wp],
            bounding_polygon_per_step=[poly],
        )
        d = traj.to_dict()
        recovered = PredictedObstacleTrajectory.from_dict(d)
        assert recovered.track_id == 10
        assert len(recovered.waypoints) == 1
        assert math.isclose(recovered.waypoints[0].x, 0.5)
        np.testing.assert_allclose(recovered.bounding_polygon_per_step[0], poly)


class TestArrayContainers:
    def test_detection_array(self):
        d1 = DynamicObstacleDetection("d1", 1.0, "base_link", (1, 0), (0.5, 0.5), 0, 0.9)
        d2 = DynamicObstacleDetection("d2", 1.0, "base_link", (2, 0), (0.5, 0.5), 0, 0.8)
        arr = DynamicObstacleDetectionArray(header_timestamp=1.0, frame_id="base_link", detections=[d1, d2])
        assert len(arr) == 2
        assert arr[0].detection_id == "d1"

        d = arr.to_dict()
        recovered = DynamicObstacleDetectionArray.from_dict(d)
        assert len(recovered) == 2
        assert recovered[1].detection_id == "d2"

    def test_tracked_obstacle_array(self):
        cov = np.eye(4) * 0.05
        t1 = TrackedObstacleState(1, 1.0, 0, 0, 0, 0, cov, (1, 1), 0, is_confirmed=False)
        t2 = TrackedObstacleState(2, 1.0, 2, 0, 1, 0, cov, (1, 1), 0, is_confirmed=True)
        arr = TrackedObstacleArray(header_timestamp=1.0, frame_id="odom", tracks=[t1, t2])

        assert len(arr) == 2
        assert arr.get_track_by_id(2) == t2
        assert arr.get_track_by_id(99) is None

        confirmed = arr.confirmed_tracks()
        assert len(confirmed) == 1
        assert confirmed[0].track_id == 2

        d = arr.to_dict()
        recovered = TrackedObstacleArray.from_dict(d)
        assert len(recovered) == 2
        assert recovered.get_track_by_id(2).is_confirmed is True

    def test_predicted_trajectory_array(self):
        cov = np.eye(4) * 0.05
        t1 = TrackedObstacleState(10, 1.0, 0, 0, 1, 0, cov, (1, 1), 0)
        wp = TrajectoryWaypoint(0.5, 1.5, 0.5, 0, 1, 0, np.eye(2) * 0.02, 0.3)
        poly = compute_obb_corners((0.5, 0), (1, 1), 0)
        traj = PredictedObstacleTrajectory(10, t1, 0.5, 0.5, [wp], [poly])

        arr = PredictedTrajectoryArray(header_timestamp=1.0, frame_id="odom", predictions=[traj])
        assert len(arr) == 1
        assert arr.get_prediction_by_id(10) == traj
        assert arr.get_prediction_by_id(99) is None

        d = arr.to_dict()
        recovered = PredictedTrajectoryArray.from_dict(d)
        assert len(recovered) == 1
        assert recovered.get_prediction_by_id(10).track_id == 10


class TestEdgeCasesAndIntegrity:
    def test_covariance_symmetrization(self):
        # Slightly asymmetric matrix that is positive definite
        asym_cov = np.array([
            [0.10, 0.01, 0.00, 0.00],
            [0.02, 0.10, 0.00, 0.00],
            [0.00, 0.00, 0.20, 0.00],
            [0.00, 0.00, 0.00, 0.20],
        ])
        track = TrackedObstacleState(
            track_id=1,
            timestamp=0.0,
            x=0,
            y=0,
            vx=0,
            vy=0,
            covariance=asym_cov,
            bbox_size=(1, 1),
            bbox_yaw=0,
        )
        # Should be symmetrized: off-diag element = (0.01 + 0.02)/2 = 0.015
        assert math.isclose(track.covariance[0, 1], 0.015)
        assert math.isclose(track.covariance[1, 0], 0.015)

    def test_invalid_cluster_points_shape(self):
        with pytest.raises(ValueError, match="raw_cluster_points"):
            DynamicObstacleDetection(
                "d", 1.0, "base_link", (0, 0), (1, 1), 0, 0.5,
                raw_cluster_points=np.array([1.0, 2.0, 3.0])  # 1D array
            )
        with pytest.raises(ValueError, match="raw_cluster_points"):
            DynamicObstacleDetection(
                "d", 1.0, "base_link", (0, 0), (1, 1), 0, 0.5,
                raw_cluster_points=np.zeros((5, 3))  # 3D points
            )

    def test_unsupported_frame_transform(self):
        det = DynamicObstacleDetection("d", 1.0, "custom_frame", (0, 0), (1, 1), 0, 0.5)
        with pytest.raises(ValueError, match="Unsupported frame transformation"):
            det.transform_to_frame("odom", (0, 0, 0))

    def test_predicted_trajectory_length_mismatch(self):
        cov = np.eye(4) * 0.05
        t1 = TrackedObstacleState(1, 1.0, 0, 0, 0, 0, cov, (1, 1), 0)
        wp = TrajectoryWaypoint(0.5, 1.5, 0.5, 0, 1, 0, np.eye(2) * 0.02, 0.3)
        # 1 waypoint, 0 polygons -> mismatch
        with pytest.raises(ValueError, match="match bounding_polygon_per_step"):
            PredictedObstacleTrajectory(1, t1, 0.5, 0.5, [wp], [])

    def test_sat_contained_polygon(self):
        # Big box [-2, 2] x [-2, 2]
        big = compute_obb_corners((0, 0), (4, 4), 0)
        # Small box [-0.5, 0.5] x [-0.5, 0.5]
        small = compute_obb_corners((0, 0), (1, 1), 0)
        assert polygon_intersection_sat(big, small) is True
        assert polygon_intersection_sat(small, big) is True

    def test_static_costmap_separation(self):
        # Verify dynamic module does NOT import or tamper with WorldTerrainMap or A*
        import sys
        assert "ai.planning.dynamic.data_model" in sys.modules
        from ai.planning.dynamic.data_model import DynamicObstacleDetection
        # Verify no costmap references exist in the data model
        import inspect
        source = inspect.getsource(DynamicObstacleDetection)
        assert "WorldTerrainMap" not in source
        assert "BEVTerrainCostmap" not in source

