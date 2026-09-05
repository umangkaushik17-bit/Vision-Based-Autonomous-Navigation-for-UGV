"""Comprehensive unit test suite for Phase 13 Milestone 13.7 — Dynamic Occupancy Layer.

Covers all 16 required test scenarios:
1. Single static obstacle prediction
2. Straight-line moving obstacle
3. Turning/moving trajectory
4. Uncertainty expansion
5. Time-bin separation
6. Interpolation / query between prediction times
7. Multiple obstacles (4 simultaneous)
8. Overlapping obstacles (conservative maximum resolution)
9. DEGRADED state
10. LOST state
11. EXPIRED rejection
12. Out-of-horizon query
13. Deterministic repeated output
14. Coordinate-frame consistency
15. Grid boundary handling
16. Empty input
"""

import math
import numpy as np
import pytest

from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleState,
)
from ai.planning.dynamic.occupancy_layer import (
    DynamicOccupancyConfig,
    DynamicOccupancyGridSlice,
    DynamicOccupancyLayer,
    DynamicRiskLevel,
)
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig


def make_track(
    track_id: int = 1,
    timestamp: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    pos_var: float = 0.04,
    vel_var: float = 0.09,
    bbox_size: tuple[float, float] = (1.5, 0.8),
    bbox_yaw: float = 0.0,
    status: str = "TRACKING",
) -> TrackedObstacleState:
    """Helper to construct a valid TrackedObstacleState."""
    cov = np.diag([pos_var, pos_var, vel_var, vel_var]).astype(np.float64)
    return TrackedObstacleState(
        track_id=track_id,
        timestamp=timestamp,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        covariance=cov,
        bbox_size=bbox_size,
        bbox_yaw=bbox_yaw,
        status=status,
    )


# ============================================================================
# TEST 1: Single static obstacle prediction
# ============================================================================
def test_1_single_static_obstacle() -> None:
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    track = make_track(x=5.0, y=5.0, vx=0.0, vy=0.0)
    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.25)
    layer.update_predictions([pred])

    # Center of obstacle must be strictly LETHAL with cost 1.0 across all time slices
    for t_bin in layer.get_time_bins():
        cost = layer.get_cost(5.0, 5.0, t_bin)
        assert cost == pytest.approx(1.0)
        assert layer.is_lethal(5.0, 5.0, t_bin)
        assert layer.is_occupied(5.0, 5.0, t_bin)
        assert layer.get_risk_level(5.0, 5.0, t_bin) == DynamicRiskLevel.LETHAL

    # Far away cell (0.0, 0.0) must be FREE with cost 0.0
    for t_bin in layer.get_time_bins():
        cost_far = layer.get_cost(0.0, 0.0, t_bin)
        assert cost_far == pytest.approx(0.0)
        assert not layer.is_occupied(0.0, 0.0, t_bin)
        assert layer.get_risk_level(0.0, 0.0, t_bin) == DynamicRiskLevel.FREE


# ============================================================================
# TEST 2: Straight-line moving obstacle
# ============================================================================
def test_2_straight_line_moving_obstacle() -> None:
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    # Moving at vx = 2.0 m/s along X
    track = make_track(x=0.0, y=0.0, vx=2.0, vy=0.0)
    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.25)
    layer.update_predictions([pred])

    # At t=0.0s, (0.0, 0.0) is lethal
    assert layer.is_lethal(0.0, 0.0, 0.0)
    assert layer.get_cost(0.0, 0.0, 0.0) == pytest.approx(1.0)

    # At t=1.0s, obstacle has moved to x = 2.0m
    assert layer.is_lethal(2.0, 0.0, 1.0)
    # Origin (0.0, 0.0) is no longer lethal at t=1.0s
    assert not layer.is_lethal(0.0, 0.0, 1.0)

    # At t=2.0s, obstacle has moved to x = 4.0m
    assert layer.is_lethal(4.0, 0.0, 2.0)
    assert not layer.is_lethal(2.0, 0.0, 2.0)


# ============================================================================
# TEST 3: Turning/moving trajectory
# ============================================================================
def test_3_turning_moving_trajectory() -> None:
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    # Diagonal velocity vx=2.0, vy=2.0
    track = make_track(x=0.0, y=0.0, vx=2.0, vy=2.0, bbox_yaw=math.pi / 4)
    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.5)
    layer.update_predictions([pred])

    # At t=0.5s: center at (1.0, 1.0)
    assert layer.is_lethal(1.0, 1.0, 0.5)
    # At t=1.0s: center at (2.0, 2.0)
    assert layer.is_lethal(2.0, 2.0, 1.0)
    # At t=2.0s: center at (4.0, 4.0)
    assert layer.is_lethal(4.0, 4.0, 2.0)


# ============================================================================
# TEST 4: Uncertainty expansion
# ============================================================================
def test_4_uncertainty_expansion() -> None:
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    track = make_track(x=0.0, y=0.0, vx=1.0, vy=0.0)
    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.25)
    layer.update_predictions([pred])

    stats_early = layer.get_slice_statistics(0.25)
    stats_late = layer.get_slice_statistics(2.00)

    # Nominal physical footprint area remains constant within raster discretization tolerance (~120 cells)
    assert stats_early["nominal_cells"] == pytest.approx(stats_late["nominal_cells"], abs=12)

    # Uncertainty zone expands over time
    assert stats_late["uncertainty_cells"] > stats_early["uncertainty_cells"]
    assert stats_late["total_affected_cells"] > stats_early["total_affected_cells"]

    # Test graded cost decay: query points at increasing lateral distance from nominal footprint
    # At t=1.0s, center is at (1.0, 0.0), half_width is 0.4m
    cost_nominal = layer.get_cost(1.0, 0.0, 1.0)    # inside box -> 1.0
    cost_near = layer.get_cost(1.0, 0.8, 1.0)       # ~0.4m from boundary (high risk)
    cost_mid = layer.get_cost(1.0, 1.6, 1.0)        # ~1.2m from boundary (medium risk)
    cost_far = layer.get_cost(1.0, 2.0, 1.0)        # ~1.6m from boundary (low risk)
    cost_outside = layer.get_cost(1.0, 5.0, 1.0)    # outside envelope -> 0.0

    assert cost_nominal == pytest.approx(1.0)
    assert 0.70 <= cost_near < 1.0
    assert 0.40 <= cost_mid < 0.70
    assert 0.15 <= cost_far < 0.40
    assert cost_outside == pytest.approx(0.0)


# ============================================================================
# TEST 5: Time-bin separation
# ============================================================================
def test_5_time_bin_separation() -> None:
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    track = make_track(x=0.0, y=0.0, vx=3.0, vy=0.0)
    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.5)
    layer.update_predictions([pred])

    slice_0 = layer.get_slice(0.0)
    slice_1 = layer.get_slice(1.0)

    # Position in slice_0 must not equal position in slice_1
    assert not np.array_equal(slice_0.cost, slice_1.cost)

    # Modifying a copied slice does not affect layer
    slice_copy = slice_0.copy()
    slice_copy.cost[10, 10] = 99.0
    assert layer.get_slice(0.0).cost[10, 10] != 99.0


# ============================================================================
# TEST 6: Interpolation / query between prediction times
# ============================================================================
def test_6_interpolation_between_prediction_times() -> None:
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    track = make_track(x=0.0, y=0.0, vx=2.0, vy=0.0)
    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.5)
    layer.update_predictions([pred])

    # Query at t=0.6s (between t=0.5 and t=1.0)
    # Nearest bin without conservative_union -> t=0.5 (where x=1.0)
    slice_nearest = layer.get_slice(0.6, conservative_union=False)
    assert slice_nearest.time_offset == pytest.approx(0.5)

    # Conservative union with conservative_union=True -> covers both x=1.0 and x=2.0
    slice_union = layer.get_slice(0.6, conservative_union=True)
    assert slice_union.time_offset == pytest.approx(0.6)
    # Both positions should be active in the union
    assert slice_union.cost[layer.world_to_grid(1.0, 0.0)] > 0.5
    assert slice_union.cost[layer.world_to_grid(2.0, 0.0)] > 0.5


# ============================================================================
# TEST 7: Multiple obstacles
# ============================================================================
def test_7_multiple_obstacles() -> None:
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    tracks = [
        make_track(track_id=1, x=-5.0, y=-5.0, vx=1.0, vy=0.0),
        make_track(track_id=2, x=5.0, y=5.0, vx=-1.0, vy=0.0),
        make_track(track_id=3, x=-5.0, y=5.0, vx=0.0, vy=-1.0),
        make_track(track_id=4, x=5.0, y=-5.0, vx=0.0, vy=1.0),
    ]
    preds = predictor.predict_tracks(tracks, horizon_s=1.0, dt_s=0.5)
    layer.update_predictions(preds)

    slice_0 = layer.get_slice(0.0)
    assert len(slice_0.active_obstacle_ids) == 4
    assert set(slice_0.active_obstacle_ids) == {1, 2, 3, 4}

    # All 4 nominal centers are lethal
    assert layer.is_lethal(-5.0, -5.0, 0.0)
    assert layer.is_lethal(5.0, 5.0, 0.0)
    assert layer.is_lethal(-5.0, 5.0, 0.0)
    assert layer.is_lethal(5.0, -5.0, 0.0)


# ============================================================================
# TEST 8: Overlapping obstacles (conservative safety precedence)
# ============================================================================
def test_8_overlapping_obstacles() -> None:
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    # Obstacle 1 at (0.0, 0.0), Obstacle 2 at (0.5, 0.0) (partially overlapping)
    tracks = [
        make_track(track_id=1, x=0.0, y=0.0, vx=0.0, vy=0.0),
        make_track(track_id=2, x=0.5, y=0.0, vx=0.0, vy=0.0),
    ]
    preds = predictor.predict_tracks(tracks, horizon_s=1.0, dt_s=0.5)
    layer.update_predictions(preds)

    # Overlapping zone must be LETHAL
    cost_mid = layer.get_cost(0.25, 0.0, 0.0)
    assert cost_mid == pytest.approx(1.0)
    assert layer.is_lethal(0.25, 0.0, 0.0)


# ============================================================================
# TEST 9: DEGRADED state
# ============================================================================
def test_9_degraded_state() -> None:
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    track_norm = make_track(track_id=1, x=0.0, y=0.0, status="TRACKING")
    track_deg = make_track(track_id=2, x=0.0, y=0.0, status="DEGRADED")

    pred_norm = predictor.predict_track(track_norm, horizon_s=2.0, dt_s=0.5)
    pred_deg = predictor.predict_track(track_deg, horizon_s=2.0, dt_s=0.5)

    layer_norm = DynamicOccupancyLayer()
    layer_norm.update_predictions([pred_norm])

    layer_deg = DynamicOccupancyLayer()
    layer_deg.update_predictions([pred_deg])

    # Safety rule: Nominal footprint remains LETHAL even when DEGRADED
    assert layer_deg.is_lethal(0.0, 0.0, 1.0)
    assert layer_deg.get_cost(0.0, 0.0, 1.0) == pytest.approx(1.0)

    # Uncertainty area for DEGRADED is strictly larger than normal TRACKING
    stats_norm = layer_norm.get_slice_statistics(2.0)
    stats_deg = layer_deg.get_slice_statistics(2.0)
    assert stats_deg["uncertainty_cells"] > stats_norm["uncertainty_cells"]


# ============================================================================
# TEST 10: LOST state
# ============================================================================
def test_10_lost_state() -> None:
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    track_lost = make_track(track_id=3, x=0.0, y=0.0, status="LOST")
    pred = predictor.predict_track(track_lost, horizon_s=2.0, dt_s=0.5)
    layer.update_predictions([pred])

    # Active at t=0.0s and t=0.5s and t=1.0s (capped at lost_max_horizon_s = 1.0s)
    assert layer.is_lethal(0.0, 0.0, 0.0)
    assert layer.is_lethal(0.0, 0.0, 1.0)

    # At t=1.5s and t=2.0s, obstacle prediction has expired (no cells occupied)
    stats_post_loss = layer.get_slice_statistics(1.5)
    assert stats_post_loss["total_affected_cells"] == 0
    assert not layer.is_occupied(0.0, 0.0, 1.5)


# ============================================================================
# TEST 11: EXPIRED rejection
# ============================================================================
def test_11_expired_rejection() -> None:
    layer = DynamicOccupancyLayer()

    track_exp = make_track(track_id=4, status="EXPIRED")
    # Fabricated trajectory for expired track
    pred = PredictedObstacleTrajectory(
        track_id=4,
        current_state=track_exp,
        horizon_s=1.0,
        dt=0.5,
        waypoints=[],
        bounding_polygon_per_step=[],
    )

    with pytest.raises(ValueError, match="EXPIRED"):
        layer.update_predictions([pred])


# ============================================================================
# TEST 12: Out-of-horizon query
# ============================================================================
def test_12_out_of_horizon_query() -> None:
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    track = make_track(x=0.0, y=0.0)
    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.5)
    layer.update_predictions([pred])

    # Negative offset raises ValueError
    with pytest.raises(ValueError, match="negative"):
        layer.get_slice(-0.5)

    with pytest.raises(ValueError, match="finite"):
        layer.get_slice(float("nan"))

    # Beyond max_horizon (e.g. t=5.0s) returns empty slice
    slice_ooh = layer.get_slice(5.0)
    assert np.all(slice_ooh.cost == 0.0)
    assert not layer.is_occupied(0.0, 0.0, 5.0)


# ============================================================================
# TEST 13: Deterministic repeated output
# ============================================================================
def test_13_deterministic_repeated_output() -> None:
    predictor = MotionPredictor()
    layer1 = DynamicOccupancyLayer()
    layer2 = DynamicOccupancyLayer()

    track = make_track(x=2.5, y=-1.5, vx=1.2, vy=-0.4)
    pred = predictor.predict_track(track, horizon_s=1.5, dt_s=0.25)

    layer1.update_predictions([pred])
    layer2.update_predictions([pred])

    for t in layer1.get_time_bins():
        s1 = layer1.get_slice(t)
        s2 = layer2.get_slice(t)
        np.testing.assert_array_equal(s1.cost, s2.cost)
        np.testing.assert_array_equal(s1.risk_level, s2.risk_level)
        np.testing.assert_array_equal(s1.nominal_obstacle, s2.nominal_obstacle)


# ============================================================================
# TEST 14: Coordinate-frame consistency
# ============================================================================
def test_14_coordinate_frame_consistency() -> None:
    layer = DynamicOccupancyLayer()

    # Round-trip transformation
    x_test, y_test = 5.30, -7.80
    r, c = layer.world_to_grid(x_test, y_test)
    x_rec, y_rec = layer.grid_to_world(r, c)

    # Resolution is 0.10m, so coordinate should be within 0.05m of cell center
    assert abs(x_test - x_rec) <= layer.resolution_m / 2.0 + 1e-6
    assert abs(y_test - y_rec) <= layer.resolution_m / 2.0 + 1e-6


# ============================================================================
# TEST 15: Grid boundary handling
# ============================================================================
def test_15_grid_boundary_handling() -> None:
    predictor = MotionPredictor()
    layer = DynamicOccupancyLayer()

    # Obstacle placed right at boundary edge
    track_edge = make_track(x=19.5, y=0.0, vx=1.0, vy=0.0)
    pred_edge = predictor.predict_track(track_edge, horizon_s=1.0, dt_s=0.5)

    # Must rasterize without crashing or throwing IndexError
    layer.update_predictions([pred_edge])

    # Obstacle placed completely outside grid (e.g. x=50.0m)
    track_out = make_track(x=50.0, y=50.0, vx=0.0, vy=0.0)
    pred_out = predictor.predict_track(track_out, horizon_s=1.0, dt_s=0.5)
    layer.update_predictions([pred_out])

    # All cells in layer remain 0.0 for obstacle outside bounds
    stats = layer.get_slice_statistics(0.0)
    assert stats["total_affected_cells"] == 0


# ============================================================================
# TEST 16: Empty input
# ============================================================================
def test_16_empty_input() -> None:
    layer = DynamicOccupancyLayer()

    # Ingest empty list
    layer.update_predictions([])

    for t in layer.get_time_bins():
        s = layer.get_slice(t)
        assert np.all(s.cost == 0.0)
        assert np.all(s.risk_level == DynamicRiskLevel.FREE)
        assert len(s.active_obstacle_ids) == 0
