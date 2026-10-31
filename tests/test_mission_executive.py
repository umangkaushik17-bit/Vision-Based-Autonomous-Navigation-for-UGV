"""Unit tests for Phase 14.1 Mission Executive, WaypointManager, and state machine invariants."""

import math
from pathlib import Path
import pytest

from ugv_mission import (
    MissionAbortReason,
    MissionExecutive,
    MissionExecutiveConfig,
    MissionState,
    Waypoint,
    WaypointManager,
    is_valid_mission_transition,
)


# ==============================================================================
# CATEGORY A: Mission Loading & Validation Tests
# ==============================================================================

def test_mission_loading_valid():
    """Verify that a valid ordered waypoint list loads correctly."""
    mgr = WaypointManager("test_mission")
    waypoints = [
        Waypoint(id="W1", x=2.0, y=0.0, xy_tolerance_m=0.35),
        Waypoint(id="W2", x=4.0, y=1.0, xy_tolerance_m=0.35),
        Waypoint(id="W3", x=6.0, y=0.0, xy_tolerance_m=0.35),
    ]
    ok, reason = mgr.load_waypoints(waypoints)
    assert ok is True
    assert mgr.total_count == 3
    assert mgr.active_index == 0
    assert mgr.active_waypoint.id == "W1"
    assert mgr.active_waypoint.position == (2.0, 0.0)


def test_mission_loading_empty_rejected():
    """Assert: Empty waypoint list is rejected and does not corrupt manager state."""
    mgr = WaypointManager()
    ok, reason = mgr.load_waypoints([])
    assert ok is False
    assert "empty" in reason.lower()
    assert mgr.total_count == 0
    assert mgr.active_waypoint is None


def test_mission_loading_invalid_coordinates():
    """Assert: Infinite or NaN coordinates are strictly rejected."""
    mgr = WaypointManager()
    bad_waypoints = [
        Waypoint(id="W1", x=float("nan"), y=0.0),
    ]
    ok, reason = mgr.load_waypoints(bad_waypoints)
    assert ok is False
    assert "finite" in reason.lower()

    bad_waypoints_inf = [
        Waypoint(id="W1", x=2.0, y=float("inf")),
    ]
    ok, reason = mgr.load_waypoints(bad_waypoints_inf)
    assert ok is False
    assert "finite" in reason.lower()


def test_mission_loading_invalid_tolerance():
    """Assert: Non-positive arrival tolerance is rejected."""
    mgr = WaypointManager()
    bad_tol = [
        Waypoint(id="W1", x=2.0, y=0.0, xy_tolerance_m=-0.10),
    ]
    ok, reason = mgr.load_waypoints(bad_tol)
    assert ok is False
    assert "positive" in reason.lower()


def test_mission_loading_from_yaml():
    """Verify loading from canonical YAML mission file."""
    yaml_path = Path("configs/mission_phase14_1_5wp.yaml")
    assert yaml_path.exists(), "configs/mission_phase14_1_5wp.yaml must exist"

    mgr = WaypointManager()
    ok, reason = mgr.load_from_yaml(yaml_path)
    assert ok is True, f"YAML loading failed: {reason}"
    assert mgr.total_count == 5
    assert [wp.id for wp in mgr.waypoints] == ["W1", "W2", "W3", "W4", "W5"]
    assert mgr.active_waypoint.id == "W1"


# ==============================================================================
# CATEGORY B: Waypoint Sequencing & Invariants
# ==============================================================================

def test_waypoint_sequencing_order():
    """Verify strictly ordered execution W1 -> W2 -> W3 -> W4 -> W5 without skipping."""
    mgr = WaypointManager()
    wps = [
        Waypoint(id="W1", x=1.0, y=0.0, xy_tolerance_m=0.30),
        Waypoint(id="W2", x=2.0, y=0.0, xy_tolerance_m=0.30),
        Waypoint(id="W3", x=3.0, y=0.0, xy_tolerance_m=0.30),
        Waypoint(id="W4", x=4.0, y=0.0, xy_tolerance_m=0.30),
        Waypoint(id="W5", x=5.0, y=0.0, xy_tolerance_m=0.30),
    ]
    mgr.load_waypoints(wps)

    # Initial state
    assert mgr.active_index == 0
    assert mgr.active_waypoint.id == "W1"
    assert mgr.is_final_waypoint is False

    # Attempting to advance before reaching W1 must raise an invariant error
    with pytest.raises(RuntimeError):
        mgr.advance()

    # Reach W1
    assert mgr.check_active_reached((1.0, 0.05, 0.0)) is True
    mgr.mark_active_reached((1.0, 0.05, 0.0))
    next_wp = mgr.advance()
    assert next_wp.id == "W2"
    assert mgr.active_index == 1
    assert mgr.active_waypoint.id == "W2"

    # Reach W2
    mgr.mark_active_reached((2.0, 0.0, 0.0))
    next_wp = mgr.advance()
    assert next_wp.id == "W3"
    assert mgr.active_index == 2

    # Reach W3
    mgr.mark_active_reached((3.0, 0.0, 0.0))
    next_wp = mgr.advance()
    assert next_wp.id == "W4"
    assert mgr.active_index == 3

    # Reach W4
    mgr.mark_active_reached((4.0, 0.0, 0.0))
    next_wp = mgr.advance()
    assert next_wp.id == "W5"
    assert mgr.active_index == 4
    assert mgr.is_final_waypoint is True

    # Reach W5 (Final)
    mgr.mark_active_reached((5.0, 0.0, 0.0))
    next_wp = mgr.advance()
    assert next_wp is None
    assert mgr.is_complete is True
    assert mgr.reached_ids == ["W1", "W2", "W3", "W4", "W5"]


# ==============================================================================
# CATEGORY C: Arrival Condition Precision & Tolerance
# ==============================================================================

def test_arrival_condition_precision():
    """Verify arrival conditions: inside, exactly on, near outside, and far outside tolerance."""
    wp = Waypoint(id="W_test", x=5.0, y=0.0, xy_tolerance_m=0.35)

    # 1. Inside tolerance (dist = 0.10m <= 0.35m)
    assert wp.is_reached((5.10, 0.0, 0.0)) is True

    # 2. Exactly on tolerance boundary (dist = 0.35m <= 0.35m)
    assert wp.is_reached((5.35, 0.0, 0.0)) is True

    # 3. Near but outside tolerance (dist = 0.36m > 0.35m)
    assert wp.is_reached((5.36, 0.0, 0.0)) is False

    # 4. Far outside tolerance (dist = 1.0m > 0.35m)
    assert wp.is_reached((6.0, 0.0, 0.0)) is False


def test_arrival_condition_with_heading():
    """Verify heading tolerance validation when yaw is specified."""
    wp = Waypoint(
        id="W_head",
        x=5.0,
        y=0.0,
        yaw=0.0,
        xy_tolerance_m=0.35,
        yaw_tolerance_rad=0.30,  # ~17.2 deg
    )

    # In position and aligned heading (yaw error = 0.10 rad <= 0.30 rad)
    assert wp.is_reached((5.10, 0.0, 0.10)) is True

    # In position but misaligned heading (yaw error = 0.50 rad > 0.30 rad)
    assert wp.is_reached((5.10, 0.0, 0.50)) is False


# ==============================================================================
# CATEGORY G: Final Waypoint Completion Requirements
# ==============================================================================

def test_final_waypoint_required_for_completion():
    """Assert: Mission is NOT complete until the final waypoint is reached."""
    exec_engine = MissionExecutive(MissionExecutiveConfig(mission_name="completion_check"))
    wps = [
        Waypoint(id="W1", x=1.0, y=0.0, xy_tolerance_m=0.30),
        Waypoint(id="W2", x=2.0, y=0.0, xy_tolerance_m=0.30),
    ]
    exec_engine.load_mission(wps)
    exec_engine.start_mission(current_pose=(0.0, 0.0, 0.0))

    assert exec_engine.state == MissionState.NAVIGATING_TO_WAYPOINT
    assert exec_engine.active_waypoint.id == "W1"

    # Step at (0.5, 0.0) -> outside W1 tolerance
    state, _, _, _ = exec_engine.step(current_time_s=1.0, robot_pose=(0.5, 0.0, 0.0))
    assert state == MissionState.NAVIGATING_TO_WAYPOINT
    assert exec_engine.state != MissionState.MISSION_COMPLETE

    # Step at W1 (1.0, 0.0) -> Reaches W1 and advances to W2
    state, _, _, _ = exec_engine.step(current_time_s=2.0, robot_pose=(1.0, 0.0, 0.0))
    assert state == MissionState.NAVIGATING_TO_WAYPOINT
    assert exec_engine.active_waypoint.id == "W2"
    assert exec_engine.state != MissionState.MISSION_COMPLETE

    # Step near W2 (1.5, 0.0) -> Not reached W2 yet
    state, _, _, _ = exec_engine.step(current_time_s=3.0, robot_pose=(1.5, 0.0, 0.0))
    assert state == MissionState.NAVIGATING_TO_WAYPOINT
    assert exec_engine.state != MissionState.MISSION_COMPLETE

    # Step at W2 (2.0, 0.0) -> Reaches final waypoint!
    state, cmd_v, cmd_w, reason = exec_engine.step(current_time_s=4.0, robot_pose=(2.0, 0.0, 0.0))
    assert state == MissionState.MISSION_COMPLETE
    assert exec_engine.state == MissionState.MISSION_COMPLETE
    assert cmd_v == 0.0 and cmd_w == 0.0


# ==============================================================================
# CATEGORY I: Invalid State Transitions
# ==============================================================================

def test_invalid_state_transitions():
    """Verify that invalid state transitions are rejected and do not corrupt state."""
    exec_engine = MissionExecutive()

    # 1. IDLE -> WAYPOINT_REACHED is invalid
    assert not is_valid_mission_transition(MissionState.IDLE, MissionState.WAYPOINT_REACHED)
    with pytest.raises(RuntimeError):
        exec_engine._transition_to(MissionState.WAYPOINT_REACHED)

    # 2. Set to MISSION_COMPLETE; transitioning to NAVIGATING directly is invalid
    exec_engine._state = MissionState.MISSION_COMPLETE
    assert not is_valid_mission_transition(MissionState.MISSION_COMPLETE, MissionState.NAVIGATING_TO_WAYPOINT)
    with pytest.raises(RuntimeError):
        exec_engine._transition_to(MissionState.NAVIGATING_TO_WAYPOINT)

    # 3. Set to MISSION_ABORTED; transitioning to NAVIGATING without reset is invalid
    exec_engine._state = MissionState.MISSION_ABORTED
    assert not is_valid_mission_transition(MissionState.MISSION_ABORTED, MissionState.NAVIGATING_TO_WAYPOINT)
    with pytest.raises(RuntimeError):
        exec_engine._transition_to(MissionState.NAVIGATING_TO_WAYPOINT)

    # 4. After reset, state returns to IDLE, allowing valid startup
    exec_engine.reset()
    assert exec_engine.state == MissionState.IDLE
    assert is_valid_mission_transition(MissionState.IDLE, MissionState.INITIALIZING)
