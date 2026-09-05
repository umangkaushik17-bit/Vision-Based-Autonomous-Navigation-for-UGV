"""Unit tests for TemporalCostmapAccumulator."""

import pytest
import numpy as np

from ai.costmap.base import CostmapMetadata
from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.costmap.temporal import TemporalCostmapAccumulator


@pytest.fixture
def test_metadata():
    return CostmapMetadata(
        resolution_m=0.10,
        shape=(20, 20),
        x_range_m=(0.0, 2.0),
        y_range_m=(-1.0, 1.0)
    )


def make_test_costmap(metadata, trav_val=0.85, has_obstacle=False):
    """Helper to create a simple test costmap."""
    cost = np.full(metadata.shape, 1.0 - trav_val, dtype=np.float32)
    trav = np.full(metadata.shape, trav_val, dtype=np.float32)
    obs = np.zeros(metadata.shape, dtype=bool)
    hazard = np.zeros(metadata.shape, dtype=np.float32)

    if has_obstacle:
        obs[10:12, 10:12] = True
        hazard[10:12, 10:12] = 1.00
        cost[10:12, 10:12] = 1.00
        trav[10:12, 10:12] = 0.00

    return BEVTerrainCostmap(
        cost=cost,
        traversability=trav,
        obstacle=obs,
        hazard=hazard,
        unknown=np.zeros(metadata.shape, dtype=bool),
        confidence=np.full(metadata.shape, 0.80, dtype=np.float32),
        slope_deg=np.zeros(metadata.shape, dtype=np.float32),
        roughness=np.zeros(metadata.shape, dtype=np.float32),
        height_var=np.zeros(metadata.shape, dtype=np.float32),
        observation_count=np.ones(metadata.shape, dtype=np.int32),
        metadata=metadata
    )


def test_temporal_accumulation_confidence(test_metadata):
    """Repeated observations increase confidence."""
    temporal = TemporalCostmapAccumulator(test_metadata)
    m1 = make_test_costmap(test_metadata)

    res1 = temporal.update(m1)
    conf_1 = res1.confidence[5, 5]

    res2 = temporal.update(m1)
    conf_2 = res2.confidence[5, 5]

    assert conf_2 > conf_1
    assert res2.observation_count[5, 5] == 2


def test_sensor_dropout_resilience(test_metadata):
    """Temporary 1-frame dropout does not instantly erase a confirmed obstacle."""
    temporal = TemporalCostmapAccumulator(test_metadata, dropout_tolerance_frames=3)
    m_obs = make_test_costmap(test_metadata, has_obstacle=True)

    # Observe obstacle twice to confirm it
    temporal.update(m_obs)
    res_confirmed = temporal.update(m_obs)
    assert bool(res_confirmed.obstacle[10, 10]) is True

    # Drop frame: zero observations across map
    m_drop = make_test_costmap(test_metadata, has_obstacle=False)
    m_drop.observation_count.fill(0)

    res_dropout = temporal.update(m_drop)
    # Due to dropout tolerance, obstacle is preserved through temporary sensor blackout
    assert bool(res_dropout.obstacle[10, 10]) is True


def test_dynamic_obstacle_clearing(test_metadata):
    """When a dynamic obstacle moves away and clean drivability is observed, it clears."""
    temporal = TemporalCostmapAccumulator(test_metadata, dynamic_decay_factor=0.50)
    m_obs = make_test_costmap(test_metadata, has_obstacle=True)

    # Initial detection
    temporal.update(m_obs)

    # Obstacle moves away: clean safe ground observed at former position
    m_clear = make_test_costmap(test_metadata, trav_val=0.95, has_obstacle=False)

    res1 = temporal.update(m_clear)
    res2 = temporal.update(m_clear)

    # Obstacle is cleared and hazard score decays rapidly
    assert bool(res2.obstacle[10, 10]) is False
    assert res2.hazard[10, 10] < 0.30


def test_stale_cells_decay_to_unknown(test_metadata):
    """Cells unobserved for max_age_frames revert to UNKNOWN."""
    temporal = TemporalCostmapAccumulator(test_metadata, max_age_frames=5)
    m1 = make_test_costmap(test_metadata)

    temporal.update(m1)

    # Simulate 6 empty frames where camera looks away
    m_empty = make_test_costmap(test_metadata)
    m_empty.observation_count.fill(0)

    for _ in range(6):
        res = temporal.update(m_empty)

    # Stale cells must revert to UNKNOWN with neutral cost
    assert bool(res.unknown[5, 5]) is True
    assert res.cost[5, 5] == 0.50
