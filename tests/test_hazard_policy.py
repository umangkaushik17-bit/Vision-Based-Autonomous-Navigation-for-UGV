"""Unit tests for Configurable Hazard Policy."""

import pytest
from pathlib import Path

from ai.obstacles.base import HazardType, RiskLevel
from ai.perception_fusion.hazard_policy import HazardPolicy


@pytest.fixture
def policy():
    return HazardPolicy()


def test_hazard_policy_loading(policy):
    """Verify hazard policy loads and parses risk tiers."""
    assert "CRITICAL" in policy.risk_tiers
    assert "HIGH" in policy.risk_tiers
    assert "MEDIUM" in policy.risk_tiers
    assert "LOW" in policy.risk_tiers
    assert "UNKNOWN" in policy.risk_tiers

    assert policy.get_risk_cost(RiskLevel.CRITICAL) == 1.00
    assert policy.get_risk_cost(RiskLevel.HIGH) == 0.85
    assert policy.get_risk_cost(RiskLevel.LOW) == 0.15


def test_hazard_class_evaluation(policy):
    """Verify class mapping rules for standard hazard classes."""
    h_type, risk = policy.evaluate_class("person")
    assert h_type == HazardType.PERSON
    assert risk == RiskLevel.CRITICAL

    h_type, risk = policy.evaluate_class("vehicle")
    assert h_type == HazardType.VEHICLE
    assert risk == RiskLevel.CRITICAL

    h_type, risk = policy.evaluate_class("water")
    assert h_type == HazardType.WATER
    assert risk == RiskLevel.CRITICAL

    h_type, risk = policy.evaluate_class("large_rock")
    assert h_type == HazardType.ROCK
    assert risk == RiskLevel.HIGH


def test_hazard_class_fallback(policy):
    """Verify fallback keyword detection for unknown classes."""
    h_type, risk = policy.evaluate_class("deep_creek_pool")
    assert h_type == HazardType.WATER
    assert risk == RiskLevel.CRITICAL

    h_type, risk = policy.evaluate_class("random_boulder_cluster")
    assert h_type == HazardType.ROCK
    assert risk == RiskLevel.HIGH


def test_speed_factor(policy):
    """Verify speed scaling factor retrieval."""
    assert policy.get_speed_factor(RiskLevel.CRITICAL) == 0.00
    assert policy.get_speed_factor(RiskLevel.HIGH) == 0.25
    assert policy.get_speed_factor(RiskLevel.LOW) == 0.90
