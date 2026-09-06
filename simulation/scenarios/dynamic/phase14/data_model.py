"""Data models for Phase 13.14 Multi-Obstacle and Failure Injection Testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from simulation.scenarios.dynamic.data_model import (
    DynamicObstacleSpec,
    DynamicScenarioConfig,
    MotionModelType,
)


class Phase14FaultType(str, Enum):
    """Enumeration of systematic fault injection modes outside the frozen algorithms."""
    DROPOUT = "DROPOUT"
    STALE_TIMESTAMP = "STALE_TIMESTAMP"
    TRACK_LOSS = "TRACK_LOSS"
    TRACK_RESET = "TRACK_RESET"
    PREDICTION_DROPOUT = "PREDICTION_DROPOUT"
    PREDICTION_HORIZON_SHORTEN = "PREDICTION_HORIZON_SHORTEN"
    PREDICTION_CONFIDENCE_DEGRADE = "PREDICTION_CONFIDENCE_DEGRADE"
    ZERO_VALID_TRAJECTORIES = "ZERO_VALID_TRAJECTORIES"
    MALFORMED_PLANNER_OUTPUT = "MALFORMED_PLANNER_OUTPUT"
    REPLANNER_NO_PATH = "REPLANNER_NO_PATH"
    REPLANNER_EXCEPTION = "REPLANNER_EXCEPTION"
    REPLANNER_INVALID_RESULT = "REPLANNER_INVALID_RESULT"
    REPLANNER_TIMEOUT = "REPLANNER_TIMEOUT"
    NOISY_OBSERVATIONS = "NOISY_OBSERVATIONS"


@dataclass
class Phase14FaultSpec:
    """Formal specification for an injected fault condition."""
    fault_id: str
    fault_type: Phase14FaultType
    target_component: str
    start_time: float
    end_time: float
    expected_response: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.fault_type, str):
            self.fault_type = Phase14FaultType(self.fault_type)
        if self.start_time < 0.0:
            raise ValueError(f"start_time must be non-negative, got {self.start_time}")
        if self.end_time < self.start_time:
            raise ValueError(f"end_time ({self.end_time}) cannot be before start_time ({self.start_time})")

    def is_active(self, timestamp_s: float) -> bool:
        """Evaluate if fault is active at given simulation timestamp."""
        return self.start_time <= timestamp_s <= self.end_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "target_component": self.target_component,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "expected_response": self.expected_response,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Phase14FaultSpec:
        return cls(
            fault_id=data["fault_id"],
            fault_type=Phase14FaultType(data["fault_type"]),
            target_component=data["target_component"],
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            expected_response=data.get("expected_response", ""),
            parameters=data.get("parameters", {}),
        )


@dataclass
class Phase14ScenarioConfig(DynamicScenarioConfig):
    """Scenario configuration extended for Phase 13.14 multi-obstacle and failure suites."""
    benchmark_category: str = "MULTI_OBSTACLE"
    fault_specs: List[Phase14FaultSpec] = field(default_factory=list)
    noise_position_std: float = 0.0
    noise_confidence_std: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["benchmark_category"] = self.benchmark_category
        d["fault_specs"] = [f.to_dict() for f in self.fault_specs]
        d["noise_position_std"] = self.noise_position_std
        d["noise_confidence_std"] = self.noise_confidence_std
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Phase14ScenarioConfig:
        base = super().from_dict(data)
        faults = [Phase14FaultSpec.from_dict(f) for f in data.get("fault_specs", [])]
        return cls(
            scenario_id=base.scenario_id,
            name=base.name,
            description=base.description,
            world_bounds_m=base.world_bounds_m,
            start_pose=base.start_pose,
            goal_pose=base.goal_pose,
            obstacles=base.obstacles,
            static_obstacles=base.static_obstacles,
            control_dt_s=base.control_dt_s,
            max_duration_s=base.max_duration_s,
            random_seed=base.random_seed,
            expected_outcome=base.expected_outcome,
            goal_tolerance_m=base.goal_tolerance_m,
            benchmark_category=data.get("benchmark_category", "MULTI_OBSTACLE"),
            fault_specs=faults,
            noise_position_std=float(data.get("noise_position_std", 0.0)),
            noise_confidence_std=float(data.get("noise_confidence_std", 0.0)),
        )
