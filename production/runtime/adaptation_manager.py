"""Runtime Adaptation Manager for Autonomous UGV Navigation.

Monitors perception latency, planning latency, control-loop frequency, and command freshness.
Transitions deterministically between operating states:
  - NORMAL
  - CONSERVATIVE
  - SAFE_STOP

Enforces the absolute safety invariant:
  LOWER COMPUTE CAPABILITY MUST NEVER CAUSE INCREASED NAVIGATION RISK.
  COMPUTATIONAL DEGRADATION NEVER INCREASES SPEED; IT ONLY REDUCES SPEED OR ENFORCES SAFE STOP.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("production.runtime.adaptation_manager")


class OperationalState(str, Enum):
    NORMAL = "NORMAL"
    CONSERVATIVE = "CONSERVATIVE"
    SAFE_STOP = "SAFE_STOP"


@dataclass
class AdaptationConfig:
    """PROVISIONAL COMPUTATIONAL SAFETY THRESHOLDS.

    NOTE: These thresholds are provisional engineering configurations derived from
    nominal measured cycle budgets and command freshness timeouts.
    They are NOT vehicle-specific dynamically certified braking limits on physical outdoor terrain;
    vehicle-specific physical braking-distance and closed-loop field validation remain required before
    physical outdoor deployment at higher velocities.
    """
    nominal_control_freq_hz: float = 20.0
    nominal_max_linear_v_mps: float = 0.50
    nominal_planning_budget_ms: float = 15.0     # C++ DWA nominal budget (~1-5 ms)
    nominal_perception_budget_ms: float = 120.0  # LightweightTraversabilityNet nominal budget (~95-115 ms)
    conservative_latency_multiplier: float = 1.5
    critical_latency_multiplier: float = 3.0
    conservative_speed_scale: float = 0.50       # Reduce velocity by 50% in CONSERVATIVE mode
    command_freshness_timeout_s: float = 0.50    # Reject commands older than 500 ms
    recovery_hysteresis_cycles: int = 5         # Cycles in healthy envelope required to recover to NORMAL


@dataclass
class RuntimeMetrics:
    timestamp_s: float
    perception_latency_ms: float
    planning_latency_ms: float
    total_loop_latency_ms: float
    command_age_s: float = 0.0
    dropped_frames: int = 0
    active_dwa_backend: str = "CPP_SIMD"
    active_perception_backend: str = "PERCEPTION_SIMD"


@dataclass
class AdaptationDecision:
    state: OperationalState
    max_allowed_linear_v_mps: float
    command_v: float
    command_w: float
    emergency_stop: bool
    reason: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class RuntimeAdaptationManager:
    """Oversees runtime computational health and adapts vehicle execution parameters conservatively."""

    def __init__(self, config: Optional[AdaptationConfig] = None) -> None:
        self.config = config or AdaptationConfig()
        self.current_state: OperationalState = OperationalState.NORMAL
        self._consecutive_healthy_cycles: int = 0
        self._history: List[RuntimeMetrics] = []

    def evaluate_cycle(
        self,
        metrics: RuntimeMetrics,
        raw_cmd_v: float,
        raw_cmd_w: float,
    ) -> AdaptationDecision:
        """Evaluate cycle computational metrics and adapt commanded velocities safely.

        Args:
            metrics: Current timing and resource measurements.
            raw_cmd_v: Commanded forward linear velocity from local planner (m/s).
            raw_cmd_w: Commanded angular velocity from local planner (rad/s).

        Returns:
            AdaptationDecision with safely clamped command velocities and state diagnosis.
        """
        self._history.append(metrics)
        if len(self._history) > 100:
            self._history.pop(0)

        # Thresholds
        crit_perc = self.config.nominal_perception_budget_ms * self.config.critical_latency_multiplier
        crit_plan = self.config.nominal_planning_budget_ms * self.config.critical_latency_multiplier
        warn_perc = self.config.nominal_perception_budget_ms * self.config.conservative_latency_multiplier
        warn_plan = self.config.nominal_planning_budget_ms * self.config.conservative_latency_multiplier

        reasons: List[str] = []
        target_state = OperationalState.NORMAL

        # 1. Critical Envelope Violations -> SAFE_STOP
        if metrics.command_age_s > self.config.command_freshness_timeout_s:
            target_state = OperationalState.SAFE_STOP
            reasons.append(f"Command stale: age={metrics.command_age_s:.3f}s > {self.config.command_freshness_timeout_s:.3f}s")

        if metrics.planning_latency_ms > crit_plan:
            target_state = OperationalState.SAFE_STOP
            reasons.append(f"Critical planning lag: {metrics.planning_latency_ms:.1f}ms > {crit_plan:.1f}ms")

        if metrics.perception_latency_ms > crit_perc:
            target_state = OperationalState.SAFE_STOP
            reasons.append(f"Critical perception lag: {metrics.perception_latency_ms:.1f}ms > {crit_perc:.1f}ms")

        # 2. Conservative Warning Envelope -> CONSERVATIVE (if not already critical)
        if target_state != OperationalState.SAFE_STOP:
            if metrics.planning_latency_ms > warn_plan:
                target_state = OperationalState.CONSERVATIVE
                reasons.append(f"Elevated planning latency: {metrics.planning_latency_ms:.1f}ms > {warn_plan:.1f}ms")

            if metrics.perception_latency_ms > warn_perc:
                target_state = OperationalState.CONSERVATIVE
                reasons.append(f"Elevated perception latency: {metrics.perception_latency_ms:.1f}ms > {warn_perc:.1f}ms")

        # 3. State Transition Logic with Hysteresis
        if target_state == OperationalState.SAFE_STOP:
            self.current_state = OperationalState.SAFE_STOP
            self._consecutive_healthy_cycles = 0
        elif target_state == OperationalState.CONSERVATIVE:
            self.current_state = OperationalState.CONSERVATIVE
            self._consecutive_healthy_cycles = 0
        else:
            # Candidate is NORMAL
            if self.current_state != OperationalState.NORMAL:
                self._consecutive_healthy_cycles += 1
                if self._consecutive_healthy_cycles >= self.config.recovery_hysteresis_cycles:
                    self.current_state = OperationalState.NORMAL
                    self._consecutive_healthy_cycles = 0
                else:
                    # Hold conservative until hysteresis clears
                    self.current_state = OperationalState.CONSERVATIVE
                    reasons.append(f"Clearing degradation hysteresis: {self._consecutive_healthy_cycles}/{self.config.recovery_hysteresis_cycles}")
            else:
                self.current_state = OperationalState.NORMAL
                self._consecutive_healthy_cycles = 0

        # 4. Action Enforcements
        if self.current_state == OperationalState.SAFE_STOP:
            max_v = 0.0
            cmd_v = 0.0
            cmd_w = 0.0
            stop = True
            reason_str = "SAFE_STOP enforced due to computational degradation: " + "; ".join(reasons)
        elif self.current_state == OperationalState.CONSERVATIVE:
            max_v = self.config.nominal_max_linear_v_mps * self.config.conservative_speed_scale
            # Clamp commanded velocity to conservative ceiling
            cmd_v = math.copysign(min(abs(raw_cmd_v), max_v), raw_cmd_v) if abs(raw_cmd_v) > 0 else 0.0
            cmd_w = raw_cmd_w  # Rotation remains available for localized maneuvering
            stop = False
            reason_str = f"CONSERVATIVE mode active (max_v clamped to {max_v:.2f} m/s): " + "; ".join(reasons)
        else:
            max_v = self.config.nominal_max_linear_v_mps
            cmd_v = math.copysign(min(abs(raw_cmd_v), max_v), raw_cmd_v) if abs(raw_cmd_v) > 0 else 0.0
            cmd_w = raw_cmd_w
            stop = False
            reason_str = "NORMAL operating envelope: compute latency nominal"

        return AdaptationDecision(
            state=self.current_state,
            max_allowed_linear_v_mps=max_v,
            command_v=cmd_v,
            command_w=cmd_w,
            emergency_stop=stop,
            reason=reason_str,
            diagnostics={
                "state": self.current_state.value,
                "planning_latency_ms": metrics.planning_latency_ms,
                "perception_latency_ms": metrics.perception_latency_ms,
                "command_age_s": metrics.command_age_s,
                "max_allowed_v": max_v,
                "active_dwa_backend": metrics.active_dwa_backend,
                "active_perception_backend": metrics.active_perception_backend,
            },
        )
