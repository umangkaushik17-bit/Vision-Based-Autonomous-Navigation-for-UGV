"""Authoritative Production Safety Governor.

Guards the vehicle against physical collisions, velocity boundary violations,
stale commands, and computational latency degradation.
Authoritative downstream filter: NO backend may bypass the Safety Governor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

from ai.planning.dynamic.data_model import TrackedObstacleState
from ai.planning.safety_governor import (
    LatchedHazard,
    SafetyGovernor,
    SafetyGovernorConfig,
)
from production.runtime.adaptation_manager import (
    AdaptationDecision,
    OperationalState,
    RuntimeAdaptationManager,
    RuntimeMetrics,
)


class AuthoritativeSafetyGovernor(SafetyGovernor):
    """Authoritative downstream Safety Governor integrating obstacle protection and compute adaptation."""

    def __init__(
        self,
        config: Optional[SafetyGovernorConfig] = None,
        adaptation_manager: Optional[RuntimeAdaptationManager] = None,
    ) -> None:
        super().__init__(config=config)
        self.adaptation_manager = adaptation_manager or RuntimeAdaptationManager()

    def evaluate_safe_command(
        self,
        requested_v: float,
        requested_w: float,
        robot_pose: Tuple[float, float, float],
        hazards: Sequence[Union[TrackedObstacleState, LatchedHazard]],
        metrics: RuntimeMetrics,
        current_time_s: float = 0.0,
    ) -> Tuple[float, float, str, Dict[str, Any]]:
        """Jointly evaluate obstacle safety and computational latency degradation.

        Returns:
            Tuple of (governed_v, governed_w, final_decision, combined_diagnostics)
        """
        # 1. Evaluate computational adaptation (watchdog on latency and freshness)
        adapt_decision: AdaptationDecision = self.adaptation_manager.evaluate_cycle(
            metrics=metrics,
            raw_cmd_v=requested_v,
            raw_cmd_w=requested_w,
        )

        # 2. Evaluate physical obstacle safety with clamped velocity
        gov_v, gov_w, obs_decision, obs_diag = self.evaluate(
            requested_v=adapt_decision.command_v,
            requested_w=adapt_decision.command_w,
            robot_pose=robot_pose,
            hazards=hazards,
            current_time_s=current_time_s,
        )

        # 3. Combine decisions: SAFE_STOP or physical STOP overrides everything
        final_v = gov_v
        final_w = gov_w
        final_decision = obs_decision

        if adapt_decision.emergency_stop:
            final_v = 0.0
            final_w = 0.0
            final_decision = "COMPUTE_EMERGENCY_STOP"

        combined_diag = {
            **obs_diag,
            "adaptation": adapt_decision.diagnostics,
            "operational_state": adapt_decision.state.value,
            "adaptation_reason": adapt_decision.reason,
        }

        return final_v, final_w, final_decision, combined_diag


__all__ = [
    "SafetyGovernorConfig",
    "SafetyGovernor",
    "LatchedHazard",
    "AuthoritativeSafetyGovernor",
]
