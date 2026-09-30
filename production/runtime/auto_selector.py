"""Automated Empirical Backend Selector for DWA Planning and Traversability Perception.

Executes the formal 12-step selection process:
  1. Detect hardware
  2. Enumerate candidates
  3. Filter unavailable
  4. Filter incompatible
  5. Warm up
  6. Short benchmark
  7. Validate numerical correctness
  8. Validate safety
  9. Reject failed candidates
  10. Select best validated backend
  11. Record selection
  12. Start navigation

Enforces:
- AUTO mode uses empirical benchmark + safety verification (never assumes GPU > CPU).
- Explicit backend requests FAIL CLEARLY if unavailable (zero silent fallback).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from production.runtime.backend_registry import (
    BackendCategory,
    BackendDescriptor,
    BackendRegistry,
)
from production.runtime.hardware_profiler import HardwareProfile, HardwareProfiler

logger = logging.getLogger("production.runtime.auto_selector")


@dataclass
class SelectionResult:
    category: BackendCategory
    selected_name: str
    selected_descriptor: BackendDescriptor
    benchmark_latency_ms: float
    evaluated_candidates: List[BackendDescriptor]
    selection_reason: str
    all_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Selected [{self.selected_name}] for {self.category.value} "
            f"(Benchmark: {self.benchmark_latency_ms:.2f} ms). Reason: {self.selection_reason}"
        )


class AutoBackendSelector:
    """Manages empirical calibration, safety validation, and backend selection."""

    def __init__(
        self,
        registry: Optional[BackendRegistry] = None,
        profile: Optional[HardwareProfile] = None,
    ) -> None:
        self.profile = profile or HardwareProfiler.profile()
        self.registry = registry or BackendRegistry.create_default(self.profile)

    def select_dwa_backend(
        self,
        requested: str = "AUTO",
        benchmark_iterations: int = 5,
    ) -> SelectionResult:
        """Select production DWA planner backend.

        Args:
            requested: 'AUTO' or explicit backend name ('CPP_SIMD', 'CPP_SCALAR', 'CUDA', 'INTEL_GPU').
            benchmark_iterations: Short calibration iteration count.

        Returns:
            SelectionResult with the validated backend.

        Raises:
            RuntimeError: If an explicit backend is requested and is unavailable/unsupported.
        """
        req_clean = requested.upper().strip()

        # Handle explicit backend request
        if req_clean != "AUTO":
            return self._handle_explicit_dwa_request(req_clean)

        # Handle AUTO mode
        return self._arbitrate_auto_dwa(benchmark_iterations=benchmark_iterations)

    def select_perception_backend(
        self,
        requested: str = "AUTO",
        benchmark_iterations: int = 3,
    ) -> SelectionResult:
        """Select production Traversability Perception backend."""
        req_clean = requested.upper().strip()

        if req_clean != "AUTO":
            return self._handle_explicit_perception_request(req_clean)

        return self._arbitrate_auto_perception(benchmark_iterations=benchmark_iterations)

    def _handle_explicit_dwa_request(self, requested_name: str) -> SelectionResult:
        desc = self.registry.get(requested_name)
        if desc is None:
            raise RuntimeError(
                f"Requested DWA backend '{requested_name}' is not recognized in BackendRegistry. "
                f"Valid registered backends: {[b.name for b in self.registry.get_by_category(BackendCategory.DWA_PLANNER)]}"
            )

        if not desc.supported:
            raise RuntimeError(
                f"Explicit DWA backend '{requested_name}' is unsupported: {desc.reason_if_rejected}"
            )

        if not desc.available:
            raise RuntimeError(
                f"Explicit DWA backend '{requested_name}' was requested, but hardware/runtime is unavailable on this system. "
                f"Reason: {desc.reason_if_rejected or 'Required hardware/binary not found'}. "
                f"System will NOT silently fall back to another backend."
            )

        # Run quick verification pass on explicitly requested backend
        val_ok, lat_ms, reason = self._benchmark_dwa_candidate(desc.name, iterations=3)
        if not val_ok:
            raise RuntimeError(
                f"Explicit DWA backend '{requested_name}' is available but failed safety/numerical verification: {reason}. "
                f"System will NOT silently switch to another backend."
            )

        desc.validated = True
        desc.safety_validated = True
        desc.benchmark_latency_ms = lat_ms

        return SelectionResult(
            category=BackendCategory.DWA_PLANNER,
            selected_name=desc.name,
            selected_descriptor=desc,
            benchmark_latency_ms=lat_ms,
            evaluated_candidates=[desc],
            selection_reason=f"Explicitly requested by user; verified healthy ({lat_ms:.2f} ms)",
        )

    def _handle_explicit_perception_request(self, requested_name: str) -> SelectionResult:
        desc = self.registry.get(requested_name)
        if desc is None:
            raise RuntimeError(
                f"Requested Perception backend '{requested_name}' is not recognized in BackendRegistry."
            )

        if desc.is_reference_only:
            raise RuntimeError(
                f"Perception backend '{requested_name}' is a research/reference asset (e.g. SAM-TP) "
                f"and cannot be used as the production runtime backend. Reason: {desc.reason_if_rejected}"
            )

        if not desc.supported:
            raise RuntimeError(
                f"Explicit Perception backend '{requested_name}' is unsupported: {desc.reason_if_rejected}"
            )

        if not desc.available:
            raise RuntimeError(
                f"Explicit Perception backend '{requested_name}' was requested, but runtime is unavailable. "
                f"Reason: {desc.reason_if_rejected}. System will NOT silently fall back."
            )

        val_ok, lat_ms, reason = self._benchmark_perception_candidate(desc.name, iterations=2)
        if not val_ok:
            raise RuntimeError(
                f"Explicit Perception backend '{requested_name}' failed verification: {reason}."
            )

        desc.validated = True
        desc.safety_validated = True
        desc.benchmark_latency_ms = lat_ms

        return SelectionResult(
            category=BackendCategory.TRAVERSABILITY_PERCEPTION,
            selected_name=desc.name,
            selected_descriptor=desc,
            benchmark_latency_ms=lat_ms,
            evaluated_candidates=[desc],
            selection_reason=f"Explicitly requested by user; verified ({lat_ms:.2f} ms)",
        )

    def _arbitrate_auto_dwa(self, benchmark_iterations: int) -> SelectionResult:
        candidates = self.registry.get_auto_candidates(BackendCategory.DWA_PLANNER)
        if not candidates:
            raise RuntimeError("AUTO DWA selection failed: zero available candidate backends in registry!")

        evaluated: List[BackendDescriptor] = []
        diagnostics: Dict[str, Any] = {}

        for cand in candidates:
            # Short calibration pass
            passed, lat_ms, diag_msg = self._benchmark_dwa_candidate(cand.name, iterations=benchmark_iterations)
            cand.validated = passed
            cand.safety_validated = passed
            cand.benchmark_latency_ms = lat_ms
            if passed:
                cand.benchmark_throughput_fps = round(1000.0 / lat_ms, 1) if lat_ms > 0 else 0.0
            else:
                cand.reason_if_rejected = diag_msg

            diagnostics[cand.name] = {
                "passed": passed,
                "latency_ms": lat_ms,
                "reason": diag_msg,
            }
            evaluated.append(cand)

        # Filter strictly for validated and safety-passed backends
        valid_candidates = [c for c in evaluated if c.auto_eligible]
        if not valid_candidates:
            raise RuntimeError(
                f"AUTO DWA selection failed: none of the candidate backends passed safety and latency validation! "
                f"Evaluated: {diagnostics}"
            )

        # Empirical selection: Lowest latency among strictly safety-validated candidates
        valid_candidates.sort(key=lambda c: (c.benchmark_latency_ms or float("inf")))
        selected = valid_candidates[0]

        reason = (
            f"Empirically calibrated fastest safety-validated backend ({selected.benchmark_latency_ms:.2f} ms). "
            f"Outperformed {len(valid_candidates) - 1} other validated candidates."
        )

        return SelectionResult(
            category=BackendCategory.DWA_PLANNER,
            selected_name=selected.name,
            selected_descriptor=selected,
            benchmark_latency_ms=selected.benchmark_latency_ms or 0.0,
            evaluated_candidates=evaluated,
            selection_reason=reason,
            all_diagnostics=diagnostics,
        )

    def _arbitrate_auto_perception(self, benchmark_iterations: int) -> SelectionResult:
        candidates = self.registry.get_auto_candidates(BackendCategory.TRAVERSABILITY_PERCEPTION)
        if not candidates:
            raise RuntimeError("AUTO Perception selection failed: zero available candidate backends in registry!")

        evaluated: List[BackendDescriptor] = []
        diagnostics: Dict[str, Any] = {}

        for cand in candidates:
            passed, lat_ms, diag_msg = self._benchmark_perception_candidate(cand.name, iterations=benchmark_iterations)
            cand.validated = passed
            cand.safety_validated = passed
            cand.benchmark_latency_ms = lat_ms
            if passed:
                cand.benchmark_throughput_fps = round(1000.0 / lat_ms, 1) if lat_ms > 0 else 0.0
            else:
                cand.reason_if_rejected = diag_msg

            diagnostics[cand.name] = {
                "passed": passed,
                "latency_ms": lat_ms,
                "reason": diag_msg,
            }
            evaluated.append(cand)

        valid_candidates = [c for c in evaluated if c.auto_eligible]
        if not valid_candidates:
            raise RuntimeError(
                f"AUTO Perception selection failed: none of the candidates passed validation! Evaluated: {diagnostics}"
            )

        valid_candidates.sort(key=lambda c: (c.benchmark_latency_ms or float("inf")))
        selected = valid_candidates[0]

        reason = (
            f"Empirically calibrated fastest safety-validated perception backend ({selected.benchmark_latency_ms:.2f} ms)."
        )

        return SelectionResult(
            category=BackendCategory.TRAVERSABILITY_PERCEPTION,
            selected_name=selected.name,
            selected_descriptor=selected,
            benchmark_latency_ms=selected.benchmark_latency_ms or 0.0,
            evaluated_candidates=evaluated,
            selection_reason=reason,
            all_diagnostics=diagnostics,
        )

    def _benchmark_dwa_candidate(self, backend_name: str, iterations: int = 5) -> Tuple[bool, float, str]:
        """Perform short empirical benchmark and safety verification of a DWA backend."""
        try:
            from ai.planning.backends.backend_interface import DWABackendType
            from ai.planning.backends.unified_planner import UnifiedDWAPlanner
            from ai.planning.base import PlanningConfig, RobotState
            from ai.planning.path import Path
            from ai.spatial_fusion.world_map import WorldTerrainMap

            planner = UnifiedDWAPlanner(config=PlanningConfig(), backend=backend_name)

            # Construct synthetic benchmark problem
            state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.5, omega=0.0)
            global_path = Path.from_xy_array(np.array([[float(i), 0.0] for i in range(10)]))
            costmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
            costmap.cost.fill(0.10)
            costmap.obstacle.fill(False)
            costmap.hazard.fill(0.0)
            # Add an obstacle ahead at x=3.0, y=0.0
            r, c = costmap.world_to_grid(3.0, 0.0)
            costmap.cost[r, c] = 1.00
            costmap.obstacle[r, c] = True

            # Warmup
            planner.plan(state, global_path, costmap)

            # Calibration passes
            latencies: List[float] = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                traj, v, w, status, _ = planner.plan(state, global_path, costmap)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt_ms)

                # Numerical & Safety invariants
                if math.isnan(v) or math.isnan(w) or math.isinf(v) or math.isinf(w):
                    return False, 0.0, "Produced NaN or Inf velocity command"
                if abs(v) > 2.0 or abs(w) > 3.0:
                    return False, 0.0, f"Command exceeds dynamic envelope (v={v}, w={w})"
                if traj is None or len(traj.points) == 0:
                    return False, 0.0, "Computed empty trajectory"

            mean_lat = float(np.mean(latencies))
            return True, round(mean_lat, 2), "Safety and numerical invariants verified"

        except Exception as e:
            return False, 0.0, f"Exception during calibration: {e}"

    def _benchmark_perception_candidate(self, backend_name: str, iterations: int = 2) -> Tuple[bool, float, str]:
        """Perform short calibration on LightweightTraversabilityNet inference."""
        try:
            from ai.traversability.dense_predictor import DenseTraversabilityPredictor

            device_map = {
                "PERCEPTION_CPU": "cpu",
                "PERCEPTION_SIMD": "cpu",
                "PERCEPTION_CUDA": "cuda:0",
                "PERCEPTION_INTEL_GPU": "xpu:0",
            }
            target_device = device_map.get(backend_name, "cpu")

            predictor = DenseTraversabilityPredictor(device=target_device)
            # Warmup
            dummy_img = np.zeros((400, 640, 3), dtype=np.uint8)
            predictor.predict(dummy_img)

            latencies: List[float] = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                res = predictor.predict(dummy_img)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt_ms)

                # Numerical verification
                if res.mask is None or np.isnan(res.mask).any():
                    return False, 0.0, "Traversability mask contains NaN"
                if res.mask.shape != (400, 640):
                    return False, 0.0, f"Unexpected mask output shape {res.mask.shape}"

            mean_lat = float(np.mean(latencies))
            return True, round(mean_lat, 2), "Perception output verified"

        except Exception as e:
            return False, 0.0, f"Exception during calibration: {e}"
