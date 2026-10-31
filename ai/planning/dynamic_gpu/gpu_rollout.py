"""Batched unicycle trajectory rollout kernel using PyTorch tensors.

Computes candidate trajectory forward rollouts using midpoint circular arc integration
with exact numerical parity to DWAPlanner._predict_trajectory.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
import torch

from ai.planning.base import RobotState
from ai.planning.dynamic_gpu.gpu_backend import GPUBackend


class GPURollout:
    """Vectorized trajectory generator for unicycle kinematic model."""

    def __init__(self, backend: GPUBackend) -> None:
        self.backend = backend
        self.device = backend.device
        self.dtype = backend.dtype

    def compute_dynamic_window(
        self,
        current_v: float,
        current_omega: float,
        dt: float,
        max_v: float,
        min_v: float,
        max_omega: float,
        max_accel_v: float,
        max_accel_omega: float,
        max_v_limit: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """Calculate reachable velocity limits based on vehicle accelerations."""
        eff_max_v = min(max_v, max_v_limit) if max_v_limit is not None else max_v
        v_low = max(min_v, current_v - max_accel_v * dt)
        v_high = min(eff_max_v, current_v + max_accel_v * dt)
        w_low = max(-max_omega, current_omega - max_accel_omega * dt)
        w_high = min(max_omega, current_omega + max_accel_omega * dt)
        return (float(v_low), float(v_high), float(w_low), float(w_high))

    def generate_candidate_grid(
        self,
        v_low: float,
        v_high: float,
        w_low: float,
        w_high: float,
        v_samples: int,
        omega_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate grid of (v, omega) candidates as tensors on device.

        Returns:
            v_tensor: (K,) linear velocities
            w_tensor: (K,) angular velocities
            where K = v_samples * omega_samples.
        """
        v_vals = torch.linspace(v_low, v_high, v_samples, device=self.device, dtype=self.dtype)
        w_vals = torch.linspace(w_low, w_high, omega_samples, device=self.device, dtype=self.dtype)

        # Meshgrid matching nested loops: for v in v_candidates: for w in omega_candidates:
        grid_v, grid_w = torch.meshgrid(v_vals, w_vals, indexing="ij")
        v_flat = grid_v.reshape(-1)
        w_flat = grid_w.reshape(-1)
        return v_flat, w_flat

    def rollout_trajectories(
        self,
        state: RobotState,
        v_flat: torch.Tensor,
        w_flat: torch.Tensor,
        pred_horizon: float,
        sim_dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate candidate trajectories forward in time.

        Args:
            state: Current robot state (x, y, yaw, timestamp).
            v_flat: (K,) linear velocities.
            w_flat: (K,) angular velocities.
            pred_horizon: Lookahead duration in seconds.
            sim_dt: Time increment per step.

        Returns:
            trajectories: Tensor of shape (K, S, 4) with [x, y, yaw, time_offset]
            times: Tensor of shape (S,) with relative future timestamps [dt, 2*dt, ...]
        """
        K = v_flat.shape[0]
        steps = max(1, int(math.ceil(pred_horizon / sim_dt)))

        trajectories = torch.zeros((K, steps, 4), device=self.device, dtype=self.dtype)

        curr_x = torch.full((K,), state.x, device=self.device, dtype=self.dtype)
        curr_y = torch.full((K,), state.y, device=self.device, dtype=self.dtype)
        curr_yaw = torch.full((K,), state.yaw, device=self.device, dtype=self.dtype)
        curr_t = torch.zeros((K,), device=self.device, dtype=self.dtype)

        dt = float(sim_dt)
        two_pi = 2.0 * math.pi
        pi = math.pi

        for s in range(steps):
            theta_mid = curr_yaw + 0.5 * w_flat * dt
            curr_x = curr_x + v_flat * torch.cos(theta_mid) * dt
            curr_y = curr_y + v_flat * torch.sin(theta_mid) * dt
            curr_yaw = curr_yaw + w_flat * dt
            # Exact Phase 13 angle normalization: (yaw + pi) % (2*pi) - pi
            curr_yaw = torch.remainder(curr_yaw + pi, two_pi) - pi
            curr_t = curr_t + dt

            trajectories[:, s, 0] = curr_x
            trajectories[:, s, 1] = curr_y
            trajectories[:, s, 2] = curr_yaw
            trajectories[:, s, 3] = curr_t

        times = torch.linspace(dt, steps * dt, steps, device=self.device, dtype=self.dtype)
        return trajectories, times
