"""Hardware-accelerated backend abstraction supporting PyTorch CUDA and CPU vectorization.

Manages device selection, memory preallocation, tensor transfers (H2D / D2H),
synchronization, and execution timing diagnostics.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


class GPUBackend:
    """Hardware acceleration backend managing device allocation and tensor buffers."""

    def __init__(
        self,
        preferred_device: str = "auto",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize accelerator backend.

        Args:
            preferred_device: 'auto' (cuda if available, else cpu), 'cuda', or 'cpu'.
            dtype: Tensor numerical precision (default: torch.float64 for strict Phase 13 parity).
        """
        self.dtype = dtype
        if preferred_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
            self.device = torch.device("cuda:0")
        elif preferred_device == "cpu":
            self.device = torch.device("cpu")
        else:  # auto
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.is_cuda = self.device.type == "cuda"
        self.device_name = torch.cuda.get_device_name(self.device) if self.is_cuda else "CPU (Vectorized)"

        # Preallocated buffers dictionary
        self._buffers: Dict[str, torch.Tensor] = {}

        # Timing diagnostics for last execution cycle
        self.last_timing: Dict[str, float] = {
            "h2d_ms": 0.0,
            "kernel_ms": 0.0,
            "d2h_ms": 0.0,
            "sync_ms": 0.0,
            "total_backend_ms": 0.0,
        }

    def synchronize(self) -> float:
        """Synchronize accelerator stream and measure synchronization overhead.

        Returns:
            Synchronization duration in milliseconds.
        """
        if not self.is_cuda:
            return 0.0
        t0 = time.perf_counter()
        torch.cuda.synchronize(self.device)
        sync_ms = (time.perf_counter() - t0) * 1000.0
        return sync_ms

    def to_tensor(
        self,
        data: Union[np.ndarray, list, float, int],
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Convert input array to backend tensor on target device and record H2D transfer time.

        Returns:
            tensor: Tensor on backend device.
            transfer_ms: Host-to-Device transfer time in milliseconds.
        """
        dt = dtype or self.dtype
        t0 = time.perf_counter()
        if isinstance(data, torch.Tensor):
            tensor = data.to(device=self.device, dtype=dt)
        else:
            tensor = torch.tensor(data, device=self.device, dtype=dt)
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        transfer_ms = (time.perf_counter() - t0) * 1000.0
        return tensor, transfer_ms

    def to_numpy(self, tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Transfer tensor back to host memory as NumPy array and record D2H transfer time.

        Returns:
            array: NumPy array on host CPU.
            transfer_ms: Device-to-Host transfer time in milliseconds.
        """
        t0 = time.perf_counter()
        if tensor.is_cuda:
            arr = tensor.detach().cpu().numpy()
        else:
            arr = tensor.detach().numpy()
        transfer_ms = (time.perf_counter() - t0) * 1000.0
        return arr, transfer_ms

    def get_or_create_buffer(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Retrieve preallocated tensor buffer or allocate if missing/shape changed."""
        dt = dtype or self.dtype
        if name in self._buffers:
            buf = self._buffers[name]
            if buf.shape == shape and buf.dtype == dt:
                return buf
        buf = torch.zeros(shape, device=self.device, dtype=dt)
        self._buffers[name] = buf
        return buf

    def get_memory_info(self) -> Dict[str, float]:
        """Query accelerator memory usage."""
        if not self.is_cuda:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
        return {
            "allocated_mb": round(torch.cuda.memory_allocated(self.device) / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(self.device) / (1024 * 1024), 2),
            "peak_mb": round(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 2),
        }
