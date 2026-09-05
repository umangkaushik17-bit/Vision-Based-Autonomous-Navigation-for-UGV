"""Device abstraction and hardware management for UGV robotics stack.

Handles dynamic detection and selection of compute backends (CPU, NVIDIA CUDA, Apple MPS)
with performance tuning for both cloud A100 training and local edge inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass(frozen=True)
class DeviceInfo:
    """Detailed information regarding the selected compute backend."""
    device_type: str
    device_name: str
    cuda_available: bool
    device_count: int
    memory_total_gb: Optional[float] = None
    is_a100: bool = False

def get_device(prefer_cuda: bool = True, force_cpu: bool = False) -> torch.device:
    """Return the optimal PyTorch compute device.
    
    Args:
        prefer_cuda: If True, uses CUDA whenever available.
        force_cpu: If True, overrides CUDA and forces CPU execution.
        
    Returns:
        torch.device instance (e.g., 'cuda:0' or 'cpu').
    """
    if force_cpu:
        return torch.device("cpu")
        
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
        
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
        
    return torch.device("cpu")

def get_device_info() -> DeviceInfo:
    """Audit the host system's hardware acceleration status."""
    cuda_avail = torch.cuda.is_available()
    dev_count = torch.cuda.device_count() if cuda_avail else 0
    
    if cuda_avail:
        dev_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        is_a100 = "A100" in dev_name.upper()
        dev_type = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev_name = "Apple Silicon MPS"
        total_mem = None
        is_a100 = False
        dev_type = "mps"
    else:
        dev_name = "Host CPU"
        total_mem = None
        is_a100 = False
        dev_type = "cpu"
        
    return DeviceInfo(
        device_type=dev_type,
        device_name=dev_name,
        cuda_available=cuda_avail,
        device_count=dev_count,
        memory_total_gb=round(total_mem, 2) if total_mem else None,
        is_a100=is_a100
    )

def configure_cpu_threading(num_threads: Optional[int] = None) -> int:
    """Configure CPU thread pool for optimal multi-threaded inference latency.
    
    Returns:
        The configured number of threads.
    """
    if num_threads is None:
        cpu_count = os.cpu_count() or 4
        # Allocate roughly 75% of available logical cores to avoid starving OS / ROS2 threads
        num_threads = max(1, int(cpu_count * 0.75))
        
    torch.set_num_threads(num_threads)
    return num_threads
