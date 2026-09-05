"""Test environment dependencies and compute device detection."""

import sys
import pytest
import numpy as np
import torch
import cv2
import scipy

from ugv_core.device import get_device, get_device_info, configure_cpu_threading

def test_python_version():
    assert sys.version_info >= (3, 10), "Python 3.10+ is required."

def test_core_dependencies():
    assert torch.__version__ is not None
    assert cv2.__version__ is not None
    assert np.__version__ is not None
    assert scipy.__version__ is not None

def test_device_selection():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cpu", "cuda", "mps"]

    # Force CPU override test
    cpu_device = get_device(force_cpu=True)
    assert cpu_device.type == "cpu"

def test_device_info_audit():
    info = get_device_info()
    assert info.device_type in ["cpu", "cuda", "mps"]
    assert isinstance(info.device_name, str)
    assert isinstance(info.cuda_available, bool)

def test_cpu_thread_configuration():
    configured_threads = configure_cpu_threading(2)
    assert configured_threads == 2
    assert torch.get_num_threads() == 2
