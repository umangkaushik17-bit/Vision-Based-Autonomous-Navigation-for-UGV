#!/usr/bin/env python3
"""Strict Kaggle Environment and NVIDIA Tesla T4 Hardware Verification.

Performs exhaustive verification of:
1. Kaggle-specific environment indicators
2. Host platform & Python runtime
3. PyTorch CUDA runtime & driver details
4. Physical GPU identification (Target: NVIDIA Tesla T4)
5. Multi-tensor CUDA computation, synchronization, and VRAM allocation on cuda:0
6. Available host memory (RAM) and filesystem disk space
7. Internet access availability
8. Generates authoritative reports/kaggle_t4_environment.json
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def check_driver_version() -> str:
    """Retrieve NVIDIA driver version via nvidia-smi if available."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=5
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().split("\n")[0].strip()
    except Exception:
        pass
    return "N/A"


def check_internet() -> bool:
    """Verify internet connectivity by testing connection to a reliable endpoint."""
    try:
        import urllib.request
        with urllib.request.urlopen("https://www.google.com", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def get_disk_space_gb(path: str) -> float:
    """Get free disk space in GB for given path."""
    try:
        usage = shutil.disk_usage(path)
        return round(usage.free / (1024 ** 3), 2)
    except Exception:
        return 0.0


def get_available_ram_gb() -> float:
    """Get available system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().available / (1024 ** 3), 2)
    except ImportError:
        try:
            # Fallback on Linux /proc/meminfo
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemAvailable" in line:
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 2)
        except Exception:
            pass
    return 0.0


def verify_environment(output_json_path: str = "reports/kaggle_t4_environment.json") -> dict:
    print("=" * 70)
    print("        STRICT KAGGLE ENVIRONMENT & NVIDIA T4 VERIFICATION          ")
    print("=" * 70)

    # 1. Kaggle Environment Indicators
    is_kaggle = (
        os.path.exists("/kaggle")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_DATA_PROXY_TOKEN" in os.environ
        or os.path.exists("/kaggle/working")
    )
    kaggle_input_dir = "/kaggle/input" if os.path.exists("/kaggle/input") else "N/A"
    kaggle_working_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "N/A"

    cwd = os.getcwd()
    os_info = platform.platform()
    py_version = platform.python_version()
    driver_version = check_driver_version()
    internet_ok = check_internet()
    avail_ram = get_available_ram_gb()
    disk_free = get_disk_space_gb(cwd)

    print(f"Kaggle Environment Detected : {'YES' if is_kaggle else 'NO'}")
    print(f"Python Version              : {py_version}")
    print(f"Operating System            : {os_info}")
    print(f"Current Working Directory   : {cwd}")
    print(f"Kaggle Input Directory      : {kaggle_input_dir}")
    print(f"Kaggle Working Directory    : {kaggle_working_dir}")
    print(f"Internet Availability       : {'YES' if internet_ok else 'NO'}")
    print(f"Available System RAM        : {avail_ram} GB")
    print(f"Available Disk Space        : {disk_free} GB")

    # 2. PyTorch & CUDA Inspection
    torch_available = False
    cuda_available = False
    torch_version = "N/A"
    cuda_version = "N/A"
    gpu_name = "None"
    gpu_count = 0
    gpu_index = None
    vram_gb = 0.0
    compute_capability = "N/A"

    try:
        import torch
        torch_available = True
        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = str(torch.version.cuda) if torch.version.cuda else "N/A"

        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_index = 0
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024 ** 3), 2)
            compute_capability = f"{props.major}.{props.minor}"
    except ImportError:
        print("[CRITICAL] PyTorch is not installed.")

    print(f"PyTorch Version             : {torch_version}")
    print(f"CUDA Available              : {cuda_available}")
    print(f"CUDA Runtime Version        : {cuda_version}")
    print(f"NVIDIA Driver Version       : {driver_version}")
    print(f"GPU Name                    : {gpu_name}")
    print(f"GPU Count                   : {gpu_count}")
    print(f"GPU Index                   : {gpu_index}")
    print(f"GPU VRAM                    : {vram_gb} GB")
    print(f"Compute Capability          : {compute_capability}")

    # 3. Real CUDA Computation, Synchronization & Memory Allocation Test
    real_cuda_success = False
    real_cuda_error = ""
    allocated_mb = 0.0

    if cuda_available:
        print("\n--- Executing Real CUDA Computation & VRAM Allocation ---")
        try:
            import torch
            torch.cuda.empty_cache()
            initial_mem = torch.cuda.memory_allocated(0)

            # Matrix multiplication test
            t0 = time.perf_counter()
            a = torch.randn(2048, 2048, device="cuda:0", dtype=torch.float32)
            b = torch.randn(2048, 2048, device="cuda:0", dtype=torch.float32)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000

            allocated_mb = round((torch.cuda.memory_allocated(0) - initial_mem) / (1024 ** 2), 2)
            tensor_sum = float(c.sum().item())
            assert torch.isfinite(c).all(), "NaN or Inf detected in CUDA computation"

            real_cuda_success = True
            print(f"[OK] CUDA 2048x2048 Matmul executed on cuda:0 in {dt:.2f} ms")
            print(f"[OK] Tensor Sum: {tensor_sum:.4f} (All finite: True)")
            print(f"[OK] VRAM Allocated for test: {allocated_mb} MB")
            torch.cuda.empty_cache()
        except Exception as e:
            real_cuda_error = str(e)
            print(f"[ERROR] CUDA computation failed: {e}")
    else:
        real_cuda_error = "CUDA unavailable on host"

    # 4. Strict T4 Validation Decision
    is_t4 = "t4" in gpu_name.lower() or "tesla t4" in gpu_name.lower()
    t4_ready = is_kaggle and cuda_available and is_t4 and real_cuda_success

    if not is_kaggle:
        decision_status = "LOCAL_ENVIRONMENT_DETECTED"
        decision_msg = "Current execution environment is local host, not Kaggle Cloud. Remote Kaggle dispatch required."
    elif not cuda_available:
        decision_status = "BLOCKED_NO_CUDA"
        decision_msg = "CUDA is not available on Kaggle instance. GPU accelerator must be enabled."
    elif not is_t4:
        decision_status = "BLOCKED_WRONG_GPU"
        decision_msg = f"Detected GPU is '{gpu_name}', expected 'NVIDIA Tesla T4'."
    elif not real_cuda_success:
        decision_status = "BLOCKED_CUDA_COMPUTE_ERROR"
        decision_msg = f"Real CUDA computation failed: {real_cuda_error}"
    else:
        decision_status = "VERIFIED_KAGGLE_T4_READY"
        decision_msg = "Kaggle NVIDIA Tesla T4 environment verified and fully operational."

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "kaggle_environment_detected": is_kaggle,
        "python_version": py_version,
        "os": os_info,
        "pytorch_version": torch_version,
        "cuda_version": cuda_version,
        "nvidia_driver_information": driver_version,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "gpu_index": gpu_index,
        "gpu_vram_gb": vram_gb,
        "cuda_available": cuda_available,
        "compute_capability": compute_capability,
        "available_ram_gb": avail_ram,
        "available_disk_space_gb": disk_free,
        "current_working_directory": cwd,
        "kaggle_input_directory": kaggle_input_dir,
        "kaggle_working_directory": kaggle_working_dir,
        "internet_availability": internet_ok,
        "real_cuda_computation_test": {
            "success": real_cuda_success,
            "error": real_cuda_error,
            "allocated_mb": allocated_mb
        },
        "target_hardware_verified": is_t4,
        "decision_status": decision_status,
        "decision_message": decision_msg,
        "t4_ready_for_training": t4_ready
    }

    out_file = Path(output_json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved environment report: {out_file.resolve()}")
    print("=" * 70)
    print(f"VERIFICATION STATUS : {decision_status}")
    print(f"DECISION MESSAGE    : {decision_msg}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    rep = verify_environment()
    if not rep["t4_ready_for_training"] and rep["kaggle_environment_detected"]:
        sys.exit(1)
    sys.exit(0)
