#!/usr/bin/env python3
"""Hardware and CUDA Runtime Verification Script for RELLIS-3D GPU Pipeline.

Exits with code 0 if a usable CUDA GPU is present and tensor operations pass.
Exits with code 1 if no CUDA GPU is available or initialization fails.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

def main() -> int:
    print("=" * 65)
    print("          GPU HARDWARE AND CUDA RUNTIME VERIFICATION            ")
    print("=" * 65)

    try:
        import torch
    except ImportError:
        print("[CRITICAL] PyTorch is not installed in the active environment.")
        return 1

    print(f"PyTorch Version  : {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available   : {cuda_available}")

    if not cuda_available:
        print("\n[ERROR] CUDA is NOT available in this environment.")
        print("Reason: PyTorch cannot detect an NVIDIA GPU with CUDA drivers.")
        print("Per strict engineering restrictions: CPU fallback training is forbidden.")
        print("Halting pipeline safely with exit code 1.")
        return 1

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    vram_gb = device_props.total_memory / (1024 ** 3)
    compute_cap = f"{device_props.major}.{device_props.minor}"

    print(f"CUDA Devices     : {device_count}")
    print(f"Active Device    : {device_name}")
    print(f"Compute Capability: sm_{compute_cap}")
    print(f"Total VRAM       : {vram_gb:.2f} GB")
    print(f"CUDA Version     : {torch.version.cuda}")

    # Run small tensor verification test on GPU
    print("\nRunning GPU Tensor Verification Test...")
    try:
        t0 = time.perf_counter()
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        print(f"[OK] GPU Tensor Multiplication verified in {dt:.2f} ms.")
    except Exception as e:
        print(f"[CRITICAL] GPU tensor test failed: {e}")
        return 1

    # Check mixed precision capability
    amp_supported = device_props.major >= 7
    print(f"Mixed Precision (AMP): {'SUPPORTED (Compute >= 7.0)' if amp_supported else 'NOT OPTIMAL'}")

    report = {
        "verified": True,
        "device_name": device_name,
        "device_count": device_count,
        "compute_capability": compute_cap,
        "vram_gb": round(vram_gb, 2),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "amp_supported": amp_supported
    }

    out_path = Path("reports/gpu_environment_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SUCCESS] GPU environment verified. Report saved to {out_path}.")
    print("=" * 65)
    return 0

if __name__ == "__main__":
    sys.exit(main())
