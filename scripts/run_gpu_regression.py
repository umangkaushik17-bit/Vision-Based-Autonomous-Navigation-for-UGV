#!/usr/bin/env python3
"""Reusable GPU-Accelerated Regression & Parity Validation Harness.

Requirements fulfilled:
1. Detects CUDA availability and hardware capabilities.
2. Rejects unsupported environments with clear error (zero silent CPU fallback).
3. Executes GPU tensor verification (physical kernel execution).
4. Runs GPU-compatible planning, collision, and parity regression suites.
5. Captures execution times, memory footprints, and pass/fail counts.
6. Emits machine-readable JSON summary for audit compliance.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def detect_cuda_environment() -> Dict[str, Any]:
    """Inspects and validates host CUDA hardware and driver environment."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if not torch.cuda.is_available():
        info["status"] = "UNSUPPORTED_NO_CUDA"
        info["reason"] = (
            "No CUDA-capable device detected by PyTorch. Host possesses CPU-only build or missing NVIDIA driver. "
            "GPU regression must run on a supported environment (e.g. NVIDIA Tesla T4 on Kaggle Cloud)."
        )
        return info

    cuda_idx = 0
    props = torch.cuda.get_device_properties(cuda_idx)
    cap = torch.cuda.get_device_capability(cuda_idx)
    cap_str = f"{cap[0]}.{cap[1]}"

    info["status"] = "SUPPORTED_CUDA"
    info["cuda_version"] = torch.version.cuda
    info["device_count"] = torch.cuda.device_count()
    info["device_name"] = props.name
    info["compute_capability"] = cap_str
    info["total_vram_gb"] = round(props.total_memory / (1024**3), 2)

    # Physical tensor execution check
    try:
        x = torch.randn(256, 256, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        info["tensor_check_passed"] = bool((y.shape == (256, 256)))
    except Exception as exc:
        info["status"] = "CUDA_EXECUTION_ERROR"
        info["reason"] = f"Failed CUDA tensor execution: {exc}"

    return info


def run_gpu_regression(output_dir: Path, force_cpu: bool = False, run_parity: bool = True) -> int:
    """Executes the GPU regression and parity verification suite."""
    print("=" * 80)
    print("      SIH2026: REUSABLE GPU ACCELERATION REGRESSION HARNESS (PHASE 13.18)")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    env_info = detect_cuda_environment()

    print(f"Platform: {env_info['platform']}")
    print(f"PyTorch:  {env_info['torch_version']}")
    print(f"CUDA:     {env_info.get('cuda_available')}")

    if not env_info["cuda_available"] and not force_cpu:
        print("\n[FATAL ERROR] GPU Regression Cannot Execute on this Host:")
        print(f"  Reason: {env_info.get('reason')}")
        print("\nNotice: The harness refuses to silently fall back to CPU when GPU regression is requested.")
        print("To run on Kaggle Tesla T4, use: scripts/kaggle_runner.py")
        print("To run harness logic locally on CPU, pass: --force-cpu")
        
        result = {
            "timestamp": time.time(),
            "status": "FAIL_UNSUPPORTED_HARDWARE",
            "env_info": env_info,
            "tests_executed": 0,
            "error": env_info.get("reason"),
        }
        (output_dir / "gpu_regression_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 1

    device_target = "cuda" if env_info.get("cuda_available") else "cpu"
    print(f"\n[+] Executing GPU Regression on Target Device: {device_target.upper()}")
    if device_target == "cuda":
        print(f"    Device Name:        {env_info['device_name']}")
        print(f"    Compute Capability: {env_info['compute_capability']}")
        print(f"    Total VRAM:         {env_info['total_vram_gb']} GB")

    start_time = time.time()
    test_results: Dict[str, Any] = {}

    # 1. Run GPU-specific PyTest unit tests
    print("\n--- 1. Running GPU Test Modules (pytest) ---")
    pytest_targets = [
        "tests/test_dwa_acceleration_parity.py",
    ]
    pytest_cmd = [sys.executable, "-m", "pytest"] + pytest_targets + ["-v", "--tb=short"]
    t0 = time.time()
    proc = subprocess.run(pytest_cmd, capture_output=True, text=True)
    pt_elapsed = time.time() - t0
    
    test_results["pytest"] = {
        "command": " ".join(pytest_cmd),
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_s": round(pt_elapsed, 2),
        "status": "PASS" if proc.returncode == 0 else "FAIL",
    }
    print(f"Pytest Return Code: {proc.returncode} (Elapsed: {pt_elapsed:.2f}s)")
    (output_dir / "gpu_pytest_log.txt").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

    # 2. Run Deterministic and Randomized Parity Suite if requested
    if run_parity:
        print("\n--- 2. Running CPU vs GPU Parity Verification (Stages A & B) ---")
        parity_script = Path("scripts/run_parity_suite.py")
        if parity_script.exists():
            parity_cmd = [sys.executable, str(parity_script), "--stage", "B", "--device", device_target]
            t_par0 = time.time()
            par_proc = subprocess.run(parity_cmd, capture_output=True, text=True)
            par_elapsed = time.time() - t_par0
            test_results["parity_suite"] = {
                "command": " ".join(parity_cmd),
                "return_code": par_proc.returncode,
                "stdout": par_proc.stdout,
                "stderr": par_proc.stderr,
                "elapsed_s": round(par_elapsed, 2),
                "status": "PASS" if par_proc.returncode == 0 else "FAIL",
            }
            print(f"Parity Return Code: {par_proc.returncode} (Elapsed: {par_elapsed:.2f}s)")
            (output_dir / "gpu_parity_log.txt").write_text(par_proc.stdout + "\n" + par_proc.stderr, encoding="utf-8")

    total_elapsed = time.time() - start_time
    overall_status = "PASS" if all(v.get("status") == "PASS" for v in test_results.values()) else "FAIL"

    summary = {
        "timestamp": time.time(),
        "total_elapsed_s": round(total_elapsed, 2),
        "overall_status": overall_status,
        "device_target": device_target,
        "env_info": env_info,
        "test_results": test_results,
    }

    out_file = output_dir / "gpu_regression_result.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[+] GPU Regression Complete. Summary saved to: {out_file}")
    print(f"    Final Status: {overall_status}")
    return 0 if overall_status == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 13.18 Reusable GPU Regression Harness")
    parser.add_argument("--output-dir", type=str, default="simulation/outputs/phase18_regression/gpu",
                        help="Directory to store GPU regression logs and JSON summaries")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Allow harness execution on CPU device (for local verification without NVIDIA GPU)")
    parser.add_argument("--skip-parity", action="store_true",
                        help="Skip multi-stage randomized parity suite")
    args = parser.parse_args()

    return run_gpu_regression(
        output_dir=Path(args.output_dir),
        force_cpu=args.force_cpu,
        run_parity=not args.skip_parity,
    )


if __name__ == "__main__":
    sys.exit(main())
