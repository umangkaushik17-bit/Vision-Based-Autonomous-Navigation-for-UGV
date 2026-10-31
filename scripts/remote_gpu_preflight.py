"""Phase 13.17.0 Remote/Local GPU Preflight Diagnostic.

Queries environment, detects GPU hardware, checks CUDA capability,
performs real CUDA computation & synchronization, tests dynamic_gpu kernels,
and outputs t4_environment.json if running on NVIDIA Tesla T4.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys


def get_git_info() -> tuple[str, str]:
    """Retrieve git branch and commit hash."""
    branch = "unknown"
    commit = "unknown"
    try:
        res_b = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=False)
        if res_b.returncode == 0:
            branch = res_b.stdout.strip()
        res_c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if res_c.returncode == 0:
            commit = res_c.stdout.strip()
    except Exception:
        pass
    return branch, commit


def get_driver_version() -> str:
    """Retrieve NVIDIA driver version via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().split("\n")[0].strip()
    except Exception:
        pass
    return "N/A"


def run_preflight() -> dict:
    git_branch, git_commit = get_git_info()
    os_info = platform.platform()
    py_version = platform.python_version()

    print("=" * 60)
    print("=== KAGGLE PREFLIGHT / ENVIRONMENT PROBE ===")
    print("=" * 60)

    # 1. PyTorch & CUDA check
    torch_available = False
    cuda_available = False
    torch_version = "N/A"
    torch_cuda_version = "N/A"
    gpu_name = "None"
    gpu_count = 0
    compute_capability = "N/A"
    vram_gb = 0.0
    driver_version = get_driver_version()

    try:
        import torch
        torch_available = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        torch_cuda_version = torch.version.cuda or "N/A"

        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            compute_capability = f"{cap[0]}.{cap[1]}"
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024 ** 3), 2)
    except Exception as e:
        print(f"PyTorch inspection note: {e}")

    print(f"OS/kernel: {os_info}")
    print(f"Python version: {py_version}")
    print(f"PyTorch version: {torch_version}")
    print(f"CUDA available: {cuda_available}")
    print(f"PyTorch CUDA version: {torch_cuda_version}")
    print(f"Driver version: {driver_version}")
    print(f"GPU name: {gpu_name}")
    print(f"GPU count: {gpu_count}")
    print(f"Compute capability: {compute_capability}")
    print(f"VRAM: {vram_gb} GB")

    # 2. Real CUDA computation test
    real_cuda_success = False
    real_cuda_error = ""

    if cuda_available:
        try:
            import torch
            print("\nExecuting real CUDA tensor computation & synchronization...")
            t0 = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
            t1 = torch.matmul(t0, t0.t())
            torch.cuda.synchronize()
            s = t1.sum().item()
            real_cuda_success = True
            print(f"[+] Real CUDA computation succeeded! (Tensor sum: {s:.4f})")
        except Exception as e:
            real_cuda_error = str(e)
            print(f"[-] Real CUDA computation FAILED: {e}")
    else:
        real_cuda_error = "CUDA not available on this device"
        print(f"[-] Real CUDA computation SKIPPED: {real_cuda_error}")

    # 3. Dynamic GPU CUDA smoke test
    dynamic_gpu_success = False
    dynamic_gpu_error = ""

    if real_cuda_success:
        try:
            print("\nExecuting ai/planning/dynamic_gpu CUDA smoke test...")
            from ai.planning.base import RobotState
            from ai.planning.dynamic_gpu.gpu_backend import GPUBackend
            from ai.planning.dynamic_gpu.gpu_rollout import GPURollout

            backend = GPUBackend(preferred_device="cuda")
            rollout = GPURollout(backend)
            v_grid, w_grid = rollout.generate_candidate_grid(0.0, 0.6, -0.8, 0.8, 25, 15)
            state = RobotState(x=0.0, y=0.0, yaw=0.0, timestamp=0.0)
            trajectories, times = rollout.rollout_trajectories(state, v_grid, w_grid, pred_horizon=3.0, sim_dt=0.2)
            torch.cuda.synchronize()

            assert trajectories.shape == (375, 15, 4), f"Unexpected shape {trajectories.shape}"
            assert torch.isfinite(trajectories).all(), "Non-finite values in trajectories"
            dynamic_gpu_success = True
            print(f"[+] dynamic_gpu CUDA smoke test succeeded! Rollout shape: {trajectories.shape}")
        except Exception as e:
            dynamic_gpu_error = str(e)
            print(f"[-] dynamic_gpu CUDA smoke test FAILED: {e}")
    else:
        dynamic_gpu_error = f"Prerequisite real CUDA test not passed ({real_cuda_error})"

    # 4. Determine Milestone 13.17.0 Status
    is_t4 = ("t4" in gpu_name.lower()) and (compute_capability == "7.5")
    status = "BLOCKED"
    block_reason = ""

    if not cuda_available:
        status = "BLOCKED — NO CUDA"
        block_reason = "CUDA is not available on execution environment."
    elif not real_cuda_success:
        status = "BLOCKED — CUDA COMPUTATION ERROR"
        block_reason = f"CUDA computation failed: {real_cuda_error}"
    elif not is_t4:
        status = "BLOCKED — WRONG GPU"
        block_reason = f"Target GPU is NVIDIA Tesla T4 (sm_75), but found '{gpu_name}' (sm_{compute_capability.replace('.', '')})."
    elif not dynamic_gpu_success:
        status = "BLOCKED — DYNAMIC_GPU ERROR"
        block_reason = f"dynamic_gpu execution failed on T4: {dynamic_gpu_error}"
    else:
        status = "READY FOR PHASE 13.17.1"

    print("\n" + "=" * 60)
    print("------------------------------------------")
    print("PHASE 13.17.0 PREFLIGHT RESULT")
    print("------------------------------------------")
    print(f"Repository: umangkaushik17-bit/Vision-Based-Autonomous-Navigation-for-UGV")
    print(f"Branch: {git_branch}")
    print(f"Commit: {git_commit}")
    print(f"GPU: {gpu_name}")
    print(f"Compute capability: {compute_capability}")
    print(f"VRAM: {vram_gb} GB")
    print(f"CUDA: {cuda_available} (Driver: {driver_version})")
    print(f"PyTorch: {torch_version}")
    print(f"PyTorch CUDA: {torch_cuda_version}")
    print(f"Real CUDA computation: {'PASS' if real_cuda_success else f'FAIL ({real_cuda_error})'}")
    print(f"Dynamic GPU CUDA smoke test: {'PASS' if dynamic_gpu_success else f'FAIL ({dynamic_gpu_error})'}")
    print(f"STATUS: {status}")
    if block_reason:
        print(f"Reason: {block_reason}")
    print("------------------------------------------")
    print("=" * 60 + "\n")

    # 5. Save Artifact if T4 or diagnostic artifact
    env_data = {
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "compute_capability": compute_capability,
        "vram_gb": vram_gb,
        "cuda_available": cuda_available,
        "cuda_version": torch_cuda_version,
        "driver_version": driver_version,
        "pytorch_version": torch_version,
        "pytorch_cuda_version": torch_cuda_version,
        "python_version": py_version,
        "platform": os_info,
        "real_cuda_test": real_cuda_success,
        "dynamic_gpu_test": dynamic_gpu_success,
        "status": status,
        "block_reason": block_reason,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_branch": git_branch,
    }

    if is_t4 and real_cuda_success:
        out_dir = os.path.join("simulation", "outputs", "performance", "cuda")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "t4_environment.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(env_data, f, indent=2)
        print(f"[+] Saved T4 environment artifact to: {out_path}")
    else:
        # Diagnostic save
        out_dir = os.path.join("simulation", "outputs", "performance", "cuda")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "diagnostic_environment.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(env_data, f, indent=2)
        print(f"[*] Saved diagnostic environment artifact to: {out_path}")

    return env_data


if __name__ == "__main__":
    run_preflight()
