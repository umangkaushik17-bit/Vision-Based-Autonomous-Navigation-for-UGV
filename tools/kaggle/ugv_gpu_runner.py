"""Kaggle GPU Execution Harness for SIH2026 Autonomous Navigation.

Runs headlessly on Kaggle Cloud GPUs (NVIDIA Tesla T4 / P100):
1. Verifies CUDA / GPU environment
2. Mounts/extracts the workspace codebase from dataset or git
3. Installs package dependencies / configures PYTHONPATH
4. Executes the targeted simulation, model training, or test benchmark
5. Copies all output metrics, logs, and plots to /kaggle/working/outputs/ for automatic download
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile

OUTPUTS_DIR = "/kaggle/working/outputs"
REPO_DIR = "/tmp/SIH2026"
INPUT_ROOT = "/kaggle/input"


def check_gpu():
    print("=" * 60)
    print("  KAGGLE CLOUD GPU HARDWARE PROFILING")
    print("=" * 60)
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
        else:
            print("WARNING: CUDA is NOT detected on this runtime instance.")
    except Exception as e:
        print(f"PyTorch probe note: {e}")
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except Exception:
        pass
    print("=" * 60 + "\n")


def setup_workspace():
    print("[1/3] Setting up workspace codebase...")
    os.makedirs(REPO_DIR, exist_ok=True)

    print(f"Scanning {INPUT_ROOT} for attached datasets...")
    found = False
    if os.path.exists(INPUT_ROOT):
        print(f"Attached inputs: {os.listdir(INPUT_ROOT)}")
        for root, dirs, files in os.walk(INPUT_ROOT):
            # Check if any zip exists
            for f in files:
                if f.endswith(".zip"):
                    zpath = os.path.join(root, f)
                    print(f"Found code archive: {zpath}. Extracting to {REPO_DIR} ...")
                    with zipfile.ZipFile(zpath, "r") as zf:
                        zf.extractall(REPO_DIR)
                    found = True
                    break
            # Check if extracted code folder exists directly
            if "ai" in dirs and any(k in files for k in ["setup.py", "pyproject.toml"]):
                print(f"Found project root in {root}. Copying to {REPO_DIR} ...")
                shutil.copytree(root, REPO_DIR, dirs_exist_ok=True)
                found = True
                break
            if found:
                break

    if not found:
        print("Warning: Could not find codebase in /kaggle/input. Contents:")
        for r, d, f in os.walk(INPUT_ROOT):
            print(f"  {r}: dirs={d}, files={f}")
        raise RuntimeError("Codebase not found in /kaggle/input!")

    os.chdir(REPO_DIR)
    print(f"Current working directory: {os.getcwd()}")
    print("Extracted files in workspace:", os.listdir("."))

    print("\n[2/3] Configuring environment and dependencies...")
    # Add REPO_DIR to PYTHONPATH
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{REPO_DIR}:{pythonpath}"
    sys.path.insert(0, REPO_DIR)

    # Install in editable mode
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=False)
    print("Environment successfully configured.\n")


DEFAULT_CMD = "python scripts/run_phase16_benchmark.py --batch all && python scripts/generate_phase16_plots.py"


def execute_tasks():
    print("[3/3] Executing designated benchmarks / tests...")
    test_cmd = os.environ.get("RUN_CMD", DEFAULT_CMD)
    print(f"Running command: {test_cmd}")
    ret = subprocess.run(test_cmd, shell=True)
    print(f"\nExecution completed with return code: {ret.returncode}\n")

    # Package outputs to /kaggle/working/outputs/
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    source_dirs = [
        "simulation/outputs",
        "validation/matlab/phase15/results",
        "validation/matlab/phase15/plots",
        "outputs",
    ]
    for src in source_dirs:
        if os.path.exists(src):
            dst = os.path.join(OUTPUTS_DIR, os.path.basename(src))
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"Exported directory: {src} -> {dst}")
            else:
                shutil.copy(src, dst)
                print(f"Exported file: {src} -> {dst}")

    print(f"\nAll artifacts staged in {OUTPUTS_DIR}. Ready for automatic download.")

    if ret.returncode != 0:
        print(f"[-] Command failed with return code {ret.returncode}")
        sys.exit(ret.returncode)


def main():
    check_gpu()
    setup_workspace()
    execute_tasks()


if __name__ == "__main__":
    main()
