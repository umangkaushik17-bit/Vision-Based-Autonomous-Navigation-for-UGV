"""Autonomous master entry point for external GPU execution (Google Colab, Kaggle, GPU VM)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import zipfile

def run_command(cmd: str, dry_run: bool = False) -> bool:
    """Execute shell command with stream logging and unified PYTHONPATH."""
    print(f"\n[AUTO_RUN EXEC] >>> {cmd}")
    if dry_run:
        print("[DRY-RUN] Command skipped.")
        return True

    env = os.environ.copy()
    root_dir = Path.cwd().resolve()
    src_dir = root_dir / "src"
    pypath_parts = [str(root_dir), str(src_dir)]
    if "PYTHONPATH" in env:
        pypath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pypath_parts)

    res = subprocess.run(cmd, shell=True, env=env)
    if res.returncode != 0:
        print(f"[!] Command failed with exit code {res.returncode}")
        return False
    return True

def stage_setup(dry_run: bool = False) -> bool:
    """Stage 1: Verify Python packages and GPU availability, and register local package."""
    print("\n" + "=" * 60)
    print("STAGE 1: HARDWARE & ENVIRONMENT AUDIT")
    print("=" * 60)
    cmd = (
        'python -c "import torch; '
        'print(\'CUDA Available :\', torch.cuda.is_available()); '
        'print(\'Device Name    :\', torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'CPU\'); '
        'print(\'PyTorch Version:\', torch.__version__)"'
    )
    ok = run_command(cmd, dry_run=dry_run)
    if not ok:
        return False

    # Ensure local package (ugv_core, ai) is registered in Python path
    if Path("pyproject.toml").exists():
        print("\n[STAGE 1] Installing local package (pyproject.toml) in editable mode...")
        run_command("pip install -e . --no-deps", dry_run=dry_run)

    return True

def stage_data(dry_run: bool = False) -> bool:
    """Stage 2: Check, extract, index public datasets (RUGD), and validate anti-leakage governance."""
    print("\n" + "=" * 60)
    print("STAGE 2: DATASET ACQUISITION, EXTRACTION & INTEGRITY VALIDATION")
    print("=" * 60)

    # 0. Check for split dataset archive (gpu_data.zip) and unpack if needed
    gpu_data_zip = Path("gpu_data.zip")
    if gpu_data_zip.exists() and not Path("data/processed/sequences").exists():
        print(f"[STAGE 2] Detected {gpu_data_zip.name}. Unpacking dataset sequences...")
        if not dry_run:
            with zipfile.ZipFile(gpu_data_zip, "r") as zf:
                zf.extractall(".")
            print("[STAGE 2] Dataset extracted successfully.")

    # 1. Automatically scan and extract any RUGD or other public archives
    ok_ext = run_command("python scripts/download_datasets.py --extract-only", dry_run=dry_run)
    if not ok_ext:
        return False

    # 2. Re-compile manifest to register newly extracted RUGD sequences
    ok_build = run_command("python scripts/build_dataset_manifest.py", dry_run=dry_run)
    if not ok_build:
        return False

    # 3. Validate dataset integrity and zero-leakage proof
    return run_command("python scripts/validate_dataset.py --manifest data/manifests/dataset_manifest.json", dry_run=dry_run)

def stage_smoke_test(dry_run: bool = False) -> bool:
    """Stage 3: Quick GPU smoke test verifying forward & backward passes."""
    print("\n" + "=" * 60)
    print("STAGE 3: FAST TRAINING SMOKE TEST")
    print("=" * 60)
    return run_command("python scripts/train_traversability.py --config configs/traversability_training.yaml --smoke-test", dry_run=dry_run)

def stage_train(dry_run: bool = False) -> bool:
    """Stage 4: Full deep-learning training run with class-weighted loss and AMP."""
    print("\n" + "=" * 60)
    print("STAGE 4: FULL GPU TRAVERSABILITY TRAINING")
    print("=" * 60)
    return run_command("python scripts/train_traversability.py --config configs/traversability_training.yaml", dry_run=dry_run)

def stage_evaluate(dry_run: bool = False) -> bool:
    """Stage 5: Comprehensive evaluation across synthetic, public, and own-real data."""
    print("\n" + "=" * 60)
    print("STAGE 5: MODEL EVALUATION & DOMAIN GAP ANALYSIS")
    print("=" * 60)
    return run_command("python scripts/evaluate_traversability.py --checkpoint outputs/checkpoints/best.pt", dry_run=dry_run)

def stage_benchmark(dry_run: bool = False) -> bool:
    """Stage 6: Benchmark inference latency, FPS, and model size on the GPU."""
    print("\n" + "=" * 60)
    print("STAGE 6: INFERENCE LATENCY & FPS BENCHMARK")
    print("=" * 60)
    return run_command("python scripts/benchmark_traversability.py", dry_run=dry_run)

def stage_export(dry_run: bool = False) -> bool:
    """Stage 7: Export model to TorchScript and verified ONNX."""
    print("\n" + "=" * 60)
    print("STAGE 7: MODEL EXPORT (TORCHSCRIPT & ONNX)")
    print("=" * 60)
    return run_command("python scripts/export_traversability.py --checkpoint outputs/checkpoints/best.pt", dry_run=dry_run)

def stage_sam_tp_eval(dry_run: bool = False) -> bool:
    """Evaluate pretrained SAM-TP foundation model across sequence splits."""
    print("\n" + "=" * 60)
    print("STAGE: PRETRAINED SAM-TP BENCHMARK & FALSE-SAFE EVALUATION")
    print("=" * 60)
    return run_command("python scripts/benchmark_sam_tp.py --max-samples 20", dry_run=dry_run)

def stage_sam_tp_finetune(dry_run: bool = False) -> bool:
    """Fine-tune SAM-TP model on multi-domain outdoor dataset (50 epochs with early stopping)."""
    print("\n" + "=" * 60)
    print("STAGE: SAM-TP GPU FINE-TUNING")
    print("=" * 60)
    return run_command("python scripts/train_traversability.py --config configs/sam_tp_finetuning.yaml", dry_run=dry_run)

def stage_package(dry_run: bool = False) -> bool:
    """Stage 8: Bundle all outputs (checkpoints, exports, reports) into gpu_results.zip."""
    print("\n" + "=" * 60)
    print("STAGE 8: PACKAGING RESULTS FOR DOWNLOAD")
    print("=" * 60)
    if dry_run:
        print("[DRY-RUN] Packaging skipped.")
        return True

    outputs_dir = Path("outputs").resolve()
    zip_target = Path("gpu_results.zip").resolve()

    if not outputs_dir.exists():
        print("[!] No outputs directory found to package.")
        return False

    with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in outputs_dir.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(outputs_dir.parent))

    print(f"\n[OK] Successfully packaged results into: {zip_target} ({zip_target.stat().st_size / (1024*1024):.2f} MB)")
    print("Download 'gpu_results.zip' and import it into your local Antigravity environment.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Autonomous GPU Execution Orchestrator")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["setup", "data", "smoke-test", "train", "evaluate", "benchmark", "export", "sam-tp-eval", "sam-tp-finetune", "package", "all"],
        help="Target execution stage"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without execution")
    args = parser.parse_args()

    # Move working directory to root if executed from within gpu_package/
    if Path.cwd().name == "gpu_package":
        os.chdir(Path.cwd().parent)

    stages = {
        "setup": stage_setup,
        "data": stage_data,
        "smoke-test": stage_smoke_test,
        "train": stage_train,
        "evaluate": stage_evaluate,
        "benchmark": stage_benchmark,
        "export": stage_export,
        "sam-tp-eval": stage_sam_tp_eval,
        "sam-tp-finetune": stage_sam_tp_finetune,
        "package": stage_package
    }

    if args.stage == "all":
        execution_order = ["setup", "data", "smoke-test", "train", "evaluate", "benchmark", "export", "package"]
        for s in execution_order:
            ok = stages[s](dry_run=args.dry_run)
            if not ok:
                print(f"\n[CRITICAL FAILURE] Pipeline halted at stage: {s}")
                sys.exit(1)
        print("\n" + "=" * 60)
        print(">>> ALL GPU TRAINING & EVALUATION STAGES COMPLETED SUCCESSFULLY <<<")
        print("=" * 60)
    else:
        ok = stages[args.stage](dry_run=args.dry_run)
        if not ok:
            sys.exit(1)

if __name__ == "__main__":
    main()
