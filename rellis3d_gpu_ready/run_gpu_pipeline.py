#!/usr/bin/env python3
"""Master Orchestrator for RELLIS-3D GPU Traversability Pipeline.

Enforces execution order:
  1. GPU Environment Verification
  2. Dataset Discovery & Safe Extraction
  3. Label Audit & Class Mapping Verification
  4. GPU-Only Training / Fine-Tuning
  5. Cross-Domain Safety Evaluation
  6. Final Report Packaging

Stops immediately with clear status if GPU or dataset is missing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

def run_step(cmd: list[str], step_name: str) -> bool:
    print(f"\n>>> [STEP] {step_name}")
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n[FAILED] {step_name} exited with error code {res.returncode}.")
        return False
    print(f"[PASSED] {step_name} completed successfully.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run RELLIS-3D GPU Pipeline")
    parser.add_argument("--smoke-test", action="store_true", help="Run bounded 2-epoch smoke test")
    parser.add_argument("--skip-extract", action="store_true", help="Skip archive extraction step")
    args = parser.parse_args()

    py = sys.executable

    # 1. GPU Verification
    if not run_step([py, "scripts/verify_gpu.py"], "1. GPU Hardware Verification"):
        print("\n[STOP] Pipeline halted: GPU unavailable. CPU training is forbidden.")
        sys.exit(1)

    # 2. Dataset Discovery
    if not run_step([py, "scripts/discover_dataset.py"], "2. Dataset Discovery"):
        print("\n[STOP] Pipeline halted: RELLIS-3D dataset could not be located.")
        sys.exit(1)

    # 3. Label Validation & Inventory
    run_step([py, "scripts/inspect_dataset.py"], "3. Dataset Inventory Inspection")
    run_step([py, "scripts/validate_labels.py"], "4. Label Class ID Audit")

    # 4. GPU Training
    train_cmd = [py, "scripts/train_gpu.py"]
    if args.smoke_test:
        train_cmd.append("--smoke-test")
    if not run_step(train_cmd, "5. GPU Training"):
        print("\n[STOP] Pipeline halted during training.")
        sys.exit(1)

    # 5. Cross-Domain Safety Evaluation
    eval_cmd = [py, "scripts/evaluate.py"]
    if args.smoke_test:
        eval_cmd.append("--smoke-test")
    run_step(eval_cmd, "6. Safety & Cross-Domain Evaluation")

    # 6. Final Report
    run_step([py, "scripts/package_report.py"], "7. Report Packaging")

    print("\n" + "=" * 65)
    print("      RELLIS-3D GPU PIPELINE EXECUTION FINISHED SUCCESSFULLY     ")
    print("=" * 65)

if __name__ == "__main__":
    main()
