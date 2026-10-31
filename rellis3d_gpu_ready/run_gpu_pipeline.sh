#!/usr/bin/env bash
# Shell entrypoint for RELLIS-3D GPU Traversability Pipeline
set -e

echo "=================================================================="
echo "    RELLIS-3D GPU TRAVERSABILITY PIPELINE INITIALIZATION          "
echo "=================================================================="

# Check Python and activate environment if needed
PYTHON_BIN=${PYTHON_BIN:-python3}

# 1. Hardware Verification
$PYTHON_BIN scripts/verify_gpu.py || {
    echo "[ABORT] GPU verification failed. CPU training is forbidden."
    exit 1
}

# 2. Dataset Discovery
$PYTHON_BIN scripts/discover_dataset.py || {
    echo "[ABORT] RELLIS-3D dataset not found. Please mount or place dataset archive."
    exit 1
}

# 3. Label Inspection
$PYTHON_BIN scripts/inspect_dataset.py
$PYTHON_BIN scripts/validate_labels.py

# 4. GPU-Only Training
$PYTHON_BIN scripts/train_gpu.py --config configs/training.yaml

# 5. Cross-Domain Evaluation
$PYTHON_BIN scripts/evaluate.py

# 6. Report Packaging
$PYTHON_BIN scripts/package_report.py

echo "=================================================================="
echo "    PIPELINE EXECUTION CONCLUDED SUCCESSFULLY                    "
echo "=================================================================="
