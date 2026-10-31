# RELLIS-3D GPU Traversability Engineering Package
**Project:** SIH26126 — Vision Based Autonomous Navigation for UGV in Outdoor Environment  
**Package Version:** 1.0.0 (Pre-Phase 14 GPU Pipeline)  

---

## 1. Package Purpose

This self-contained package enables automated discovery, inspection, training, and evaluation of the **RELLIS-3D** outdoor semantic dataset for UGV dense traversability estimation. It builds upon the verified **RUGD baseline model** (`MobileNetV3-Small` encoder-decoder) to enable domain adaptation, safety-weighted loss optimization, and direct integration with downstream Bird's Eye View (BEV) costmaps.

---

## 2. Directory Structure

```text
rellis3d_gpu_ready/
├── README.md                          # Comprehensive technical documentation
├── QUICKSTART_GPU.md                  # Quick-start instructions for GPU execution
├── requirements-gpu.txt               # Strict PyTorch CUDA dependencies
├── environment.yml                    # Conda environment definition
├── run_gpu_pipeline.py                # Python master workflow runner
├── run_gpu_pipeline.sh                # Shell runner script
├── configs/
│   ├── dataset.yaml                   # Discovery roots, target resolutions, split ratios
│   ├── training.yaml                  # Hyperparameters, AMP, early stopping, loss weights
│   ├── inference.yaml                 # Inference settings and costmap thresholds
│   └── rellis_class_mapping.yaml      # Authoritative 20-class to 3-class safety mapping
├── scripts/
│   ├── verify_gpu.py                  # Strict CUDA hardware checker (exits 1 if CPU)
│   ├── discover_dataset.py            # Automated filesystem crawler for RELLIS data
│   ├── extract_dataset.py             # Safe zip/tar.gz extraction avoiding traversal
│   ├── inspect_dataset.py             # Image/mask pair validator and spatial checker
│   ├── validate_labels.py             # Class ID distribution and pixel frequency auditor
│   ├── train_gpu.py                   # Strict GPU-only training & fine-tuning pipeline
│   ├── evaluate.py                    # Safety metrics calculator (mIoU, unsafe recall)
│   ├── infer_gpu.py                   # Forward pass with BEV costmap generation
│   └── package_report.py              # Report aggregator
├── src/
│   ├── models/traversability_net.py   # Lightweight MobileNetV3-Small architecture
│   ├── datasets/rellis_dataset.py     # RELLIS-3D PyTorch dataset with outdoor augmentations
│   ├── losses/traversability_loss.py  # Focal Loss + Dice Loss with ignore_index=255
│   └── metrics/safety_metrics.py      # Unsafe recall and false-safe pixel ratio calculations
├── checkpoints/
│   ├── rugd_baseline_best.pt          # Authoritative pre-trained RUGD baseline (4.5 MB)
│   └── traversability_model.onnx      # ONNX export for edge deployment
└── reports/
    ├── gpu_environment_report.json    # Host hardware and CUDA audit
    ├── dataset_discovery.json         # Filesystem discovery log
    ├── baseline_report.md             # RUGD baseline architecture & evaluation audit
    ├── costmap_validation_report.md   # DWA costmap interface verification
    └── final_engineering_decision.md  # Engineering audit decision
```

---

## 3. Where to Place the Dataset

Place the uncompressed RELLIS-3D directory or its archive file (`rellis_sample.zip` or `RELLIS-3D.zip`) in any of the following locations:
1. `./data/raw/public/RELLIS-3D/`
2. `./data/raw/RELLIS-3D/`
3. Any custom directory specified in `configs/dataset.yaml`.

The dataset directory must contain camera sequence folders (e.g., `00000/`, `00001/`) with:
- `pylon_camera_node/`: Color images (`*.jpg` or `*.png`).
- `pylon_camera_node_label_id/`: Semantic mask images (`*.png`).

---

## 4. How Execution Works

### 1. GPU Verification:
`scripts/verify_gpu.py` checks for an active NVIDIA CUDA device. If CUDA is unavailable, it terminates immediately with exit code 1 to strictly prevent unauthorized CPU training.

### 2. Dataset Discovery & Extraction:
`scripts/discover_dataset.py` searches candidate paths. If a `.zip` or `.tar.gz` archive is detected, `scripts/extract_dataset.py` unpacks it into the target folder without overwriting unrelated files.

### 3. Training & Fine-Tuning:
`scripts/train_gpu.py` loads `configs/training.yaml`, initializes weights from `checkpoints/rugd_baseline_best.pt`, and trains on the GPU using Automatic Mixed Precision (AMP) and safety-weighted focal/dice loss.

### 4. Evaluation & Costmap Inference:
`scripts/evaluate.py` calculates overall mIoU, per-class IoU, dangerous-terrain recall, and false-safe pixel ratios. `scripts/infer_gpu.py` generates visual traversability overlays and 2D BEV costmaps.

---

## 5. Summary of Actions & Status

- **Files Added:** Complete `rellis3d_gpu_ready/` directory structure, standalone scripts, configs, source models, and baseline checkpoints.
- **Production Code Status:** `ai/planning/dynamic/` was **NOT MODIFIED (ZERO DIFF)**.
- **Executed Steps:** Host environment discovery (M13.20.0), RELLIS-3D filesystem discovery (M13.20.1), authoritative online research (M13.20.2), RUGD baseline audit (M13.20.3), package assembly (M13.20.8).
- **Pending Steps:** Physical GPU training (M13.20.5) and empirical RELLIS validation (M13.20.6) are **BLOCKED** on the local host due to absence of an NVIDIA CUDA GPU.

---

## 6. License & Usage Considerations

- **RUGD Dataset:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (`CC BY-NC-SA 4.0`).
- **RELLIS-3D Dataset:** Creative Commons Attribution-NonCommercial 4.0 (`CC BY-NC 4.0`).
- **Permissible Use:** Non-commercial academic hackathons and research prototypes only.
