# SIH26126 — Technology Stack & Dependency Audit

This document details the software dependencies, runtime environments, hardware capabilities, and version compatibility matrix for the project.

---

## 1. Host Environment Audit (Windows 11)

| Parameter | Value / Status | Verification Command | Notes |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 11 (x86_64) | `[System.Environment]::OSVersion` | Primary development host |
| **Python Version** | Python 3.10.10 | `python --version` | Located in user AppData |
| **Alternate Pythons** | Python 3.12, 3.13 | `where.exe python` | Detected on host system |
| **PyTorch** | 2.6.0+cpu | `python -c "import torch; ..."` | CPU build active on host |
| **TorchVision** | 0.21.0 | `python -c "import torchvision; ..."` | Operational |
| **OpenCV** | 4.8.1 / 5.0.0 | `python -c "import cv2; ..."` | Multi-threaded CPU build |
| **NumPy** | 2.2.6 | `python -c "import numpy; ..."` | Compatible with PyTorch 2.6 |
| **SciPy** | 1.15.2 | `python -c "import scipy; ..."` | Scientific computing & spline math |
| **Hardware GPU** | No local NVIDIA GPU | `where.exe nvidia-smi` | CPU-only local execution |
| **WSL Subsystem** | WSL2 (Ubuntu 24.04.3 LTS) | `wsl.exe -l -v` | Installed, available for ROS 2 |
| **ROS 2 (Host)** | Not installed natively | `where.exe ros2` | Intentionally deferred to WSL2 |
| **Gazebo (Host)** | Not installed natively | `where.exe gazebo` | Intentionally deferred to WSL2 |
| **MATLAB** | Not in PATH | `where.exe matlab` | Scripts will have Python equivalents |

---

## 2. Cloud GPU Environment (NVIDIA A100)

| Parameter | Specification | Purpose |
| :--- | :--- | :--- |
| **Compute Target** | NVIDIA A100 (40GB / 80GB SXM4/PCIe) | Cloud model training & distillation |
| **Acceleration** | Ampere Tensor Cores with AMP (FP16/BF16) | $10\times$ faster training than local CPU |
| **Platforms** | Google Colab Pro / Kaggle / Lambda / AWS | Zero-cost or low-cost cloud execution |
| **Deployment Flow** | `ai/training/train_traversability_a100.py` | Headless execution or Colab notebook |
| **Export Target** | ONNX (Opset 14) / TorchScript | Quantized for zero-lag edge CPU inference |

---

## 3. Dependency Inventory

### Core Installed Packages
```
numpy==2.2.6
scipy==1.15.2
pyyaml==6.0.3
pydantic==2.11.9
torch==2.6.0+cpu
torchvision==0.21.0
opencv-python==4.8.1.78
pillow==12.0.0
matplotlib==3.8.4
pytest==9.1.1
timm==0.9.7
segmentation-models-pytorch==0.3.4
```

### Known Warnings & Fixes
- **Warning:** `WARNING: Ignoring invalid distribution -5py` and `-angchain` in `site-packages`.
- **Root Cause:** Incomplete uninstallation artifacts where pip renames folders with a `~` prefix (`~5py`, `~angchain`, etc.).
- **Impact:** Harmless warning emitted by pip; does not affect runtime execution or tests.
- **Resolution:** Can be safely deleted during site-packages maintenance if desired.

---

## 4. Subsystem Responsibility Matrix

```
┌────────────────────────────────────────────────────────────────────────┐
│                        COMPUTE ARCHITECTURE                            │
├────────────────────────────────────────────────────────────────────────┤
│  LOCAL WORKSPACE (Windows 11)                                          │
│  ├── Phases 0–7: Pure Python + OpenCV + PyTorch                        │
│  │   • Computer Vision preprocessing & contour extraction              │
│  │   • Lightweight traversability inference (CPU / ONNX Runtime)      │
│  │   • 2D/3D costmap generation & inflation                            │
│  │   • Global & Local path planners (A*, Hybrid A*, DWA)               │
│  │   • Visual odometry (optical flow tracking)                         │
│  │   • Multi-tier risk governor & safety watchdogs                     │
│  │   • Interactive visualization HUD & diagnostic plots                │
│  │                                                                     │
│  CLOUD ACCELERATOR (NVIDIA A100)                                       │
│  ├── Phases 2–4: Heavy ML Training & Distillation                     │
│  │   • Large-scale outdoor terrain dataset training (RUGD, RELLIS-3D)  │
│  │   • Deep neural network distillation into edge student models       │
│  │   • Automated ONNX / TensorRT export                                │
│  │                                                                     │
│  LINUX SUBSYSTEM (WSL2 Ubuntu 24.04 LTS)                               │
│  └── Phases 8–16: ROS 2 & Gazebo Integration                           │
│      • ROS 2 Jazzy communication nodes & TF2 trees                     │
│      • Nav2 lifecycle navigation nodes                                 │
│      • Gazebo physics engine & 4WD UGV simulated model                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Requirements Strategy

1. **Self-Contained Packaging:** All custom modules are organized inside `src/` and registered under the editable `sih26126-ugv` package.
2. **Zero Hard Dependencies on External Middleware for Phases 0–7:** Algorithms are verified in pure Python first before coupling to ROS 2 or Gazebo. This prevents early integration bottlenecks.
3. **Reproducible Test Harness:** Tests run via `pytest` and must achieve 100% pass rate before advancing to each subsequent phase.
