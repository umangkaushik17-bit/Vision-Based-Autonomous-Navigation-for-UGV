# Kaggle GPU Cloud Integration & Automated Test Runner

## 1. Overview

The SIH2026 UGV project now features an automated **Kaggle Cloud GPU Bridge**. This integration allows Antigravity IDE to dispatch heavy simulations, machine learning training, and test benchmarks to Kaggle's cloud GPUs (NVIDIA Tesla P100 16GB / Tesla T4 x 2) completely headlessly and retrieve generated artifacts back to the local repository.

```
+-------------------------------------------------------------+
|                     Local Environment                       |
|  - Laptop / Antigravity IDE (Intel Core Ultra 7 CPU)        |
|  - Authoritative Codebase (ai/, simulation/, tests/, etc.)  |
+------------------------------+------------------------------+
                               |
                   [scripts/kaggle_runner.py]
                               |
        1. Sync Lean Code Package (tools/kaggle/dataset/)
        2. Push Kernel Job (tools/kaggle/ugv_gpu_runner.py)
        3. Poll Execution Status via Official Kaggle API
        4. Auto-Download Results & Plots
                               |
                               v
+-------------------------------------------------------------+
|                     Kaggle Cloud GPU                        |
|  - Hardware: NVIDIA Tesla P100 (16GB VRAM) / Tesla T4 x 2   |
|  - Dataset: umangkaushik17/sih2026-ugv-code (Mounted)       |
|  - Kernel: umangkaushik17/sih2026-ugv-gpu-runner           |
|  - Outputs Staged: /kaggle/working/outputs/                 |
+-------------------------------------------------------------+
```

---

## 2. Core Components

| Component | Path | Description |
|---|---|---|
| **Cloud Controller** | [`scripts/kaggle_runner.py`](file:///c:/Users/shres/Desktop/SIH2026/scripts/kaggle_runner.py) | CLI automation manager for code packaging, kernel submission, status polling, and artifact retrieval. |
| **GPU Execution Harness** | [`tools/kaggle/ugv_gpu_runner.py`](file:///c:/Users/shres/Desktop/SIH2026/tools/kaggle/ugv_gpu_runner.py) | Headless cloud script that verifies CUDA, unpacks workspace into `/tmp/SIH2026`, runs pytest/benchmarks, and exports output artifacts to `/kaggle/working/outputs/`. |
| **Kernel Metadata** | [`tools/kaggle/kernel-metadata.json`](file:///c:/Users/shres/Desktop/SIH2026/tools/kaggle/kernel-metadata.json) | Kaggle kernel specification with GPU and internet enabled and dataset mounted. |
| **Local Outputs** | [`simulation/outputs/kaggle/`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/kaggle/) | Destination directory for downloaded logs, metrics JSON, and visualization plots. |

---

## 3. Quick Usage Guide

### A. Run Any Command on Cloud GPU (One-Click)
To automatically sync your latest local code changes, submit the GPU job, wait for completion, and download all generated plots and metrics:

```bash
python scripts/kaggle_runner.py --run "<your_command_here>"
```

**Examples:**
```bash
# Run dynamic SIL core scenarios:
python scripts/kaggle_runner.py --run "pytest tests/test_dynamic_sil_core_scenarios.py -v"

# Run Phase 14 multi-obstacle stress tests:
python scripts/kaggle_runner.py --run "pytest tests/test_phase14_planner_stress.py -v"

# Run full dynamic navigation SIL benchmark:
python scripts/kaggle_runner.py --run "python run_dynamic_sil.py"
```

### B. Manual Step-by-Step Control
```bash
# 1. Sync local code package to Kaggle dataset
python scripts/kaggle_runner.py --sync-code

# 2. Submit GPU kernel
python scripts/kaggle_runner.py --submit

# 3. Check execution status (QUEUED, RUNNING, COMPLETE, ERROR)
python scripts/kaggle_runner.py --status

# 4. Download output plots and metrics
python scripts/kaggle_runner.py --download
```

---

## 4. Initial Verification Record

- **Date**: 2026-09-06
- **Kaggle Account**: `umangkaushik17`
- **Assigned Cloud GPU**: NVIDIA Tesla P100-PCIE-16GB (15.89 GB VRAM, CUDA 13.0 Driver 580.159.04)
- **Benchmark Executed**: `pytest tests/test_dynamic_sil_core_scenarios.py -v`
- **Result**: `12 passed, 13 warnings in 344.32s (0:05:44)`
- **Exit Code**: `0 (SUCCESS)`
- **Downloaded Artifacts**:
  - `simulation/outputs/kaggle/sih2026-ugv-gpu-runner.log`
  - `simulation/outputs/kaggle/outputs/results/`
  - `simulation/outputs/kaggle/outputs/plots/`
