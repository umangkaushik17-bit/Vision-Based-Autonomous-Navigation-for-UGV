# SIH26126 — External GPU Execution Guide

This guide details how to upload and autonomously execute **Phase 3 Traversability AI Full GPU Training** in an external cloud GPU environment (Google Colab, Kaggle, or a cloud GPU VM).

---

## 1. Quick Start (Colab / Kaggle 1-Click Execution)

### Recommended: Fast 2-Zip Workflow (Saves Upload Time!)
We split the package into **Code** and **Database**:
1. **`gpu_code.zip` (~90 KB)**: Uploads in 1 second. Re-upload this whenever you modify code.
2. **`gpu_data.zip` (~1.25 GB)**: Upload **once** to Google Drive or your Colab instance.

```bash
# 1. Unpack code package (instant)
unzip -q -o gpu_code.zip

# 2. Unpack database package (or leave it in root, auto_run extracts it automatically)
unzip -q -o gpu_data.zip

# 3. Install requirements & execute
cd gpu_package
pip install -r requirements_gpu.txt
python auto_run.py --stage all
```

### Alternative: Single-Zip Workflow
If you prefer a single archive:
```bash
unzip -q -o gpu_package.zip
cd gpu_package
pip install -r requirements_gpu.txt
python auto_run.py --stage all
```

Download the resulting **`gpu_results.zip`** archive back to your local Antigravity workstation.

---

## 2. Granular Stage Execution

You can run individual stages independently:

| Stage | Command | Purpose |
| :--- | :--- | :--- |
| **1. Setup** | `python auto_run.py --stage setup` | Detects GPU (CUDA/cuDNN) and verifies Python libraries |
| **2. Data** | `python auto_run.py --stage data` | Validates dataset manifest and anti-leakage governance |
| **3. Smoke Test** | `python auto_run.py --stage smoke-test` | Runs a 1-epoch GPU dry run to verify forward & backward passes |
| **4. Train** | `python auto_run.py --stage train` | Launches full 50-epoch training with class-weighted Focal Loss, AMP & early stopping |
| **5. Evaluate** | `python auto_run.py --stage evaluate` | Evaluates checkpoint across synthetic and real-world test splits |
| **6. Benchmark** | `python auto_run.py --stage benchmark` | Benchmarks latency (ms), FPS, and model parameters on the GPU |
| **7. Export** | `python auto_run.py --stage export` | Exports model to TorchScript and verified ONNX |
| **8. Package** | `python auto_run.py --stage package` | Bundles all weights, reports, and ONNX files into `gpu_results.zip` |

---

## 3. Importing Results Back to Antigravity

Once `gpu_results.zip` is downloaded to your workstation:
```powershell
cd c:\Users\shres\Desktop\SIH2026
python scripts/import_gpu_results.py --input path\to\gpu_results.zip
```
This automatically unpacks checkpoints to `outputs/checkpoints/` and evaluation reports to `outputs/reports/`, updating the project status.
