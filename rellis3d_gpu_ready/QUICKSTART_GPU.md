# QUICKSTART GUIDE — RELLIS-3D GPU PIPELINE

Follow these steps when running on any system equipped with an active NVIDIA CUDA GPU (e.g., Linux workstation, remote GPU server, or Jetson Orin):

---

## 1. Setup Environment

```bash
# Option A: Conda
conda env create -f environment.yml
conda activate rellis3d-traversability

# Option B: Pip
pip install -r requirements-gpu.txt
```

---

## 2. Verify GPU & Hardware

```bash
python scripts/verify_gpu.py
```
*Expected output: Exit code 0, device name, VRAM, and fast tensor multiplication benchmark.*

---

## 3. Mount or Place Dataset

Place your RELLIS-3D directory or archive in `./data/raw/public/RELLIS-3D` or specify its path in `configs/dataset.yaml`.

Run discovery and label audit:
```bash
python scripts/discover_dataset.py
python scripts/inspect_dataset.py
python scripts/validate_labels.py
```

---

## 4. Run Smoke Test (Bounded 2-Epoch Verification)

```bash
python scripts/train_gpu.py --smoke-test
```

---

## 5. Execute Full GPU Fine-Tuning

```bash
python scripts/train_gpu.py --config configs/training.yaml
```

---

## 6. Evaluate & Run Inference

```bash
# Evaluate checkpoint
python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt

# Run inference on sample image
python scripts/infer_gpu.py --image sample.jpg --checkpoint outputs/checkpoints/best.pt
```

---

## 7. Master One-Click Runner

```bash
python run_gpu_pipeline.py
# Or on Linux:
bash run_gpu_pipeline.sh
```
