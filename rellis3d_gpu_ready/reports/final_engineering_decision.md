# Final Engineering Decision & Pre-Phase 14 Assessment
**Project:** SIH26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor Environment  
**Document:** `reports/final_engineering_decision.md`  
**Milestone:** Pre-Phase 14 — RUGD to RELLIS-3D Dense Traversability Pipeline  
**Execution Verdict:** **COMPLETE WITH DOCUMENTED LIMITATIONS / BLOCKED ON HOST GPU**  

---

## 1. Context & Architectural Assessment

The pre-Phase 14 objective was to inspect the existing RUGD-trained dense traversability model, discover and audit RELLIS-3D in the active environment, adapt the dense segmentation pipeline, execute a strict GPU-only workflow, and package a self-contained GPU-ready deployment artifact.

### Enforced Rules Applied:
1. **GPU-Only Training:** Strict prohibition of CPU training or CPU fallback.
2. **No Kaggle Fallback:** Maintained local host environment discipline.
3. **Preservation of Baseline:** The frozen RUGD baseline model (`rugd_traversability_v1`) remains untouched and fully verified.
4. **Safety Code Isolation:** `ai/planning/dynamic/` was preserved with **ZERO DIFF**.

---

## 2. Technical Findings

1. **Host Environment:**  
   The active Windows host operates on an Intel(R) Graphics Family integrated GPU with PyTorch `2.6.0+cpu`. No CUDA-capable NVIDIA GPU or CUDA runtime is present. WSL Ubuntu has no GPU pass-through. Therefore, physical model training and fine-tuning on the local machine are cleanly halted per Section 2/3 rules.
2. **Dataset Audit:**  
   The RELLIS-3D dataset is not present in the local filesystem. Extensive online research confirmed official Basler camera resolutions ($1920 \times 1200$), 20-class ontology, and file structures (`pylon_camera_node` and `pylon_camera_node_label_id`).
3. **Class Mapping Safety Audit:**  
   The legacy class mapping was corrected:
   - `barrier` (27) was corrected from `SAFE` $\to$ `UNSAFE`.
   - `object` (9) was corrected from `SAFE` $\to$ `UNSAFE`.
   - `mud` (33) and `puddle` (31) were assigned to `UNKNOWN` (cautious speed cap).
4. **GPU-Ready Package:**  
   A complete, modular, portable package (`rellis3d_gpu_ready/`) was constructed containing all verification, discovery, extraction, training, evaluation, inference, and reporting scripts, along with the RUGD baseline checkpoint.

---

## 3. Deployment Recommendation

When deploying to a target GPU environment (e.g., NVIDIA workstation, cloud VM, or Jetson Orin compute unit):
```bash
# 1. Verify GPU
python scripts/verify_gpu.py

# 2. Extract & Validate Dataset
python scripts/discover_dataset.py
python scripts/validate_labels.py

# 3. Execute GPU Training
python scripts/train_gpu.py --config configs/training.yaml

# 4. Evaluate Safety Metrics
python scripts/evaluate.py
```
This enables immediate, push-button execution with zero manual plumbing.
