# RUGD Baseline Pipeline Architecture & Checkpoint Audit
**Project:** SIH26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor Environment  
**Document:** `reports/rugd_pipeline_audit.md`  
**Model Name:** `RUGD-Traversability-Net (Model B)`  
**Baseline Model ID:** `rugd_traversability_v1`  
**Audit Date:** September 2026  

---

## 1. Executive Summary

The existing RUGD-trained dense traversability model represents a fully trained, tested, and validated baseline for off-road traversability estimation. This audit formally recovers, inspects, and documents the model architecture, checkpoint artifacts, input/output contracts, and performance baselines to guarantee that:
1. The RUGD baseline remains permanently available as an independent reference.
2. The model weights, ONNX exports, and TorchScript artifacts are mathematically verified.
3. The traversability-to-costmap interface contract is clearly preserved for downstream planning.

---

## 2. Checkpoint Verification & Hashes

The primary model directory is located at [`models/rugd_traversability/`](file:///c:/Users/shres/Desktop/SIH2026/models/rugd_traversability/). All files have been verified on disk:

| File Name | Relative Path | Format | Size (Bytes) | SHA-256 Checksum | Status |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `best.pt` | `models/rugd_traversability/outputs/checkpoints/best.pt` | PyTorch State Dict | 4,549,086 | `a49d4222a208b0d79aa2a1df9c8d28c5f31809955bd0a85336fc92d5e849cd91` | **VERIFIED** |
| `latest.pt` | `models/rugd_traversability/outputs/checkpoints/latest.pt` | Checkpoint + Optimizer | 13,517,655 | `6c700d47f9f3523bb64417c5e9270f2f1b8080809b2a38f63b24fa283391eac8` | **VERIFIED** |
| `traversability_model.onnx` | `models/rugd_traversability/outputs/exports/traversability_model.onnx` | ONNX Runtime | 4,426,805 | `0ec992665be7d1cd4a8d5911f0dad1cb933b506699a7cee7e2b754691107752f` | **VERIFIED** |
| `traversability_model.pt` | `models/rugd_traversability/outputs/exports/traversability_model.pt` | TorchScript JIT | 4,882,206 | `07bbd200f36ae551153b9bfbb44f289314cfdad2f5fd8671956a4a419e6746fe` | **VERIFIED** |
| `model_manifest.json` | `models/rugd_traversability/model_manifest.json` | JSON Metadata | 2,189 | Manifest definition | **VERIFIED** |

---

## 3. Network Architecture

The architecture is implemented in [`ai/models/traversability_net.py`](file:///c:/Users/shres/Desktop/SIH2026/ai/models/traversability_net.py) as `LightweightTraversabilityNet`:
- **Backbone:** MobileNetV3-Small feature extractor with ImageNet initialization.
  - Encoder Stage 1 (`enc1`): Features 0–1 $\to$ 16 channels, stride $/2$.
  - Encoder Stage 2 (`enc2`): Features 2–3 $\to$ 24 channels, stride $/4$.
  - Encoder Stage 3 (`enc3`): Features 4–8 $\to$ 48 channels, stride $/8$.
  - Encoder Stage 4 (`enc4`): Features 9–11 $\to$ 96 channels, stride $/16$.
- **Decoder:** Feature Pyramid upsampling with skip connections via `ConvBlock` (Conv2d + BatchNorm2d + Hardswish):
  - `dec3`: 96 + 48 $\to$ 64 channels.
  - `dec2`: 64 + 24 $\to$ 48 channels.
  - `dec1`: 48 + 16 $\to$ 32 channels.
- **Head:** Dropout2d ($p=0.20$) + Conv2d(32, 32, 3) + BatchNorm2d + Hardswish + Conv2d(32, `num_classes`, 1).
- **Total Parameters:** $\sim 1.15\,\text{M}$ parameters ($\sim 4.5\,\text{MB}$ file size), highly edge-optimized for embedded mobile compute.

---

## 4. Input & Output Contract

### Input Specification:
- **Color Space:** RGB format.
- **Resolution:** Fixed target resolution of $480 \times 640$ ($H \times W$).
- **Normalization:** Standard ImageNet statistics:
  $$\text{Mean} = [0.485, 0.456, 0.406], \quad \text{Std} = [0.229, 0.224, 0.225]$$
- **Input Tensor Shape:** $[B, 3, 480, 640]$ `float32`.

### Output Specification:
- **Raw Logits:** $[B, 3, 480, 640]$ `float32`.
- **Classes:** 3 canonical semantic categories:
  - Class 0: `SAFE` (Flat ground, trail, dry dirt, benign grass).
  - Class 1: `UNKNOWN` (Overgrown brush, obscured shadows, transition zones).
  - Class 2: `UNSAFE` (Boulders, ditches, tree trunks, water, obstacles).
- **Softmax Probabilities:** $[B, 3, 480, 640] \in [0.0, 1.0]$.
- **Downstream Safety Adapter:**
  - `prob_map` = Probability of Class 0 (`SAFE`).
  - `unknown_map` = Probability of Class 1 (`UNKNOWN`).
  - `confidence_map` = $\max(\text{probs}, \text{dim}=1)$.
  - `unknown_mask` = `unknown_map` $\ge 0.50$.

---

## 5. RUGD Baseline Benchmark Metrics

From `models/rugd_traversability/outputs/reports/evaluation_report.json`:
- **Test Samples Evaluated:** 28 representative benchmark sequences.
- **Overall Mean IoU (mIoU):** **0.9810** ($98.10\%$).
- **Per-Class IoU:**
  - Safe IoU: **0.9896**
  - Unknown IoU: **0.9981**
  - Unsafe IoU: **0.9553**
- **Safety Critical Metrics:**
  - **Unsafe Recall (Obstacle Detection):** **0.9709** ($97.09\%$ of lethal hazards correctly classified).
  - **False-Safe Ratio:** **0.0221** ($2.21\%$, well below the $\le 3\%$ safety ceiling).
  - **Traversable Precision:** **0.9901** ($99.01\%$).

---

## 6. Training & Inference Entry Points

- **Training Entry Point:** [`scripts/train_traversability.py`](file:///c:/Users/shres/Desktop/SIH2026/scripts/train_traversability.py)
  - Features AMP (`torch.cuda.amp.autocast`), `GradScaler`, gradient accumulation (4 steps), `EarlyStopping` on validation mIoU, and `CombinedTraversabilityLoss` (Focal Loss + Dice Loss).
- **Inference Entry Point:** [`ai/traversability/rugd_model.py`](file:///c:/Users/shres/Desktop/SIH2026/ai/traversability/rugd_model.py)
  - Implements `RUGDTraversabilityPredictor(TraversabilityPredictorBase)`.
  - Supports both PyTorch and ONNX runtime backends.
- **Costmap Adapter:** [`ai/traversability/bev_projection.py`](file:///c:/Users/shres/Desktop/SIH2026/ai/traversability/bev_projection.py)
  - Projects `prob_map` into Bird's Eye View (BEV) rolling costmap grids.

---

## 7. Audit Conclusion

The RUGD baseline is healthy, modular, and fully functional. It will remain locked as the authoritative reference model for all cross-domain evaluations against RELLIS-3D.
