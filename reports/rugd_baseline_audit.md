# RUGD BASELINE MODEL AUDIT REPORT
**Target Model:** `LightweightTraversabilityNet`  
**Checkpoint Path:** `models/rugd_traversability/outputs/checkpoints/best.pt`  
**Checkpoint SHA-256:** `a49d4222a208b0d79aa2a1df9c8d28c5f31809955bd0a85336fc92d5e849cd91`  
**Status:** PRESERVED & SECURED (Read-Only)

---

## 1. Architectural Specification
- **Architecture:** Lightweight Encoder-Decoder with Multi-Scale Skip Connections.
- **Backbone:** `MobileNetV3-Small` (pretrained on ImageNet, fine-tuned on RUGD).
  - Stage 1 (`enc1`): Features 0-1, 16 channels, downsample /2.
  - Stage 2 (`enc2`): Features 2-3, 24 channels, downsample /4.
  - Stage 3 (`enc3`): Features 4-8, 48 channels, downsample /8.
  - Stage 4 (`enc4`): Features 9-11, 96 channels, downsample /16.
- **Decoder:** Feature Pyramid Upsampling blocks with skip connections:
  - `dec3`: 96 + 48 -> 64 channels.
  - `dec2`: 64 + 24 -> 48 channels.
  - `dec1`: 48 + 16 -> 32 channels.
- **Head:** `Conv2d(32, 16, 3)` -> `Hardswish` -> `Dropout(0.2)` -> `Conv2d(16, 3, 1)`.
- **Parameter Count:** 1,152,067 parameters (~4.55 MB uncompressed state_dict).

---

## 2. Input / Output Contracts
- **Input Channels:** 3 (RGB).
- **Nominal Resolution:** 640x480 (also fully compatible with 640x400 aspect-ratio preserving mode).
- **Normalization:** ImageNet standard:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- **Output Channels:** 3 logits (`[Batch, 3, Height, Width]`).
- **Semantic Classes:**
  - `0: SAFE` (Flat terrain, paved trail, benign short grass).
  - `1: UNKNOWN` (Dense brush, shadows, deep mud, water reflections).
  - `2: UNSAFE` (Lethal obstacles, trees, boulders, vehicles, steep drop-offs).
- **Ignore Index:** `255` (Void, ego-hood, sky - masked during training).

---

## 3. Training & Optimization History (RUGD Baseline)
- **Loss:** `CombinedTraversabilityLoss` (Focal Cross-Entropy + Soft Dice Loss).
- **Optimizer:** `AdamW` (learning rate: `5e-4`, weight decay: `1e-4`).
- **Scheduler:** `ReduceLROnPlateau` with patience=5.
- **Augmentation:** Random horizontal flip, color jitter, affine rotation/scale.
- **Epochs Completed:** 38 epochs (Early stopped at epoch 38; best checkpoint saved at epoch 31).
- **Baseline Performance Metrics:**
  - Validation mIoU: **0.9803**
  - Safe Class IoU: **0.9896**
  - Unknown Class IoU: **0.9981**
  - Unsafe Class IoU: **0.9553**
  - Unsafe Recall: **97.09%**
  - False-Safe Pixel Ratio: **2.21%**

---

## 4. Preservation & Non-Regression Guarantee
1. The checkpoint `models/rugd_traversability/outputs/checkpoints/best.pt` has been cryptographically fingerprinted (SHA-256: `a49d4222a208b0d79aa2a1df9c8d28c5f31809955bd0a85336fc92d5e849cd91`).
2. Snapshot is stored in `artifacts/pre_rellis_training_snapshot/` and `artifacts/pre_rellis_training_snapshot.tar.gz`.
3. Under no circumstances will this baseline file be overwritten, modified, or deleted during RELLIS-3D training.
4. RELLIS-3D training will output to a distinct tree: `outputs/rellis3d/checkpoints/`.
