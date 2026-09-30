# Phase 13.26A — Intel GPU/XPU Perception Acceleration State & Plan

**Date**: 2026-09-30T17:25:00+05:30  
**Starting Commit**: `3fe36cb`  
**Current Branch**: `demo-01-interactive-ugv`  
**Target Hardware**: Intel Core Ultra 7 155H (Meteor Lake), Intel Arc Xe-LPG Integrated GPU (8 Xe cores), Intel AI Boost NPU  
**Status**: INITIAL AUDIT & BASELINE ESTABLISHED  

---

## 1. Starting State Audit

### Model Implementation
- **Source**: `ai/models/traversability_net.py` -> `LightweightTraversabilityNet`
- **Architecture**: MobileNetV3-Small encoder (`mobilenet_v3_small`, ~3.2M params) with multi-scale skip connections and lightweight decoder.
- **Semantic Classes**: 3 classes (0: SAFE, 1: UNKNOWN/RISKY, 2: UNSAFE).
- **Frozen Weights**: `outputs/rellis3d/checkpoints/best.pt` (trained on RELLIS-3D).

### Predictor Wrapper API
- **Source**: `ai/traversability/dense_predictor.py` -> `DenseTraversabilityPredictor(TraversabilityPredictorBase)`
- **Current Signature**:
  ```python
  def __init__(
      self,
      checkpoint: Optional[Union[str, Path]] = None,
      device: Optional[str] = None,
      target_wh: Tuple[int, int] = (640, 400),
      overlay_alpha: float = 0.45
  ) -> None:
  ```
- **Inference Entrypoint**:
  ```python
  def predict(self, image: Union[np.ndarray, str, Path]) -> TraversabilityResult:
  ```
- **Output Data Structure**: `TraversabilityResult(mask, logits, overlay, image, input_shape, inference_s, device, model_name, confidence, unknown_mask)`

### Current Device Path & Limitations
- If `device is None`, `DenseTraversabilityPredictor` defaults to `"cuda:0" if torch.cuda.is_available() else "cpu"`.
- It does **not** auto-detect Intel XPU (`torch.xpu.is_available()`).
- Synchronization in `predict()`:
  ```python
  if self._torch_device.type == "cuda":
      torch.cuda.synchronize()
  ```
  It synchronizes for CUDA, but **not for XPU** (`torch.xpu.synchronize()`). Because XPU execution is asynchronous on SYCL streams, accurate timing requires synchronizing XPU devices.
- Explicit device request: If `device="xpu"` is requested, it does not validate XPU availability or give an actionable error message if XPU is unavailable.

### Available Intel XPU Runtime
- **Virtual Environment**: `.venv_xpu` located at `c:\Users\shres\Desktop\SIH2026\.venv_xpu`.
- **PyTorch Version**: `2.6.0+xpu`
- **XPU Availability**: `torch.xpu.is_available() == True`
- **Active Device**: `xpu:0` -> `Intel(R) Graphics` (Meteor Lake Intel Arc integrated GPU).
- **SYCL Kernel Execution**: Verified with direct tensor matrix multiplication and initial model forward pass.

---

## 2. Intended Minimal, Safe, Additive Changes

1. **Extend Device Selection in `DenseTraversabilityPredictor`**:
   - Support `device="auto"`, `device="xpu"`, `device="cpu"`, and `device="cuda"`.
   - In `auto` mode:
     1. If CUDA is available -> `"cuda:0"`
     2. If XPU is available and validated -> `"xpu:0"`
     3. Otherwise -> `"cpu"`
   - In explicit mode (e.g., `device="xpu"` or `device="xpu:0"`):
     If XPU is unavailable, **raise `RuntimeError` immediately with a clear, actionable error message**. Do **not** silently fall back to CPU.
2. **Proper Hardware Synchronization for Accurate Latency**:
   - In `predict()`:
     ```python
     if self._torch_device.type == "cuda":
         torch.cuda.synchronize(self._torch_device)
     elif self._torch_device.type == "xpu":
         torch.xpu.synchronize(self._torch_device)
     ```
3. **Preserve Exact Existing API & Downstream Semantics**:
   - `TraversabilityResult` structure and types remain 100% identical.
   - Preprocessing (`cv2.resize` to `(640, 400)`, ImageNet normalization) and postprocessing (`F.softmax`, drivability calculation, overlay rendering) remain unchanged.
   - No modifications to C++ DWA, tracking, costmaps, safety governor, or ROS2 nodes.
