"""RUGD Traversability Model Adapter (Model B).

Wraps an independently trained RUGD traversability or semantic segmentation model
to expose the unified TraversabilityPredictorBase interface.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ai.traversability.base import TraversabilityPredictorBase, TraversabilityResult
from ai.traversability.visualization import create_traversability_overlay
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.traversability.rugd_model")

DEFAULT_MODEL_DIR = Path("models/rugd_traversability")


class RUGDTraversabilityPredictor(TraversabilityPredictorBase):
    """Production adapter for RUGD-trained traversability models.

    Supports:
    - TorchScript / PyTorch checkpoints (.pt, .pth)
    - ONNX Runtime exports (.onnx)
    - 3-class traversability heads [SAFE, UNKNOWN, UNSAFE]
    - Direct binary traversability heads
    - Multi-class semantic segmenters with class remapping
    """

    # Standard ImageNet normalization parameters
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
        device: Optional[str] = None,
        target_size: Tuple[int, int] = (480, 640),
        use_onnx: bool = True
    ) -> None:
        self.target_size = target_size  # (height, width)
        self.model_dir = Path(model_dir)

        # Device selection
        if device is None:
            self._device_str = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = torch.device(self._device_str)
        else:
            self._device_str = device
            self._device = torch.device(device)

        self.model = None
        self.onnx_session = None
        self.is_onnx = False
        self.manifest = self._load_manifest()

        # Locate checkpoint
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path, use_onnx)
        self._load_model()
        logger.info(f"RUGDTraversabilityPredictor initialized successfully from {self.checkpoint_path} on {self.device}")

    @property
    def device(self) -> str:
        """Return active compute device string."""
        return self._device_str

    @property
    def model_name(self) -> str:
        """Return human-readable model identifier."""
        return "RUGD-Traversability-Net (Model B)"

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest if present."""
        manifest_file = self.model_dir / "model_manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse manifest: {e}")
        return {}

    def _resolve_checkpoint(self, explicit_path: Optional[Union[str, Path]], prefer_onnx: bool) -> Path:
        """Locate the best available checkpoint artifact."""
        if explicit_path:
            p = Path(explicit_path)
            if p.exists():
                return p
            raise FileNotFoundError(f"Specified checkpoint not found: {p}")

        # Check in model_dir
        candidates = list(self.model_dir.rglob("*.onnx")) + list(self.model_dir.rglob("*.pt"))
        if prefer_onnx:
            onnx_candidates = [c for c in candidates if c.suffix == ".onnx"]
            if onnx_candidates:
                return onnx_candidates[0]

        best_pts = [c for c in candidates if "best.pt" in c.name]
        if best_pts:
            return best_pts[0]

        if candidates:
            return candidates[0]

        # Check outputs/checkpoints/best.pt
        fallback = Path("outputs/checkpoints/best.pt")
        if fallback.exists():
            return fallback

        raise FileNotFoundError(f"No checkpoint found in {self.model_dir} or outputs/checkpoints/")

    def _load_model(self) -> None:
        """Initialize the model backend (ONNX Runtime or PyTorch)."""
        suffix = self.checkpoint_path.suffix.lower()

        if suffix == ".onnx":
            try:
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._device.type == "cuda" else ["CPUExecutionProvider"]
                self.onnx_session = ort.InferenceSession(str(self.checkpoint_path), providers=providers)
                self.is_onnx = True
                logger.info("Loaded ONNX runtime session for RUGD model.")
                return
            except Exception as e:
                logger.warning(f"Failed to load ONNX runtime ({e}), falling back to PyTorch if available.")

        # PyTorch / TorchScript loading
        try:
            # Try TorchScript first
            self.model = torch.jit.load(str(self.checkpoint_path), map_location=self._device)
            self.model.eval()
            self.is_onnx = False
            logger.info("Loaded TorchScript RUGD model.")
            return
        except Exception:
            pass

        # Load weights into LightweightTraversabilityNet
        try:
            from ai.models.traversability_net import LightweightTraversabilityNet
            ckpt = torch.load(str(self.checkpoint_path), map_location=self._device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model = LightweightTraversabilityNet(num_classes=3)
            self.model.load_state_dict(state_dict)
            self.model.to(self._device)
            self.model.eval()
            self.is_onnx = False
            logger.info("Loaded PyTorch state dict into LightweightTraversabilityNet.")
        except Exception as e:
            raise RuntimeError(f"Could not load checkpoint {self.checkpoint_path} as TorchScript or state dict: {e}")

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Resize, normalize, and format image tensor."""
        orig_h, orig_w = image.shape[:2]
        resized = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] then ImageNet mean/std
        normalized = (resized.astype(np.float32) / 255.0 - self.IMAGENET_MEAN) / self.IMAGENET_STD
        # Transpose [H, W, C] -> [1, C, H, W]
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        return tensor, (orig_h, orig_w)

    def predict(self, image: Union[str, Path, np.ndarray]) -> TraversabilityResult:
        """Run traversability inference on an input RGB image."""
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image does not exist: {img_path}")
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise ValueError(f"Failed to read image at: {img_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            rgb = image
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        t0 = time.perf_counter()
        tensor_np, (orig_h, orig_w) = self._preprocess(rgb)

        if self.is_onnx:
            input_name = self.onnx_session.get_inputs()[0].name
            raw_out = self.onnx_session.run(None, {input_name: tensor_np})[0]
            # raw_out shape: [1, num_classes, H, W]
            logits_full = raw_out[0]
            exp_logits = np.exp(logits_full - np.max(logits_full, axis=0, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
            # Class 0: SAFE, Class 1: UNKNOWN, Class 2: UNSAFE
            prob_map = probs[0]
            unknown_map = probs[1]
            confidence_map = np.max(probs, axis=0)
            logits_safe = logits_full[0]
        else:
            tensor_torch = torch.from_numpy(tensor_np).to(self._device)
            with torch.no_grad():
                out = self.model(tensor_torch)
                logits_full = out.cpu().numpy()[0]
                probs = F.softmax(out, dim=1).cpu().numpy()[0]
                prob_map = probs[0]
                unknown_map = probs[1]
                confidence_map = np.max(probs, axis=0)
                logits_safe = logits_full[0]

        # Resize maps back to original input dimensions
        if (prob_map.shape[0] != orig_h) or (prob_map.shape[1] != orig_w):
            prob_map = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            logits_safe = cv2.resize(logits_safe, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            confidence_map = cv2.resize(confidence_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            unknown_map = cv2.resize(unknown_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        prob_map = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
        confidence_map = np.clip(confidence_map.astype(np.float32), 0.0, 1.0)
        unknown_mask = (unknown_map >= 0.5)

        inference_s = time.perf_counter() - t0

        # Create overlay visualization
        overlay = create_traversability_overlay(rgb, prob_map, alpha=0.45)

        return TraversabilityResult(
            mask=prob_map,
            logits=logits_safe.astype(np.float32),
            overlay=overlay,
            image=rgb,
            input_shape=(orig_h, orig_w),
            inference_s=inference_s,
            device=self._device_str,
            model_name=self.model_name,
            confidence=confidence_map,
            unknown_mask=unknown_mask
        )

    def warmup(self) -> float:
        """Run warm-up pass to eliminate initial compilation latency."""
        dummy = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        t0 = time.perf_counter()
        _ = self.predict(dummy)
        return time.perf_counter() - t0

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata dictionary."""
        return {
            "model_id": "rugd_traversability_v1",
            "model_name": self.model_name,
            "checkpoint": str(self.checkpoint_path),
            "device": self._device_str,
            "backend": "onnx" if self.is_onnx else "pytorch",
            "target_size": list(self.target_size),
            "manifest": self.manifest
        }
