"""Standalone traversability inference tool supporting images, recorded videos, and live webcam."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.core.frame_source import ImageSource, VideoSource, CameraSource, VisionFrame
from ai.core.timing import FrameTimer, FPSMeter
from ai.datasets.base import TraversabilityTargetClass
from ai.models.traversability_net import create_traversability_model
from ai.traversability.baseline_classifier import BaselineTerrainClassifier
from ai.traversability.temporal_stabilizer import TemporalStabilizer
from ai.traversability.uncertainty import apply_uncertainty_threshold
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("run_traversability")

# Stylized Visualization Palette
COLOR_MAP = {
    TraversabilityTargetClass.SAFE: (46, 204, 113),      # Emerald Green
    TraversabilityTargetClass.UNKNOWN: (241, 196, 15),   # Sun Yellow / Amber
    TraversabilityTargetClass.UNSAFE: (231, 76, 60),     # Crimson Red
    TraversabilityTargetClass.VOID: (0, 0, 0)            # Black / Unmasked
}

class TraversabilityInferencePipeline:
    """End-to-end inference engine with uncertainty thresholding and temporal smoothing."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_temporal_stabilizer: bool = True,
        confidence_threshold: float = 0.60
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_traversability_model().to(self.device)
        self.model.eval()

        self.has_trained_weights = False
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.has_trained_weights = True
            logger.info(f"Loaded trained checkpoint: {checkpoint_path}")
        else:
            logger.info("Using baseline classical heuristic classifier.")
            self.baseline_classifier = BaselineTerrainClassifier()

        self.stabilizer = TemporalStabilizer() if use_temporal_stabilizer else None
        self.conf_threshold = confidence_threshold

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Run traversability inference on single BGR frame."""
        h_orig, w_orig = frame_bgr.shape[:2]
        t0 = time.perf_counter()

        if self.has_trained_weights:
            # DL Inference
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb_res = cv2.resize(rgb, (640, 480))
            tensor = torch.from_numpy(rgb_res.transpose(2, 0, 1)).float() / 255.0
            tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                preds_t, conf_t, _ = apply_uncertainty_threshold(logits, confidence_threshold=self.conf_threshold)

            pred_mask = preds_t.cpu().numpy().astype(np.uint8)
            pred_mask = cv2.resize(pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        else:
            # Baseline Classifier Fallback
            pred_mask = self.baseline_classifier.predict(frame_bgr)

        # Optional Temporal Smoothing
        if self.stabilizer is not None:
            # Mock one-hot for stabilizer
            one_hot = np.zeros((h_orig, w_orig, 3), dtype=np.float32)
            for c in (0, 1, 2):
                one_hot[:, :, c] = (pred_mask == c).astype(np.float32)
            pred_mask = self.stabilizer.update(one_hot)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Create overlay visualization
        overlay = frame_bgr.copy()
        for cls_id, color in COLOR_MAP.items():
            if cls_id == TraversabilityTargetClass.VOID:
                continue
            mask_c = (pred_mask == cls_id)
            if np.any(mask_c):
                colored_layer = np.full_like(frame_bgr, fill_value=color, dtype=np.uint8)
                alpha = 0.35 if cls_id != TraversabilityTargetClass.UNSAFE else 0.45
                overlay[mask_c] = cv2.addWeighted(frame_bgr, 1.0 - alpha, colored_layer, alpha, 0.0)[mask_c]

        safe_pct = float(np.mean(pred_mask == TraversabilityTargetClass.SAFE)) * 100.0
        return pred_mask, overlay, {"latency_ms": latency_ms, "safe_percent": safe_pct}

def main():
    parser = argparse.ArgumentParser(description="Run Traversability AI inference on video, image, or camera feed")
    parser.add_argument("--input", "-i", type=str, default="data/test/sample.mp4", help="Input media path")
    parser.add_argument("--output", "-o", type=str, default="data/processed/traversability_demo.mp4", help="Output MP4 path")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt", help="Model checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Process limited frames for testing")
    args = parser.parse_args()

    pipeline = TraversabilityInferencePipeline(checkpoint_path=args.checkpoint)
    source = VideoSource(args.input)
    fps_meter = FPSMeter()

    out_p = Path(args.output).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    print(f"\n[INFERENCE] Streaming from {args.input} -> {args.output}...")
    frame_count = 0

    with source:
        for vframe in source:
            timer = FrameTimer()
            timer.start_process()
            pred_mask, overlay, info = pipeline.process_frame(vframe.raw_frame)
            timer.stop_process()
            snapshot = fps_meter.update(timer)

            # Draw HUD
            cv2.putText(overlay, f"UGV Traversability | FPS: {fps_meter.moving_average_fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Latency: {info['latency_ms']:.1f}ms | Safe: {info['safe_percent']:.1f}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 204, 113), 2)

            if writer is None:
                writer = cv2.VideoWriter(str(out_p), fourcc, 30.0, (overlay.shape[1], overlay.shape[0]))
            writer.write(overlay)

            frame_count += 1
            if args.smoke_test and frame_count >= 15:
                break

    if writer:
        writer.release()
    print(f"[INFERENCE COMPLETE] Processed {frame_count} frames. Output: {out_p}")

if __name__ == "__main__":
    main()
