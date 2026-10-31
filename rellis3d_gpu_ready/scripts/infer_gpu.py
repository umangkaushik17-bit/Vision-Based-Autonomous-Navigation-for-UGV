#!/usr/bin/env python3
"""Unified Inference and Costmap Projection Utility.

Performs forward inference on RGB camera images, supports CPU and GPU backends,
and formats the output into traversability probability maps for costmap integration.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.traversability_net import LightweightTraversabilityNet

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image_rgb: np.ndarray, target_size: Tuple[int, int] = (480, 640)) -> Tuple[torch.Tensor, Tuple[int, int]]:
    orig_h, orig_w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, (orig_h, orig_w)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Input RGB image path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rugd_baseline_best.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--output-dir", type=str, default="outputs/inference", help="Directory to save overlay and costmap")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Initializing inference on device: {device}")

    # Load Model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = LightweightTraversabilityNet(num_classes=3)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Load and Preprocess Image
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image path does not exist: {img_path}")
        sys.exit(1)

    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    tensor, (orig_h, orig_w) = preprocess(rgb)
    tensor = tensor.to(device)

    # Run Inference
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    dt = (time.perf_counter() - t0) * 1000

    # Extract maps
    prob_safe = cv2.resize(probs[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    prob_unknown = cv2.resize(probs[1], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    prob_unsafe = cv2.resize(probs[2], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Convert to Costmap Grid Values [0, 255]
    # High cost for unsafe/unknown, low cost for safe
    costmap = np.clip((prob_unsafe * 254.0 + prob_unknown * 128.0), 0, 254).astype(np.uint8)

    # Create visualization overlay
    overlay = rgb.copy()
    overlay[prob_safe >= 0.50] = overlay[prob_safe >= 0.50] * 0.5 + np.array([0, 255, 0]) * 0.5 # Green: Safe
    overlay[prob_unsafe >= 0.50] = overlay[prob_unsafe >= 0.50] * 0.5 + np.array([255, 0, 0]) * 0.5 # Red: Unsafe
    overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"{img_path.stem}_traversability_overlay.jpg"
    costmap_path = out_dir / f"{img_path.stem}_costmap.png"

    cv2.imwrite(str(overlay_path), overlay_bgr)
    cv2.imwrite(str(costmap_path), costmap)

    print("=" * 65)
    print("                 INFERENCE EXECUTION SUCCESS                     ")
    print("=" * 65)
    print(f"Device           : {device}")
    print(f"Latency          : {dt:.2f} ms ({1000/dt:.1f} FPS)")
    print(f"Safe Area Ratio  : {(prob_safe >= 0.50).mean()*100:.1f}%")
    print(f"Hazard Ratio     : {(prob_unsafe >= 0.50).mean()*100:.1f}%")
    print(f"Overlay Saved    : {overlay_path}")
    print(f"Costmap Saved    : {costmap_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
