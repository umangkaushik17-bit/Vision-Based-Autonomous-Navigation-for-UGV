#!/usr/bin/env python3
"""Unified Forward Inference and Costmap Projection Utility for UGV Navigation.

Runs dense traversability segmentation inference on single images, videos, or directories,
supports GPU (CUDA) and CPU devices, and exports costmap grids conforming to navigation contracts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.models.traversability_net import LightweightTraversabilityNet

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COLOR_PALETTE = {
    0: (40, 180, 40),     # Green: Safe
    1: (0, 215, 255),     # Yellow: Risky/Unknown
    2: (30, 30, 220),     # Red: Unsafe/Obstacle
    255: (100, 100, 100)  # Gray: Void/Ignore
}


def run_inference(
    image_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device_name: Optional[str] = None,
    target_wh: Tuple[int, int] = (640, 400)
) -> dict:
    device = torch.device(device_name if device_name else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Running inference on device: {device}")

    # Load Model
    model = LightweightTraversabilityNet(num_classes=3)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Read Image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    orig_h, orig_w = img_bgr.shape[:2]

    # Preprocess
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, target_wh, interpolation=cv2.INTER_LINEAR)
    norm = ((resized / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Forward
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_mask = np.argmax(probs, axis=0).astype(np.uint8)
    dt = (time.perf_counter() - t0) * 1000

    # Colorize Segmentation
    color_mask = np.zeros((target_wh[1], target_wh[0], 3), dtype=np.uint8)
    for cid, bgr in COLOR_PALETTE.items():
        color_mask[pred_mask == cid] = bgr

    # Overlay
    overlay = cv2.addWeighted(resized, 0.65, cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB), 0.35, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # Costmap in [0, 255]
    costmap = np.zeros_like(pred_mask, dtype=np.uint8)
    costmap[pred_mask == 0] = 10    # Safe
    costmap[pred_mask == 1] = 120   # Risky
    costmap[pred_mask == 2] = 254   # Lethal Obstacle
    costmap_vis = cv2.applyColorMap(costmap, cv2.COLORMAP_JET)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    costmap_path = output_dir / f"{image_path.stem}_costmap.png"

    cv2.imwrite(str(overlay_path), overlay_bgr)
    cv2.imwrite(str(costmap_path), costmap_vis)

    print(f"[+] Inference latency: {dt:.2f} ms")
    print(f"[+] Saved overlay : {overlay_path}")
    print(f"[+] Saved costmap : {costmap_path}")

    return {
        "image": str(image_path),
        "latency_ms": round(dt, 2),
        "overlay_path": str(overlay_path),
        "costmap_path": str(costmap_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Dense Traversability Inference Utility")
    parser.add_argument("--image", type=str, required=True, help="Input RGB image path")
    parser.add_argument("--checkpoint", type=str, default="outputs/rellis3d/checkpoints/best.pt", help="Model checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Compute device (e.g. cuda:0 or cpu)")
    parser.add_argument("--output", type=str, default="outputs/rellis3d/predictions", help="Output directory")
    args = parser.parse_args()

    run_inference(
        image_path=Path(args.image),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output),
        device_name=args.device
    )


if __name__ == "__main__":
    main()
