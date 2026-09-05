"""Generate standard Phase 1 test assets (sample.jpg and sample.mp4)."""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_synthetic_outdoor_data import generate_synthetic_frame

def main():
    root = Path(__file__).resolve().parent.parent
    test_dir = root / "data" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    width = 640
    height = 480
    fps = 30.0
    num_frames = 45 # 1.5 seconds of video

    print(f"[Test Assets] Generating sample image and video in {test_dir}...")

    # 1. Generate sample image
    sample_rgb, _, _, _ = generate_synthetic_frame(step=0, total_steps=num_frames, width=width, height=height)
    img_path = test_dir / "sample.jpg"
    cv2.imwrite(str(img_path), sample_rgb)
    print(f"[Test Assets] Created: {img_path} ({width}x{height})")

    # 2. Generate sample video
    vid_path = test_dir / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (width, height))
    
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vid_path = test_dir / "sample.avi"
        writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (width, height))

    for step in range(num_frames):
        frame_rgb, _, _, _ = generate_synthetic_frame(step=step, total_steps=num_frames, width=width, height=height)
        writer.write(frame_rgb)

    writer.release()
    print(f"[Test Assets] Created: {vid_path} ({num_frames} frames @ {fps} FPS)")

if __name__ == "__main__":
    main()
