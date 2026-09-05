"""CLI script to stream and process frames from a live webcam or USB camera."""

import argparse
import sys
import time
from pathlib import Path

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.core.config import load_vision_config
from ai.core.frame_source import CameraSource
from ai.core.preprocessing import Preprocessor
from ai.core.processor import create_processor
from ai.core.timing import FrameTimer, FPSMeter
from ai.io.frame_writer import VideoWriter
from ai.visualization.display import create_side_by_side, draw_hud_overlay, display_frame
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("process_camera")

def main():
    parser = argparse.ArgumentParser(description="Process live camera stream through UGV vision pipeline")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Target capture width")
    parser.add_argument("--height", type=int, default=480, help="Target capture height")
    parser.add_argument("--fps", type=int, default=30, help="Target capture FPS")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output video save path")
    parser.add_argument("--config", type=str, default=None, help="Path to vision_config.yaml")
    parser.add_argument("--processor", "-p", type=str, default=None, help="Vision processor algorithm")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window display")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Stop after N frames (useful for headless verification)")
    args = parser.parse_args()

    cfg = load_vision_config(args.config)
    if args.processor:
        cfg.processing.baseline_processor = args.processor

    # Initialize CameraSource with graceful fallback
    source = CameraSource(
        camera_index=args.camera,
        target_width=args.width,
        target_height=args.height,
        target_fps=args.fps
    )

    try:
        source.open()
    except RuntimeError as err:
        print("\n" + "=" * 60)
        print("[CAMERA NOT DETECTED]")
        print(f"Could not connect to physical camera device index {args.camera}.")
        print("In a software-only / remote simulation environment, run with a recorded video:")
        print("   python scripts/process_video.py --input data/test/sample.mp4")
        print("=" * 60 + "\n")
        sys.exit(0) # Exit cleanly with user guidance

    preprocessor = Preprocessor(cfg.preprocessing)
    processor = create_processor(cfg.processing)
    fps_meter = FPSMeter(window_size=cfg.timing.fps_window_size, source_fps=source.fps)
    timer = FrameTimer()

    # Optional video writer
    video_writer = None
    target_w = cfg.preprocessing.resize.target_width
    target_h = cfg.preprocessing.resize.target_height
    composite_w = (target_w * 2) if cfg.visualization.side_by_side else target_w

    if args.output:
        video_writer = VideoWriter(
            output_path=args.output,
            fps=source.fps,
            width=composite_w,
            height=target_h
        )
        video_writer.open()

    print(f"\nStreaming live camera index {args.camera}...")
    print(f"Native resolution: {source.width}x{source.height} @ {source.fps:.1f} FPS")
    print("Press 'q' or 'ESC' in the display window to exit.\n")

    frame_count = 0
    show_window = not args.no_display

    try:
        for vision_frame in source:
            timer.start_frame()

            # Preprocessing
            timer.start_preprocess()
            processed = preprocessor.process(vision_frame)
            timer.stop_preprocess()

            # Vision Processing
            timer.start_process()
            output_vis = processor.process(processed)
            timer.stop_process()

            timer.stop_frame()
            timing_snap = fps_meter.update(timer)

            # Visualization
            if cfg.visualization.side_by_side:
                display_img = create_side_by_side(
                    processed.original_frame,
                    output_vis,
                    label_left=f"CAMERA [{source.width}x{source.height}]",
                    label_right=f"{cfg.processing.baseline_processor.upper()} [{target_w}x{target_h}]"
                )
            else:
                display_img = output_vis

            display_img = draw_hud_overlay(
                display_img,
                frame_idx=timing_snap.frame_index,
                fps=fps_meter.moving_average_fps,
                latency_ms=timing_snap.total_frame_ms,
                preprocess_ms=timing_snap.preprocess_ms,
                process_ms=timing_snap.process_ms
            )

            if video_writer:
                video_writer.write_frame(display_img)

            if show_window:
                key = display_frame(f"UGV Live Camera ({args.camera})", display_img, delay_ms=1)
                if key == 27 or key == ord("q"):
                    print("\nLive camera stream stopped by user.")
                    break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                print(f"Reached max-frames limit ({args.max_frames}).")
                break

            if frame_count % 30 == 0:
                print(f"Camera Stream: Frame {frame_count:04d} | Live FPS: {fps_meter.moving_average_fps:5.1f}")

    finally:
        source.release()
        if video_writer:
            video_writer.close()
        if show_window:
            import cv2
            cv2.destroyAllWindows()

    report = fps_meter.generate_report()
    print("\n" + report.summary_string())

if __name__ == "__main__":
    main()
