"""CLI script to stream and process video files through the UGV vision pipeline."""

import argparse
import sys
import time
from pathlib import Path

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.core.config import load_vision_config
from ai.core.frame_source import VideoSource
from ai.core.preprocessing import Preprocessor
from ai.core.processor import create_processor
from ai.core.timing import FrameTimer, FPSMeter
from ai.io.frame_writer import VideoWriter
from ai.visualization.display import create_side_by_side, draw_hud_overlay, display_frame
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("process_video")

def main():
    parser = argparse.ArgumentParser(description="Stream and process video through UGV vision pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video (.mp4, .avi, etc.)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save processed video (.mp4)")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to vision_config.yaml")
    parser.add_argument("--processor", "-p", type=str, default=None, help="Vision processor algorithm")
    parser.add_argument("--width", type=int, default=None, help="Override target frame width")
    parser.add_argument("--height", type=int, default=None, help="Override target frame height")
    parser.add_argument("--display", "-d", action="store_true", help="Display live processing window")
    parser.add_argument("--save", "-s", action="store_true", help="Save output video (uses --output or default)")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Limit number of frames to process")
    args = parser.parse_args()

    # Load configuration
    cfg = load_vision_config(args.config)
    if args.processor:
        cfg.processing.baseline_processor = args.processor
    if args.width:
        cfg.preprocessing.resize.target_width = args.width
    if args.height:
        cfg.preprocessing.resize.target_height = args.height

    # Setup FrameSource
    source = VideoSource(args.input)
    source.open()

    # Setup pipeline components
    preprocessor = Preprocessor(cfg.preprocessing)
    processor = create_processor(cfg.processing)
    fps_meter = FPSMeter(window_size=cfg.timing.fps_window_size, source_fps=source.fps)
    timer = FrameTimer()

    # Setup VideoWriter if saving requested
    output_path = args.output
    if args.save and output_path is None:
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"processed_{Path(args.input).stem}.mp4")

    video_writer = None
    target_w = cfg.preprocessing.resize.target_width
    target_h = cfg.preprocessing.resize.target_height
    composite_w = (target_w * 2) if cfg.visualization.side_by_side else target_w

    if output_path:
        video_writer = VideoWriter(
            output_path=output_path,
            fps=source.fps,
            width=composite_w,
            height=target_h
        )
        video_writer.open()

    print(f"\nProcessing video: {args.input}")
    print(f"Source: {source.width}x{source.height} @ {source.fps:.1f} FPS | Total Frames: {source.total_frames or 'Unknown'}")
    print(f"Target: {target_w}x{target_h} | Side-by-side: {cfg.visualization.side_by_side}\n")

    frame_count = 0
    start_total_time = time.perf_counter()

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

            # Visualization composition
            if cfg.visualization.side_by_side:
                display_img = create_side_by_side(
                    processed.original_frame,
                    output_vis,
                    label_left=f"SOURCE [{source.width}x{source.height}]",
                    label_right=f"{cfg.processing.baseline_processor.upper()} [{target_w}x{target_h}]"
                )
            else:
                display_img = output_vis

            # Overlay HUD
            percentiles = fps_meter.compute_percentiles([95])
            display_img = draw_hud_overlay(
                display_img,
                frame_idx=timing_snap.frame_index,
                fps=fps_meter.moving_average_fps,
                latency_ms=timing_snap.total_frame_ms,
                preprocess_ms=timing_snap.preprocess_ms,
                process_ms=timing_snap.process_ms,
                p95_ms=percentiles.get(95)
            )

            # Write to output video
            if video_writer:
                video_writer.write_frame(display_img)

            # Interactive display
            if args.display:
                key = display_frame(cfg.visualization.window_title, display_img, delay_ms=1)
                if key == 27 or key == ord("q"): # ESC or 'q'
                    print("\nProcessing interrupted by user (Quit).")
                    break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                print(f"Reached maximum requested frames ({args.max_frames}). Stopping.")
                break

            # Progress log every 15 frames
            if frame_count % 15 == 0:
                print(
                    f"Frame {frame_count:04d} | Moving FPS: {fps_meter.moving_average_fps:5.1f} | "
                    f"Latency: {timing_snap.total_frame_ms:5.1f} ms"
                )

    finally:
        source.release()
        if video_writer:
            video_writer.close()
        if args.display:
            import cv2
            cv2.destroyAllWindows()

    # Generate and print performance report
    report = fps_meter.generate_report()
    print("\n" + report.summary_string())

    if output_path:
        print(f"\nProcessed video successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
