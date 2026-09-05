"""CLI script to load, preprocess, and process static images."""

import argparse
import sys
import time
from pathlib import Path

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.core.config import load_vision_config
from ai.core.frame_source import ImageSource
from ai.core.preprocessing import Preprocessor
from ai.core.processor import create_processor
from ai.core.timing import FrameTimer
from ai.io.frame_writer import save_image
from ai.visualization.display import create_side_by_side, draw_hud_overlay, display_frame
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("process_image")

def main():
    parser = argparse.ArgumentParser(description="Process static image through UGV vision pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output image save path")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to vision_config.yaml")
    parser.add_argument("--processor", "-p", type=str, default=None, help="Processor algorithm ('canny_edge', 'grayscale')")
    parser.add_argument("--width", type=int, default=None, help="Override target width")
    parser.add_argument("--height", type=int, default=None, help="Override target height")
    parser.add_argument("--display", "-d", action="store_true", help="Display output in OpenCV window")
    args = parser.parse_args()

    # Load configuration
    cfg = load_vision_config(args.config)
    if args.processor:
        cfg.processing.baseline_processor = args.processor
    if args.width:
        cfg.preprocessing.resize.target_width = args.width
    if args.height:
        cfg.preprocessing.resize.target_height = args.height

    # Initialize pipeline
    source = ImageSource(args.input)
    preprocessor = Preprocessor(cfg.preprocessing)
    processor = create_processor(cfg.processing)
    timer = FrameTimer()

    timer.start_frame()
    source.open()
    frame = source.read()
    if frame is None:
        logger.error("Failed to read image frame from source.")
        sys.exit(1)

    timer.start_preprocess()
    processed_frame = preprocessor.process(frame)
    timer.stop_preprocess()

    timer.start_process()
    visual_output = processor.process(processed_frame)
    timer.stop_process()
    total_ms = timer.stop_frame()

    fps_instant = 1000.0 / total_ms if total_ms > 0 else 0.0

    # Compose output view
    if cfg.visualization.side_by_side:
        display_img = create_side_by_side(
            processed_frame.original_frame,
            visual_output,
            label_left="INPUT IMAGE",
            label_right=f"PROCESSED ({cfg.processing.baseline_processor.upper()})"
        )
    else:
        display_img = visual_output

    # Draw HUD overlay
    display_img = draw_hud_overlay(
        display_img,
        frame_idx=0,
        fps=fps_instant,
        latency_ms=total_ms,
        preprocess_ms=timer.preprocess_ms,
        process_ms=timer.process_ms
    )

    orig_res = f"{processed_frame.original_resolution[0]}x{processed_frame.original_resolution[1]}"
    proc_res = f"{processed_frame.preprocessed_resolution[0]}x{processed_frame.preprocessed_resolution[1]}"

    print("\n================= IMAGE PROCESSING RESULTS =================")
    print(f"Input File          : {args.input}")
    print(f"Source Resolution   : {orig_res}")
    print(f"Target Resolution   : {proc_res}")
    print(f"Preprocess Latency  : {timer.preprocess_ms:.2f} ms")
    print(f"Processing Latency  : {timer.process_ms:.2f} ms")
    print(f"Total Frame Latency : {total_ms:.2f} ms")
    print(f"Instantaneous FPS   : {fps_instant:.1f} FPS")
    print("============================================================")

    if args.output:
        save_image(display_img, args.output)
        print(f"Output saved to: {args.output}")

    if args.display:
        print("Displaying output window. Press any key to exit.")
        display_frame("UGV Vision Pipeline - Image", display_img, delay_ms=0)

    source.release()

if __name__ == "__main__":
    main()
