"""CLI tool to ingest our own real-world outdoor video recordings into sequence datasets."""

import argparse
import sys
from pathlib import Path

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.datasets.own_dataset import OwnDataImporter
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("import_own_data")

def main():
    parser = argparse.ArgumentParser(description="Import real-world outdoor video into sequence dataset")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video file (.mp4, .avi)")
    parser.add_argument("--sequence-id", "-s", type=str, required=True, help="Unique sequence ID (e.g. 'seq_own_dirt_01')")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--env", "-e", type=str, default="dirt_trail", help="Environment type (dirt_trail, gravel, grass, rocks, mixed)")
    parser.add_argument("--lighting", "-l", type=str, default="sunlight", help="Lighting condition (sunlight, shadows, overcast, glare)")
    parser.add_argument("--stride", type=int, default=5, help="Extract every N-th frame to prevent near-duplicate frame redundancy")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to extract")
    parser.add_argument("--output-dir", type=str, default="data/processed/sequences/own_data", help="Output sequence directory")
    args = parser.parse_args()

    importer = OwnDataImporter(output_root=args.output_dir)
    samples = importer.import_video_sequence(
        video_path=args.input,
        sequence_id=args.sequence_id,
        split=args.split,
        environment=args.env,
        lighting=args.lighting,
        sample_every_n_frames=args.stride,
        max_frames=args.max_frames
    )

    print(f"\nSuccessfully imported sequence '{args.sequence_id}':")
    print(f"Total Keyframes Extracted : {len(samples)}")
    print(f"Assigned Split            : {args.split}")
    print(f"Environment Type          : {args.env}")
    print(f"Lighting Condition        : {args.lighting}")
    print(f"Location                  : {Path(args.output_dir) / args.sequence_id}")

if __name__ == "__main__":
    main()
