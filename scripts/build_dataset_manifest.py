"""CLI script to assemble and validate the unified dataset manifest across all sources."""

import argparse
import sys
from pathlib import Path

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.datasets.base import DatasetSample
from ai.datasets.split_manager import SplitManager
from ai.datasets.synthetic_dataset import SyntheticSequenceGenerator
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("build_dataset_manifest")

def build_default_synthetic_sequences(generator: SyntheticSequenceGenerator, manager: SplitManager):
    """Generate balanced synthetic sequences across train, val, and test splits."""
    sequence_specs = [
        # Train Split
        {"seq_id": "syn_seq_01_trail_sun", "split": "train", "env": "trail", "lighting": "sunlight", "frames": 12},
        {"seq_id": "syn_seq_02_trail_shadows", "split": "train", "env": "trail", "lighting": "shadows", "frames": 12},
        {"seq_id": "syn_seq_03_rocks_sun", "split": "train", "env": "rocks", "lighting": "sunlight", "frames": 12},
        # Validation Split (Held-out sequences)
        {"seq_id": "syn_seq_04_grass_overcast", "split": "val", "env": "grass", "lighting": "overcast", "frames": 8},
        {"seq_id": "syn_seq_05_rocks_shadows", "split": "val", "env": "rocks", "lighting": "shadows", "frames": 8},
        # Final Test Split (STRICTLY UNSEEN)
        {"seq_id": "syn_seq_06_mixed_unseen_01", "split": "test", "env": "mixed", "lighting": "glare", "frames": 10},
        {"seq_id": "syn_seq_07_ditch_unseen_02", "split": "test", "env": "rocks", "lighting": "shadows", "frames": 10}
    ]

    print("[Manifest Builder] Generating standardized synthetic sequences...")
    for spec in sequence_specs:
        samples = generator.generate_sequence(
            sequence_id=spec["seq_id"],
            num_frames=spec["frames"],
            split=spec["split"],
            environment=spec["env"],
            lighting=spec["lighting"]
        )
        for s in samples:
            manager.add_sample(s)
        print(f" -> Generated {spec['seq_id']} ({len(samples)} frames) -> Split: {spec['split']}")

def scan_own_data_sequences(own_data_dir: Path, manager: SplitManager):
    """Scan and index all imported own real-world sequences."""
    if not own_data_dir.exists():
        return

    import json
    for seq_folder in own_data_dir.iterdir():
        if seq_folder.is_dir():
            meta_file = seq_folder / "sequence_meta.json"
            rgb_dir = seq_folder / "rgb"
            mask_dir = seq_folder / "masks"
            if meta_file.exists() and rgb_dir.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                seq_id = meta.get("sequence_id", seq_folder.name)
                split = meta.get("split", "train")
                env = meta.get("environment", "dirt_trail")
                lighting = meta.get("lighting", "sunlight")

                image_files = sorted(rgb_dir.glob("*.png"))
                for img in image_files:
                    mask_candidate = mask_dir / img.name
                    sample = DatasetSample(
                        sample_id=img.stem,
                        sequence_id=seq_id,
                        image_path=str(img),
                        mask_path=str(mask_candidate) if mask_candidate.exists() else None,
                        split=split,
                        source_type="own_real",
                        environment=env,
                        lighting=lighting,
                        metadata=meta
                    )
                    manager.add_sample(sample)
                print(f" -> Indexed Own Sequence '{seq_id}' ({len(image_files)} frames) -> Split: {split}")

def scan_public_rugd_sequences(public_dir: Path, manager: SplitManager):
    """Scan and index all extracted RUGD public benchmark sequences."""
    rugd_root = public_dir / "RUGD"
    if not rugd_root.exists():
        rugd_root = public_dir

    if not rugd_root.exists():
        return

    from ai.datasets.public_dataset import RUGDAdapter
    adapter = RUGDAdapter(root_dir=str(rugd_root))

    # Look for sequence directories
    seq_dirs = [d for d in rugd_root.iterdir() if d.is_dir() and d.name not in ("annotations", "images", "cache")]
    if not seq_dirs:
        return

    print(f"[Manifest Builder] Discovered {len(seq_dirs)} RUGD sequence directories in {rugd_root.name}...")
    for idx, s_dir in enumerate(seq_dirs):
        seq_name = s_dir.name
        # Sequence-level split assignment: 70% train, 15% val, 15% test
        if idx % 5 in (0, 1, 2):
            split = "train"
        elif idx % 5 == 3:
            split = "val"
        else:
            split = "test"

        samples = adapter.parse_sequence(seq_name, split=split)
        for s in samples:
            manager.add_sample(s)
        if samples:
            print(f" -> Indexed RUGD Sequence '{seq_name}' ({len(samples)} frames) -> Split: {split}")

def main():
    parser = argparse.ArgumentParser(description="Assemble unified dataset manifest with zero-leakage enforcement")
    parser.add_argument("--output", "-o", type=str, default="data/manifests/dataset_manifest.json", help="Manifest destination")
    parser.add_argument("--own-data-dir", type=str, default="data/processed/sequences/own_data", help="Own data sequence root")
    parser.add_argument("--public-dir", type=str, default="data/raw/public", help="Public benchmark root")
    parser.add_argument("--synthetic-dir", type=str, default="data/processed/sequences/synthetic", help="Synthetic sequence root")
    args = parser.parse_args()

    manager = SplitManager(args.output)
    syn_gen = SyntheticSequenceGenerator(base_output_dir=args.synthetic_dir)

    # 1. Populate synthetic sequences
    build_default_synthetic_sequences(syn_gen, manager)

    # 2. Index own data sequences
    scan_own_data_sequences(Path(args.own_data_dir), manager)

    # 3. Index public RUGD sequences if present/extracted
    scan_public_rugd_sequences(Path(args.public_dir), manager)

    # 4. Validate zero leakage and save manifest
    manifest_path = manager.save_manifest()
    print(f"\n[Manifest Builder] Successfully compiled dataset manifest: {manifest_path}")

if __name__ == "__main__":
    main()
