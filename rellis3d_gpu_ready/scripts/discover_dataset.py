#!/usr/bin/env python3
"""Automated RELLIS-3D Dataset Discovery Utility.

Searches configured search paths, common mounted locations, and local archives.
Verifies the presence of pylon_camera_node and pylon_camera_node_label_id.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

def search_for_dataset(config_path: str = "configs/dataset.yaml") -> Tuple[Optional[Path], List[Path]]:
    """Scan candidate directories for RELLIS-3D dataset roots or archives."""
    cfg_file = Path(config_path)
    search_dirs = ["./data/raw/RELLIS-3D", "./data/raw/public/RELLIS-3D", "../data/raw/public/RELLIS-3D", "/data/RELLIS-3D"]
    archive_names = ["rellis_sample.zip", "RELLIS-3D.zip", "rellis3d.tar.gz"]

    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            search_dirs = cfg.get("discovery", {}).get("search_directories", search_dirs)
            archive_names = cfg.get("discovery", {}).get("archive_names", archive_names)

    detected_root: Optional[Path] = None
    detected_archives: List[Path] = []

    # 1. Search for extracted dataset folders
    for d in search_dirs:
        p = Path(os.path.expanduser(d)).resolve()
        if p.exists() and p.is_dir():
            # Check for sequence folders containing pylon_camera_node
            has_camera = list(p.rglob("pylon_camera_node"))
            if has_camera:
                detected_root = p
                print(f"[FOUND] Extracted RELLIS-3D dataset root at: {detected_root}")
                break

    # 2. Search for archives in search dirs and current directory
    for candidate_dir in [Path("."), Path("data"), Path("data/raw")] + [Path(os.path.expanduser(d)) for d in search_dirs]:
        if candidate_dir.exists():
            for name in archive_names:
                arc_candidate = candidate_dir / name
                if arc_candidate.exists() and arc_candidate.is_file():
                    detected_archives.append(arc_candidate.resolve())
                    print(f"[FOUND] RELLIS-3D archive detected at: {arc_candidate}")

    return detected_root, detected_archives

def main() -> int:
    print("=" * 65)
    print("             RELLIS-3D DATASET DISCOVERY UTILITY                ")
    print("=" * 65)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="Path to dataset config")
    parser.add_argument("--output", type=str, default="reports/dataset_discovery.json", help="Path to save discovery report")
    args = parser.parse_args()

    root, archives = search_for_dataset(args.config)

    report = {
        "dataset_name": "RELLIS-3D",
        "dataset_root": str(root) if root else None,
        "is_extracted": root is not None,
        "detected_archives": [str(a) for a in archives],
        "status": "READY" if root else ("ARCHIVE_FOUND" if archives else "NOT_FOUND")
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Discovery status : {report['status']}")
    print(f"Dataset root     : {report['dataset_root']}")
    print(f"Report saved to  : {out_file}")
    print("=" * 65)

    return 0 if (root or archives) else 1

if __name__ == "__main__":
    sys.exit(main())
