#!/usr/bin/env python3
"""Safe Archive Extraction Utility for RELLIS-3D.

Safely extracts zip and tar.gz archives avoiding path traversal attacks.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import zipfile
from pathlib import Path

def is_safe_path(base_dir: Path, path: Path) -> bool:
    """Ensure extracted file path stays strictly within the target directory."""
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        return str(resolved_path).startswith(str(resolved_base))
    except Exception:
        return False

def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract zip or tar.gz file cleanly into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()

    print(f"Extracting {archive_path.name} -> {dest_dir}...")

    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.namelist():
                target_file = dest_dir / member
                if not is_safe_path(dest_dir, target_file):
                    print(f"[SECURITY] Skipped unsafe zip member: {member}")
                    continue
                zf.extract(member, dest_dir)
        print("[OK] Zip extraction complete.")
        return True

    elif suffix in [".tar", ".gz", ".tgz"]:
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                target_file = dest_dir / member.name
                if not is_safe_path(dest_dir, target_file):
                    print(f"[SECURITY] Skipped unsafe tar member: {member.name}")
                    continue
                tf.extract(member, dest_dir)
        print("[OK] Tar extraction complete.")
        return True

    else:
        print(f"[ERROR] Unsupported archive format: {suffix}")
        return False

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=str, required=True, help="Path to archive file")
    parser.add_argument("--dest", type=str, default="./data/raw/public/RELLIS-3D", help="Target destination folder")
    args = parser.parse_args()

    arc = Path(args.archive)
    if not arc.exists():
        print(f"[ERROR] Archive not found: {arc}")
        return 1

    success = extract_archive(arc, Path(args.dest))
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
