"""Assembles and validates deployable packages for external GPU execution.

Supports both:
1. Split Two-Zip Architecture (Recommended):
   - gpu_code.zip: Code, models, configs, scripts, auto_run (~90 KB) - instantaneous upload
   - gpu_data.zip: Manifests and processed dataset sequences (~1.25 GB) - upload once to Drive/Colab
2. Single-Zip Architecture:
   - gpu_package.zip: Combined archive containing both code and datasets
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Optional

CODE_PATTERNS = [
    "src",
    "ai",
    "configs",
    "scripts",
    "gpu_package",
    "pyproject.toml"
]

DATA_PATTERNS = [
    "data/manifests",
    "data/processed/sequences"
]

EXCLUDE_EXTS = {".pyc", ".git", ".log"}
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git", "outputs"}


def _zip_items(
    zip_path: Path,
    patterns: List[str],
    root_dir: Path,
    compress: bool = True
) -> int:
    """Pack items matching patterns into a ZIP file."""
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    total_files = 0
    with zipfile.ZipFile(zip_path, "w", compression) as zf:
        for item in patterns:
            item_path = root_dir / item
            if not item_path.exists():
                continue

            if item_path.is_file():
                zf.write(item_path, arcname=item_path.relative_to(root_dir))
                total_files += 1
            elif item_path.is_dir():
                for f in item_path.rglob("*"):
                    if f.is_file():
                        if any(part in EXCLUDE_DIRS for part in f.parts):
                            continue
                        if f.suffix in EXCLUDE_EXTS:
                            continue
                        zf.write(f, arcname=f.relative_to(root_dir))
                        total_files += 1
    return total_files


def build_code_package(output_zip: str = "gpu_code.zip") -> str:
    """Bundle only source code, configs, and orchestrators into a lightweight zip (< 100 KB)."""
    root_dir = Path(__file__).resolve().parent.parent
    zip_path = root_dir / output_zip
    t0 = time.time()
    print(f"\n[PACKAGING] Compiling lightweight CODE package -> {zip_path.name}...")
    total_files = _zip_items(zip_path, CODE_PATTERNS, root_dir, compress=True)
    size_kb = zip_path.stat().st_size / 1024
    print(f"[OK] Successfully built {zip_path.name}: {total_files} files ({size_kb:.2f} KB) in {time.time() - t0:.2f}s")
    return str(zip_path)


def build_data_package(output_zip: str = "gpu_data.zip", include_rugd: bool = False) -> str:
    """Bundle manifests and processed dataset sequences into a standalone database zip."""
    root_dir = Path(__file__).resolve().parent.parent
    zip_path = root_dir / output_zip
    patterns = list(DATA_PATTERNS)
    if include_rugd:
        patterns.append("data/raw/public")

    t0 = time.time()
    print(f"\n[PACKAGING] Compiling DATABASE package -> {zip_path.name} (Include RUGD: {include_rugd})...")
    # For large datasets with PNG/JSON, ZIP_STORED avoids recompressing already-compressed PNGs
    total_files = _zip_items(zip_path, patterns, root_dir, compress=False)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Successfully built {zip_path.name}: {total_files} files ({size_mb:.2f} MB) in {time.time() - t0:.2f}s")
    return str(zip_path)


def build_gpu_package(output_zip: str = "gpu_package.zip", include_rugd: bool = False) -> str:
    """Bundle both code and datasets into a single combined zip."""
    root_dir = Path(__file__).resolve().parent.parent
    zip_path = root_dir / output_zip
    combined_patterns = CODE_PATTERNS + DATA_PATTERNS
    if include_rugd:
        combined_patterns.append("data/raw/public")

    t0 = time.time()
    print(f"\n[PACKAGING] Compiling combined package -> {zip_path.name}...")
    total_files = _zip_items(zip_path, combined_patterns, root_dir, compress=True)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Successfully built {zip_path.name}: {total_files} files ({size_mb:.2f} MB) in {time.time() - t0:.2f}s")
    return str(zip_path)


def main():
    parser = argparse.ArgumentParser(description="Build external GPU execution zip packages")
    parser.add_argument("--code-only", action="store_true", help="Build only gpu_code.zip")
    parser.add_argument("--data-only", action="store_true", help="Build only gpu_data.zip")
    parser.add_argument("--combined-only", action="store_true", help="Build only combined gpu_package.zip")
    parser.add_argument("--include-rugd", action="store_true", help="Include full 5GB RUGD folder inside the zip")
    args = parser.parse_args()

    if args.code_only:
        build_code_package("gpu_code.zip")
    elif args.data_only:
        build_data_package("gpu_data.zip", include_rugd=args.include_rugd)
    elif args.combined_only:
        build_gpu_package("gpu_package.zip", include_rugd=args.include_rugd)
    else:
        # Default: Build both split packages (code + data)
        build_code_package("gpu_code.zip")
        build_data_package("gpu_data.zip", include_rugd=args.include_rugd)


if __name__ == "__main__":
    main()
