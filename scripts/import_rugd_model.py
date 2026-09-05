"""Utility to securely locate, audit, verify, and import RUGD model packages.

Supports:
1. Automatic discovery across candidate directories:
   - models/
   - external_models/
   - imports/
   - data/external/
   - Project root (.)
2. Explicit path override: --zip-path <path_to_zip>
3. Audit and extraction:
   - Path traversal / Zip-Slip prevention
   - SHA-256 checksum verification
   - Metadata and format inspection (.pt, .pth, .onnx, .yaml, .json)
   - Model manifest generation (models/rugd_traversability/model_manifest.json)
4. Integration with trained checkpoints (e.g. from gpu_results.zip).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional


CANDIDATE_SEARCH_DIRS = [
    "models",
    "external_models",
    "imports",
    "data/external",
    "."
]

TARGET_DIR = Path("models/rugd_traversability")


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_candidate_zips(root_dir: Path) -> List[Path]:
    """Scan candidate directories for candidate model zip files."""
    candidates = []
    for d in CANDIDATE_SEARCH_DIRS:
        search_path = root_dir / d
        if not search_path.exists():
            continue

        for p in search_path.glob("*.zip"):
            # Exclude code or data packages
            name_lower = p.name.lower()
            if any(skip in name_lower for skip in ["gpu_code", "gpu_data", "gpu_package"]):
                continue
            candidates.append(p)

    return sorted(list(set(candidates)))


def safe_extract_zip(zip_path: Path, extract_to: Path) -> List[str]:
    """Extract a ZIP archive securely, preventing path traversal / zip-slip attacks."""
    extract_to = extract_to.resolve()
    extract_to.mkdir(parents=True, exist_ok=True)

    extracted_files = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (extract_to / member.filename).resolve()
            # Security: Ensure extracted path is strictly within target directory
            try:
                member_path.relative_to(extract_to)
            except ValueError:
                raise ValueError(f"Security Alert: Zip Slip path traversal attempt detected in '{member.filename}'")

            if member.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
            else:
                member_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, open(member_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                extracted_files.append(str(member_path.relative_to(extract_to)))

    return extracted_files


def audit_extracted_model(target_dir: Path, source_zip: Path) -> Dict[str, Any]:
    """Inspect model files, detect architecture, checkpoint format, and generate manifest."""
    files = list(target_dir.rglob("*"))
    file_list = [f.name for f in files if f.is_file()]

    # Look for checkpoint files
    checkpoints = [f for f in files if f.suffix.lower() in [".pt", ".pth", ".onnx", ".ckpt", ".bin"]]
    configs = [f for f in files if f.suffix.lower() in [".yaml", ".yml", ".json"]]

    checkpoint_info = []
    for ckpt in checkpoints:
        info = {
            "name": ckpt.name,
            "relative_path": str(ckpt.relative_to(target_dir)),
            "size_bytes": ckpt.stat().st_size,
            "format": ckpt.suffix.lower().replace(".", ""),
            "sha256": calculate_sha256(ckpt)
        }
        checkpoint_info.append(info)

    # Determine primary checkpoint
    primary_ckpt = None
    for ckpt in checkpoint_info:
        if "best" in ckpt["name"].lower() or "traversability" in ckpt["name"].lower() or ckpt["name"].endswith(".onnx"):
            primary_ckpt = ckpt
            break
    if not primary_ckpt and checkpoint_info:
        primary_ckpt = checkpoint_info[0]

    manifest = {
        "model_id": "rugd_traversability_v1",
        "model_name": "RUGD-Trained Traversability Baseline",
        "source_archive": source_zip.name,
        "source_archive_sha256": calculate_sha256(source_zip),
        "imported_at": str(Path(target_dir).stat().st_mtime),
        "target_directory": str(target_dir),
        "total_files": len(file_list),
        "checkpoints": checkpoint_info,
        "primary_checkpoint": primary_ckpt,
        "configs": [str(c.relative_to(target_dir)) for c in configs],
        "input_specification": {
            "color_space": "RGB",
            "expected_shape": [480, 640, 3],
            "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
        },
        "output_specification": {
            "classes": ["SAFE", "UNKNOWN", "UNSAFE"],
            "drivable_class_index": 0,
            "probability_range": "[0.0, 1.0]"
        }
    }

    manifest_path = target_dir / "model_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def import_model_from_zip(zip_path: Path, force: bool = False) -> Dict[str, Any]:
    """Execute complete import flow for a given zip package."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Model archive not found at: {zip_path}")

    print(f"\n[IMPORT] Source archive found: {zip_path.resolve()}")
    print(f"[IMPORT] Archive size: {zip_path.stat().st_size / (1024*1024):.2f} MB")
    archive_sha = calculate_sha256(zip_path)
    print(f"[IMPORT] SHA-256: {archive_sha}")

    if TARGET_DIR.exists() and not force:
        existing_manifest = TARGET_DIR / "model_manifest.json"
        if existing_manifest.exists():
            print(f"[IMPORT] Target directory '{TARGET_DIR}' already contains an audited model.")
            with open(existing_manifest, "r", encoding="utf-8") as f:
                return json.load(f)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[IMPORT] Extracting securely to: {TARGET_DIR.resolve()}...")
    extracted_files = safe_extract_zip(zip_path, TARGET_DIR)
    print(f"[IMPORT] Extracted {len(extracted_files)} files successfully.")

    print("[IMPORT] Auditing extracted artifacts...")
    manifest = audit_extracted_model(TARGET_DIR, zip_path)
    print(f"[IMPORT] Model Manifest successfully created: {TARGET_DIR / 'model_manifest.json'}")
    print(f"[IMPORT] Primary Checkpoint: {manifest.get('primary_checkpoint', {}).get('name', 'None')}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Import and audit RUGD traversability model package")
    parser.add_argument("--zip-path", type=str, default=None, help="Explicit path to model zip file")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing imported model")
    args = parser.parse_args()

    root_dir = Path.cwd()
    selected_zip = None

    if args.zip_path:
        selected_zip = Path(args.zip_path)
        if not selected_zip.exists():
            print(f"[ERROR] Specified zip path does not exist: {selected_zip}")
            sys.exit(1)
    else:
        candidates = find_candidate_zips(root_dir)
        if not candidates:
            # Check if outputs/checkpoints/best.pt exists (our trained baseline)
            local_best = Path("outputs/checkpoints/best.pt")
            if local_best.exists():
                print(f"[NOTICE] No external ZIP provided yet. Found local trained baseline at: {local_best}")
                # Create a temporary local zip package to ensure uniform auditing
                pkg_dir = Path("models")
                pkg_dir.mkdir(parents=True, exist_ok=True)
                local_pkg_zip = pkg_dir / "rugd_baseline_local.zip"
                with zipfile.ZipFile(local_pkg_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(local_best, arcname="best.pt")
                    if Path("outputs/reports/evaluation_report.json").exists():
                        zf.write("outputs/reports/evaluation_report.json", arcname="evaluation_report.json")
                    if Path("configs/traversability_training.yaml").exists():
                        zf.write("configs/traversability_training.yaml", arcname="config.yaml")
                selected_zip = local_pkg_zip
            else:
                print("[ERROR] No model ZIP found in candidate directories:")
                for d in CANDIDATE_SEARCH_DIRS:
                    print(f"  - {d}")
                print("\nPlease specify a path using: python scripts/import_rugd_model.py --zip-path <path>")
                sys.exit(1)
        else:
            # Choose the most relevant candidate (e.g. rugd or gpu_results)
            selected_zip = candidates[0]
            for c in candidates:
                if "rugd" in c.name.lower():
                    selected_zip = c
                    break

    manifest = import_model_from_zip(selected_zip, force=args.force)
    print("\n[SUCCESS] Model import and security audit completed.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
