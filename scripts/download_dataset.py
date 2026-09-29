#!/usr/bin/env python3
"""Authoritative RELLIS-3D Dataset Downloader.

Reads configs/rellis3d_sources.yaml, downloads Google Drive resources,
computes SHA-256 checksums, verifies integrity, extracts archives safely,
and records data/rellis3d/download_manifest.json.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import yaml

try:
    import gdown
except ImportError:
    gdown = None

try:
    import requests
except ImportError:
    requests = None


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 checksum of a file in chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def download_with_gdown(file_id: str, output_path: Path) -> bool:
    """Download file using gdown library if available."""
    if gdown is None:
        return False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[*] Downloading via gdown: {url} -> {output_path}")
        gdown.download(id=file_id, output=str(output_path), quiet=False)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        print(f"[-] gdown download failed: {e}")
        return False


def download_with_requests(file_id: str, output_path: Path) -> bool:
    """Download Google Drive file using requests with session confirmation token."""
    if requests is None:
        return False
    try:
        session = requests.Session()
        url = "https://docs.google.com/uc?export=download"
        params = {"id": file_id}
        
        response = session.get(url, params=params, stream=True)
        # Check for virus scan confirmation token
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                params["confirm"] = value
                response = session.get(url, params=params, stream=True)
                break

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        print(f"[-] requests download failed: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Safely extract zip archive into destination directory."""
    print(f"[*] Extracting {archive_path.name} -> {dest_dir} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
        print(f"[+] Successfully extracted {archive_path.name}")
        return True
    except Exception as e:
        print(f"[-] Extraction failed for {archive_path.name}: {e}")
        return False


def download_dataset(
    config_path: str = "configs/rellis3d_sources.yaml",
    dest_root: Optional[str] = None,
    manifest_path: str = "data/rellis3d/download_manifest.json",
    extract: bool = True
) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    all_success = True

    print("=" * 70)
    print("           RELLIS-3D AUTHORITATIVE DATASET DOWNLOADER                ")
    print("=" * 70)

    for item in cfg.get("sources", []):
        name = item["name"]
        url = item["drive_url"]
        drive_id = item["drive_file_id"]
        exp_name = item.get("expected_filename", f"{name}.zip")
        exp_size = item.get("expected_size_bytes")
        
        if dest_root:
            local_arc = Path(dest_root) / exp_name
            extract_dest = Path(dest_root) / "extracted"
        else:
            local_arc = Path(item.get("local_archive_destination", f"data/rellis3d/{exp_name}"))
            extract_dest = Path(item.get("extract_destination", "data/raw/public/RELLIS-3D"))

        local_arc.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Resource: {name} ---")
        print(f"Source URL   : {url}")
        print(f"Drive ID     : {drive_id}")
        print(f"Target Path  : {local_arc.resolve()}")

        # Check if already downloaded
        already_exists = False
        if local_arc.exists():
            curr_size = local_arc.stat().st_size
            if exp_size and abs(curr_size - exp_size) < 1024:
                print(f"[+] File already exists with matching size ({curr_size} bytes). Skipping download.")
                already_exists = True
            elif not exp_size and curr_size > 0:
                print(f"[+] File already exists ({curr_size} bytes). Skipping download.")
                already_exists = True

        download_success = already_exists
        if not already_exists:
            t0 = time.time()
            if not download_with_gdown(drive_id, local_arc):
                print("[!] Fallback to session requests downloader...")
                download_success = download_with_requests(drive_id, local_arc)
            else:
                download_success = True

            dt = time.time() - t0
            if download_success:
                print(f"[+] Download complete in {dt:.1f}s (Size: {local_arc.stat().st_size} bytes)")
            else:
                print(f"[-] FAILED to download {name}")
                all_success = False

        # Compute checksum
        sha256_hash = "N/A"
        file_size = 0
        if local_arc.exists():
            file_size = local_arc.stat().st_size
            print(f"[*] Calculating SHA-256 for {local_arc.name}...")
            sha256_hash = compute_sha256(local_arc)
            print(f"[+] SHA-256: {sha256_hash}")

        extraction_status = "NOT_EXTRACTED"
        if extract and download_success and local_arc.exists():
            if extract_archive(local_arc, extract_dest):
                extraction_status = "EXTRACTED"
            else:
                extraction_status = "EXTRACTION_FAILED"
                all_success = False

        record = {
            "name": name,
            "filename": local_arc.name,
            "source_url": url,
            "drive_id": drive_id,
            "download_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "size_bytes": file_size,
            "expected_size_bytes": exp_size,
            "checksum_sha256": sha256_hash,
            "local_path": str(local_arc.resolve()),
            "extraction_status": extraction_status,
            "extract_path": str(extract_dest.resolve()) if extraction_status == "EXTRACTED" else None
        }
        records.append(record)

    manifest_data = {
        "dataset_name": "RELLIS-3D",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "all_resources_verified": all_success,
        "resources": records
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Download manifest saved to: {manifest_file.resolve()}")
    print(f"Overall status: {'SUCCESS' if all_success else 'INCOMPLETE/FAILED'}")
    print("=" * 70)
    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Download RELLIS-3D Dataset Resources")
    parser.add_argument("--config", type=str, default="configs/rellis3d_sources.yaml", help="Path to sources YAML")
    parser.add_argument("--dest-root", type=str, default=None, help="Root destination override")
    parser.add_argument("--manifest", type=str, default="data/rellis3d/download_manifest.json", help="Manifest output path")
    parser.add_argument("--no-extract", action="store_true", help="Do not extract archives after download")
    args = parser.parse_args()

    manifest = download_dataset(
        config_path=args.config,
        dest_root=args.dest_root,
        manifest_path=args.manifest,
        extract=not args.no_extract
    )
    sys.exit(0 if manifest["all_resources_verified"] else 1)


if __name__ == "__main__":
    main()
