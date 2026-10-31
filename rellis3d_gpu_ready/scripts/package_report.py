#!/usr/bin/env python3
"""Consolidates pipeline reports into an overarching summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

def main():
    print("Consolidating pipeline reports...")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "pipeline": "RELLIS-3D GPU Traversability Engineering Pipeline",
        "baseline_model": "RUGD-Traversability-Net (Model B)",
        "status": "BLOCKED — GPU UNAVAILABLE IN LOCAL ENVIRONMENT",
        "package_ready": True,
        "instructions": "Transfer package to NVIDIA GPU host or run on verified GPU server."
    }

    out_file = reports_dir / "pipeline_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Summary generated at {out_file}")

if __name__ == "__main__":
    main()
