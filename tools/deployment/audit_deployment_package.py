"""Deployment Package Size Audit and Manifest Generator.

Audits Development Repository size against Minimal Production Deployment.
Identifies and quantifies research assets excluded:
  - SAM-TP weights & runtime
  - Python DWA reference files
  - Training datasets and raw imagery
  - Benchmark-only assets and offline logs
  - Temporary virtual environments and development caches
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def get_dir_size(path: Path, exclude_patterns: List[str] = None) -> int:
    """Calculate total size in bytes of a path, skipping excluded patterns."""
    exclude_patterns = exclude_patterns or []
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size

    for root, dirs, files in os.walk(path):
        # Filter directories in-place
        dirs[:] = [
            d for d in dirs
            if not any(pat in (Path(root) / d).as_posix() for pat in exclude_patterns)
        ]
        for f in files:
            file_path = Path(root) / f
            posix_path = file_path.as_posix()
            if any(pat in posix_path for pat in exclude_patterns):
                continue
            try:
                total += file_path.stat().st_size
            except (OSError, FileNotFoundError):
                pass
    return total


def format_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def audit_repository(repo_root: Path) -> Dict[str, Any]:
    print("=" * 70)
    print("AUDITING DEVELOPMENT REPOSITORY VS PRODUCTION DEPLOYMENT")
    print(f"Target Repository: {repo_root}")
    print("=" * 70)

    # 1. Measure Core Development Subdirectories
    tracked_dirs = [
        "ai", "src", "production", "backends", "research_reference",
        "configs", "models", "outputs", "data", "rellis3d_gpu_ready",
        "benchmarks", "reports", "scripts", "simulation", "ros2_ws",
        "tests", "tools", "docs"
    ]

    dir_sizes: Dict[str, int] = {}
    for d in tracked_dirs:
        p = repo_root / d
        if p.exists():
            sz = get_dir_size(p)
            dir_sizes[d] = sz

    # Measure specific files/artifacts
    cpp_simd_dll = repo_root / "ai" / "planning" / "native_cpp" / "dwa_cpp.dll"
    cpp_scalar_dll = repo_root / "ai" / "planning" / "native_cpp" / "dwa_cpp_scalar.dll"
    prod_model = repo_root / "outputs" / "rellis3d" / "checkpoints" / "best.pt"

    simd_dll_size = cpp_simd_dll.stat().st_size if cpp_simd_dll.exists() else 0
    scalar_dll_size = cpp_scalar_dll.stat().st_size if cpp_scalar_dll.exists() else 0
    prod_model_size = prod_model.stat().st_size if prod_model.exists() else 0

    # 2. Excluded Research Assets
    # SAM-TP models / weights if any
    sam_assets_size = 0
    for cand in ["models/sam", "models/sam_tp", "checkpoints/sam"]:
        p = repo_root / cand
        if p.exists():
            sam_assets_size += get_dir_size(p)

    # Datasets
    data_size = dir_sizes.get("data", 0)
    rellis_data_size = 0
    if (repo_root / "rellis3d_gpu_ready").exists():
        # Check subdirs with images/labels
        rellis_data_size = get_dir_size(repo_root / "rellis3d_gpu_ready", exclude_patterns=["checkpoints"])

    # Development caches and virtualenvs
    venv_xpu_size = get_dir_size(repo_root / ".venv_xpu")
    pytest_cache_size = get_dir_size(repo_root / ".pytest_cache")
    artifacts_size = get_dir_size(repo_root / "artifacts")
    experiments_size = get_dir_size(repo_root / "experiments")

    # 3. Production Deployment Package Formulation
    # Production deployment includes:
    # - production/ runtime package
    # - backends/ execution bridges
    # - ai/ navigation core (planning C++, costmap, safety, localization, models/traversability_net.py)
    # - LightweightTraversabilityNet best.pt checkpoint
    # - compiled shared libraries (dwa_cpp.dll, dwa_cpp_scalar.dll)
    # - configs/
    # - deployment launch entrypoints
    # Excludes:
    # - .git, .pytest_cache, .venv_xpu
    # - data/, experiments/, artifacts/
    # - SAM-TP weights & modules
    # - rellis3d_gpu_ready/ raw images
    # - training checkpoints (except best.pt)
    # - benchmarks/, notebooks/

    production_exclusion_patterns = [
        ".git",
        ".pytest_cache",
        ".venv_xpu",
        "data",
        "experiments",
        "artifacts",
        "notebooks",
        "scratch",
        "benchmarks",
        "clean_github_snapshot",
        "research_reference",
        "sam_tp",
        "rellis3d_gpu_ready/train",
        "rellis3d_gpu_ready/val",
        "rellis3d_gpu_ready/test",
    ]

    total_repo_excl_git = get_dir_size(repo_root, exclude_patterns=[".git", ".venv_xpu"])
    git_dir_size = get_dir_size(repo_root / ".git")

    # Estimate production deployment files:
    prod_ai_size = get_dir_size(
        repo_root / "ai",
        exclude_patterns=["sam_tp.py", "__pycache__", "training"]
    )
    prod_production_size = get_dir_size(repo_root / "production")
    prod_backends_size = get_dir_size(repo_root / "backends")
    prod_configs_size = get_dir_size(repo_root / "configs")
    prod_scripts_size = get_dir_size(repo_root / "scripts", exclude_patterns=["train", "benchmark"])
    prod_sim_size = get_dir_size(repo_root / "simulation", exclude_patterns=["__pycache__"])

    minimal_prod_package_size = (
        prod_ai_size
        + prod_production_size
        + prod_backends_size
        + prod_configs_size
        + prod_scripts_size
        + prod_sim_size
        + prod_model_size
        + simd_dll_size
        + scalar_dll_size
    )

    excluded_research_total = (
        total_repo_excl_git - minimal_prod_package_size
    )

    audit_summary = {
        "repository_path": str(repo_root),
        "development_repository_size_bytes": total_repo_excl_git,
        "development_repository_size_mb": round(total_repo_excl_git / (1024 * 1024), 2),
        "git_metadata_size_mb": round(git_dir_size / (1024 * 1024), 2),
        "minimal_production_package_size_bytes": minimal_prod_package_size,
        "minimal_production_package_size_mb": round(minimal_prod_package_size / (1024 * 1024), 2),
        "excluded_research_assets_mb": round(excluded_research_total / (1024 * 1024), 2),
        "size_reduction_percent": round(
            ((total_repo_excl_git - minimal_prod_package_size) / total_repo_excl_git) * 100, 1
        ) if total_repo_excl_git > 0 else 0.0,
        "components": {
            "native_cpp_simd_binary_kb": round(simd_dll_size / 1024, 1),
            "native_cpp_scalar_binary_kb": round(scalar_dll_size / 1024, 1),
            "production_model_lightweight_traversability_net_mb": round(prod_model_size / (1024 * 1024), 2),
            "production_runtime_and_safety_kb": round((prod_production_size + prod_backends_size) / 1024, 1),
            "production_navigation_core_mb": round(prod_ai_size / (1024 * 1024), 2),
            "configurations_kb": round(prod_configs_size / 1024, 1),
        },
        "excluded_categories": {
            "training_datasets_mb": round((data_size + rellis_data_size) / (1024 * 1024), 2),
            "sam_tp_weights_and_runtime_mb": round(sam_assets_size / (1024 * 1024), 2),
            "temporary_venv_xpu_mb": round(venv_xpu_size / (1024 * 1024), 2),
            "benchmarks_and_reports_mb": round(
                (dir_sizes.get("benchmarks", 0) + dir_sizes.get("reports", 0)) / (1024 * 1024), 2
            ),
            "artifacts_and_scratch_mb": round((artifacts_size + experiments_size) / (1024 * 1024), 2),
            "pytest_cache_mb": round(pytest_cache_size / (1024 * 1024), 2),
        },
    }

    # Print Report
    print(f"\nTotal Development Repo (working tree): {audit_summary['development_repository_size_mb']} MB")
    print(f"Minimal Production Deployment:        {audit_summary['minimal_production_package_size_mb']} MB")
    print(f"Excluded Research Assets:              {audit_summary['excluded_research_assets_mb']} MB")
    print(f"Deployment Size Reduction:             {audit_summary['size_reduction_percent']} %")
    print("-" * 70)
    print("PRODUCTION COMPONENTS:")
    for k, v in audit_summary["components"].items():
        print(f"  - {k:<45}: {v}")
    print("-" * 70)
    print("EXCLUDED RESEARCH CATEGORIES:")
    for k, v in audit_summary["excluded_categories"].items():
        print(f"  - {k:<45}: {v} MB")
    print("=" * 70)

    # Save output
    output_path = repo_root / "reports" / "deployment_size_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Saved deployment audit report to: {output_path}")

    return audit_summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    audit_repository(root)
