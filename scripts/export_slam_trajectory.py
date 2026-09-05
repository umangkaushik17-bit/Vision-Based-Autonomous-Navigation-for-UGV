"""Utility script to convert, format, and export SLAM trajectories into canonical CSV and MATLAB .mat files."""

from __future__ import annotations

import argparse
from pathlib import Path
from ai.localization.trajectory import Trajectory3D


def export_trajectory(
    input_file: str,
    output_dir: str = "outputs/slam/trajectories",
    matlab_dir: str = "matlab/data"
):
    inp = Path(input_file)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    out_d = Path(output_dir)
    mat_d = Path(matlab_dir)
    out_d.mkdir(parents=True, exist_ok=True)
    mat_d.mkdir(parents=True, exist_ok=True)

    traj = Trajectory3D.from_csv(inp)
    stem = inp.stem

    csv_out = out_d / f"{stem}_canonical.csv"
    mat_out = mat_d / f"{stem}.mat"

    traj.to_csv(csv_out)
    traj.to_mat(mat_out, var_name="trajectory")

    print(f"[OK] Exported {len(traj)} points:")
    print(f"  Canonical CSV : {csv_out}")
    print(f"  MATLAB .mat   : {mat_out}")


def main():
    parser = argparse.ArgumentParser(description="Export SLAM trajectory to canonical CSV and MATLAB .mat.")
    parser.add_argument("input_file", help="Input trajectory CSV file")
    parser.add_argument("--output-dir", default="outputs/slam/trajectories")
    parser.add_argument("--matlab-dir", default="matlab/data")
    args = parser.parse_args()

    export_trajectory(args.input_file, args.output_dir, args.matlab_dir)


if __name__ == "__main__":
    main()
