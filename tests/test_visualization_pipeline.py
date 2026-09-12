"""Tests for Phase 13.17 Dynamic Navigation Visualization Pipeline.

Verifies:
1. Ingestion of canonical telemetry schemas.
2. Graceful rejection of malformed or empty telemetry files.
3. Generation of high-resolution diagnostic plots (XY trajectory, velocity, clearance,
   prediction fan, tracking, and dashboard).
4. Physical image integrity: valid encodings, non-zero dimensions, non-blank pixels.
5. Multi-frame animated GIF generation, frame count > 1, frame dimension consistency.
6. Robust handling of edge cases (e.g. open corridor with zero dynamic obstacles, infinite clearances).
7. Strict preservation of finite numeric boundaries (no unhandled NaN/Inf).
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from PIL import Image

from scripts.visualize_dynamic_tracks import (
    DynamicVisualizationEngine,
    TelemetryDataset,
    TelemetryFrame,
    TelemetryParser,
)


@pytest.fixture
def sample_telemetry_b05_path() -> Path:
    """Returns path to canonical Phase 16 single dynamic obstacle telemetry."""
    path = Path("simulation/outputs/phase16/telemetry/B05_single_dynamic_seed42.json")
    if not path.exists():
        pytest.skip(f"Canonical telemetry file not found: {path}")
    return path


@pytest.fixture
def sample_telemetry_b01_path() -> Path:
    """Returns path to canonical Phase 16 open baseline telemetry."""
    path = Path("simulation/outputs/phase16/telemetry/B01_flat_baseline_seed42.json")
    if not path.exists():
        pytest.skip(f"Canonical telemetry file not found: {path}")
    return path


class TestTelemetryParser:
    """Validates parser ingestion, schema detection, and data constraints."""

    def test_load_valid_telemetry(self, sample_telemetry_b05_path: Path):
        dataset = TelemetryParser.load_file(sample_telemetry_b05_path)
        assert isinstance(dataset, TelemetryDataset)
        assert dataset.scenario_id == "B05_single_dynamic"
        assert len(dataset.frames) > 50
        assert dataset.seed == 42

        first_frame = dataset.frames[0]
        assert isinstance(first_frame, TelemetryFrame)
        assert first_frame.step == 0
        assert math.isfinite(first_frame.robot_x)
        assert math.isfinite(first_frame.robot_y)
        assert math.isfinite(first_frame.robot_v)
        assert math.isfinite(first_frame.min_clearance_m)
        assert first_frame.behavior_state in ["PROCEED", "AVOID", "WAIT", "STOP"]

    def test_load_nonexistent_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            TelemetryParser.load_file(tmp_path / "nonexistent_telemetry.json")

    def test_load_malformed_json(self, tmp_path: Path):
        bad_json = tmp_path / "corrupted.json"
        bad_json.write_text("{ unclosed_json: true, ", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed JSON"):
            TelemetryParser.load_file(bad_json)

    def test_load_empty_points(self, tmp_path: Path):
        empty_file = tmp_path / "empty.json"
        empty_file.write_text(json.dumps({"telemetry_points": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="contains no step records"):
            TelemetryParser.load_file(empty_file)

    def test_reject_nan_coordinates(self, tmp_path: Path):
        nan_file = tmp_path / "nan_coord.json"
        nan_data = {
            "telemetry_points": [
                {
                    "step": 0,
                    "sim_time": 0.0,
                    "robot_x": float("nan"),
                    "robot_y": 0.0,
                    "robot_v": 0.0,
                    "robot_w": 0.0,
                }
            ]
        }
        nan_file.write_text(json.dumps(nan_data), encoding="utf-8")
        with pytest.raises(ValueError, match="Non-finite value"):
            TelemetryParser.load_file(nan_file)

    def test_infinite_clearance_sanitization(self, tmp_path: Path):
        inf_file = tmp_path / "inf_clearance.json"
        inf_data = {
            "telemetry_points": [
                {
                    "step": 0,
                    "sim_time": 0.0,
                    "robot_x": 0.0,
                    "robot_y": 0.0,
                    "robot_v": 0.0,
                    "robot_w": 0.0,
                    "min_clearance_m": float("inf"),
                }
            ]
        }
        inf_file.write_text(json.dumps(inf_data), encoding="utf-8")
        dataset = TelemetryParser.load_file(inf_file)
        assert math.isfinite(dataset.frames[0].min_clearance_m)
        assert dataset.frames[0].min_clearance_m <= 10.0


class TestDynamicVisualizationEngine:
    """Validates plotting, dashboard generation, animation rendering, and image encoding."""

    def test_generate_all_artifacts(self, sample_telemetry_b05_path: Path, tmp_path: Path):
        dataset = TelemetryParser.load_file(sample_telemetry_b05_path)
        engine = DynamicVisualizationEngine(dataset, tmp_path)
        artifacts = engine.generate_all(create_animation=True)

        expected_keys = [
            "trajectory_xy",
            "velocity_profiles",
            "clearance_vs_time",
            "prediction_fan",
            "dynamic_tracking",
            "engineering_dashboard",
            "avoidance_animation",
        ]
        for k in expected_keys:
            assert k in artifacts, f"Missing expected artifact key: {k}"
            p = artifacts[k]
            assert p.exists(), f"Artifact file does not exist: {p}"
            assert p.stat().st_size > 5000, f"Artifact {p} is suspiciously small ({p.stat().st_size} bytes)"

    def test_image_visual_integrity(self, sample_telemetry_b05_path: Path, tmp_path: Path):
        dataset = TelemetryParser.load_file(sample_telemetry_b05_path)
        engine = DynamicVisualizationEngine(dataset, tmp_path)

        png_artifacts = [
            engine.plot_trajectory_xy(),
            engine.plot_velocity_profiles(),
            engine.plot_clearance_vs_time(),
            engine.plot_prediction_fan(),
            engine.plot_dynamic_tracking(),
            engine.plot_engineering_dashboard(),
        ]

        for p in png_artifacts:
            with Image.open(p) as img:
                assert img.format == "PNG", f"File {p} is not a valid PNG"
                w, h = img.size
                assert w >= 800, f"Image {p} width {w} is below 800px"
                assert h >= 600, f"Image {p} height {h} is below 600px"

                # Check non-blank image: variance across pixels must be significant
                arr = np.array(img.convert("RGB"))
                std_dev = np.std(arr)
                assert std_dev > 10.0, f"Image {p} appears completely blank (std={std_dev:.2f})"

    def test_animation_integrity(self, sample_telemetry_b05_path: Path, tmp_path: Path):
        dataset = TelemetryParser.load_file(sample_telemetry_b05_path)
        engine = DynamicVisualizationEngine(dataset, tmp_path)
        gif_path = engine.generate_avoidance_animation(max_frames=10, fps=5)

        assert gif_path.exists()
        assert gif_path.stat().st_size > 10000

        with Image.open(gif_path) as gif:
            assert gif.format == "GIF", f"File {gif_path} is not a valid GIF"
            n_frames = gif.n_frames
            assert n_frames >= 5, f"Animation has fewer frames than expected: {n_frames}"

            # Check consistency of dimensions across frames
            base_size = gif.size
            for frame_idx in range(n_frames):
                gif.seek(frame_idx)
                assert gif.size == base_size, f"Frame {frame_idx} size mismatch: {gif.size} vs {base_size}"

    def test_open_baseline_scenario_handling(self, sample_telemetry_b01_path: Path, tmp_path: Path):
        """Validates that scenarios with zero obstacles (e.g. B01) plot without errors or crashes."""
        dataset = TelemetryParser.load_file(sample_telemetry_b01_path)
        engine = DynamicVisualizationEngine(dataset, tmp_path)
        artifacts = engine.generate_all(create_animation=False)

        for name, p in artifacts.items():
            assert p.exists()
            with Image.open(p) as img:
                assert img.size[0] > 0 and img.size[1] > 0
