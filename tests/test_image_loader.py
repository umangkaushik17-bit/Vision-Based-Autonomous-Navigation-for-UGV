"""Automated tests for image loader and writer."""

from pathlib import Path
import numpy as np
import pytest

from ai.io.image_loader import load_image
from ai.io.frame_writer import save_image
from ai.core.frame_source import ImageSource

def test_load_valid_image():
    sample_path = "data/test/sample.jpg"
    img, (w, h) = load_image(sample_path, color_format="BGR")
    assert isinstance(img, np.ndarray)
    assert img.shape == (480, 640, 3)
    assert (w, h) == (640, 480)

def test_load_image_color_formats():
    sample_path = "data/test/sample.jpg"
    rgb, _ = load_image(sample_path, color_format="RGB")
    assert rgb.shape == (480, 640, 3)

    gray, _ = load_image(sample_path, color_format="GRAY")
    assert gray.shape == (480, 640)
    assert gray.ndim == 2

def test_load_nonexistent_image():
    with pytest.raises(FileNotFoundError):
        load_image("data/test/does_not_exist.jpg")

def test_load_unsupported_extension(tmp_path):
    invalid_file = tmp_path / "test.xyz"
    invalid_file.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported image extension"):
        load_image(str(invalid_file))

def test_image_source_abstraction():
    source = ImageSource("data/test/sample.jpg")
    with source:
        assert source.is_opened
        assert source.width == 640
        assert source.height == 480
        assert source.total_frames == 1

        frames = list(source)
        assert len(frames) == 1
        f = frames[0]
        assert f.raw_frame.shape == (480, 640, 3)
        assert f.source_type == "image"

def test_save_image(tmp_path):
    out_file = tmp_path / "subdir" / "out.jpg"
    dummy = np.full((100, 100, 3), 128, dtype=np.uint8)
    saved_path = save_image(dummy, str(out_file))
    assert Path(saved_path).exists()
    assert Path(saved_path).stat().st_size > 0
