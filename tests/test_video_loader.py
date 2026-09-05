"""Automated tests for video loader, video source, and video writer."""

from pathlib import Path
import numpy as np
import pytest

from ai.io.video_loader import VideoReader
from ai.io.frame_writer import VideoWriter
from ai.core.frame_source import VideoSource

SAMPLE_VIDEO = "data/test/sample.mp4"

def test_video_reader_metadata():
    reader = VideoReader(SAMPLE_VIDEO)
    with reader:
        assert reader.width == 640
        assert reader.height == 480
        assert reader.fps > 0
        assert reader.total_frames == 45
        assert reader.duration_sec > 0

def test_video_reader_streaming():
    reader = VideoReader(SAMPLE_VIDEO)
    frames = list(reader.stream_frames())
    assert len(frames) == 45
    idx, first_frame = frames[0]
    assert idx == 0
    assert first_frame.shape == (480, 640, 3)

def test_video_reader_nonexistent():
    reader = VideoReader("data/test/nonexistent_video.mp4")
    with pytest.raises(FileNotFoundError):
        reader.open()

def test_video_source_abstraction():
    source = VideoSource(SAMPLE_VIDEO)
    with source:
        assert source.is_opened
        assert source.width == 640
        assert source.height == 480
        
        frames = list(source)
        assert len(frames) == 45
        assert frames[0].source_type == "video"
        assert frames[-1].frame_index == 44

def test_video_writer(tmp_path):
    out_video = tmp_path / "test_out.mp4"
    writer = VideoWriter(str(out_video), fps=30.0, width=320, height=240)
    with writer:
        for _ in range(10):
            dummy = np.zeros((240, 320, 3), dtype=np.uint8)
            writer.write_frame(dummy)

    assert out_video.exists()
    assert out_video.stat().st_size > 0
