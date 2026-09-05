"""Automated tests for high-resolution timing, FPS estimation, and percentiles."""

import time
import numpy as np
import pytest

from ai.core.timing import FrameTimer, FPSMeter

def test_frame_timer():
    timer = FrameTimer()
    timer.start_frame()
    timer.start_preprocess()
    time.sleep(0.005)
    timer.stop_preprocess()

    timer.start_process()
    time.sleep(0.005)
    timer.stop_process()

    tot_ms = timer.stop_frame()
    assert timer.preprocess_ms >= 4.0
    assert timer.process_ms >= 4.0
    assert tot_ms >= (timer.preprocess_ms + timer.process_ms)

def test_fps_meter_and_percentiles():
    meter = FPSMeter(window_size=10, source_fps=30.0)
    timer = FrameTimer()

    # Feed 20 frames with synthetic timings
    for i in range(20):
        timer.start_frame()
        timer.start_preprocess()
        time.sleep(0.002)
        timer.stop_preprocess()
        timer.start_process()
        time.sleep(0.002)
        timer.stop_process()
        timer.stop_frame()

        snap = meter.update(timer)
        assert snap.frame_index == i + 1
        assert snap.total_frame_ms > 0

    assert meter.frame_count == 20
    assert meter.moving_average_fps > 0
    assert meter.overall_average_fps > 0

    percentiles = meter.compute_percentiles([50, 95])
    assert 50 in percentiles
    assert 95 in percentiles
    assert percentiles[95] >= percentiles[50]

    report = meter.generate_report()
    assert report.total_frames_processed == 20
    assert report.source_fps == 30.0
    assert report.actual_average_fps > 0
    assert "VISION PERFORMANCE REPORT" in report.summary_string()
