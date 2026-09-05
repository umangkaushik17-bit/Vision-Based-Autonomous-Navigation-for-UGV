"""High-precision timing, FPS estimation, and latency analytics for vision processing."""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
import numpy as np

@dataclass
class FrameTimingSnapshot:
    """Latency breakdown for an individual processed frame in milliseconds."""
    frame_index: int
    preprocess_ms: float
    process_ms: float
    total_frame_ms: float
    instantaneous_fps: float

@dataclass
class PerformanceReport:
    """Consolidated performance statistics for a processing run."""
    total_frames_processed: int
    total_elapsed_sec: float
    source_fps: Optional[float]
    actual_average_fps: float
    avg_total_latency_ms: float
    min_total_latency_ms: float
    max_total_latency_ms: float
    p50_total_latency_ms: float
    p95_total_latency_ms: float
    avg_preprocess_ms: float
    avg_process_ms: float

    def summary_string(self) -> str:
        """Format an informative multi-line performance summary."""
        lines = [
            "================ VISION PERFORMANCE REPORT ================",
            f"Total Frames Processed : {self.total_frames_processed}",
            f"Total Elapsed Time     : {self.total_elapsed_sec:.2f} s",
            f"Source Nominal FPS     : {self.source_fps if self.source_fps else 'N/A'}",
            f"Actual Processing FPS  : {self.actual_average_fps:.2f} FPS",
            f"Latency (Avg)          : {self.avg_total_latency_ms:.2f} ms",
            f"Latency (Min / Max)    : {self.min_total_latency_ms:.2f} ms / {self.max_total_latency_ms:.2f} ms",
            f"Latency (P50 Median)   : {self.p50_total_latency_ms:.2f} ms",
            f"Latency (P95 Tail)     : {self.p95_total_latency_ms:.2f} ms",
            f"Preprocess Time (Avg)  : {self.avg_preprocess_ms:.2f} ms",
            f"Processing Time (Avg)  : {self.avg_process_ms:.2f} ms",
            "==========================================================="
        ]
        return "\n".join(lines)

class FrameTimer:
    """Scoped monotonic timer measuring individual sub-stages of frame execution."""
    
    def __init__(self):
        self._start_preprocess: float = 0.0
        self._end_preprocess: float = 0.0
        self._start_process: float = 0.0
        self._end_process: float = 0.0
        self._start_frame: float = 0.0
        self._end_frame: float = 0.0

    def start_frame(self):
        self._start_frame = time.perf_counter()

    def start_preprocess(self):
        self._start_preprocess = time.perf_counter()

    def stop_preprocess(self):
        self._end_preprocess = time.perf_counter()

    def start_process(self):
        self._start_process = time.perf_counter()

    def stop_process(self):
        self._end_process = time.perf_counter()

    def stop_frame(self) -> float:
        self._end_frame = time.perf_counter()
        return (self._end_frame - self._start_frame) * 1000.0

    @property
    def preprocess_ms(self) -> float:
        return max(0.0, (self._end_preprocess - self._start_preprocess) * 1000.0)

    @property
    def process_ms(self) -> float:
        return max(0.0, (self._end_process - self._start_process) * 1000.0)

    @property
    def total_frame_ms(self) -> float:
        return max(0.0, (self._end_frame - self._start_frame) * 1000.0)

class FPSMeter:
    """High-resolution rolling window FPS estimator and benchmark metrics collector."""
    
    def __init__(self, window_size: int = 30, source_fps: Optional[float] = None):
        self.window_size = max(1, window_size)
        self.source_fps = source_fps
        self._timestamps: Deque[float] = collections.deque(maxlen=self.window_size)
        self._preprocess_latencies: List[float] = []
        self._process_latencies: List[float] = []
        self._total_latencies: List[float] = []
        self._start_time: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self.frame_count: int = 0

    def update(self, timer: FrameTimer) -> FrameTimingSnapshot:
        """Record completed frame metrics and update rolling statistics."""
        now = time.perf_counter()
        if self._start_time is None:
            self._start_time = now

        self.frame_count += 1
        self._timestamps.append(now)

        pre_ms = timer.preprocess_ms
        proc_ms = timer.process_ms
        tot_ms = timer.total_frame_ms

        self._preprocess_latencies.append(pre_ms)
        self._process_latencies.append(proc_ms)
        self._total_latencies.append(tot_ms)

        # Instantaneous FPS
        inst_fps = (1000.0 / tot_ms) if tot_ms > 0.0 else 0.0
        self._last_timestamp = now

        return FrameTimingSnapshot(
            frame_index=self.frame_count,
            preprocess_ms=pre_ms,
            process_ms=proc_ms,
            total_frame_ms=tot_ms,
            instantaneous_fps=inst_fps
        )

    @property
    def moving_average_fps(self) -> float:
        """Calculate moving-average FPS across the rolling window."""
        if len(self._timestamps) < 2:
            return 0.0
        time_span = self._timestamps[-1] - self._timestamps[0]
        if time_span <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / time_span

    @property
    def overall_average_fps(self) -> float:
        """Calculate aggregate FPS across the entire run."""
        if self._start_time is None or self._last_timestamp is None:
            return 0.0
        total_time = self._last_timestamp - self._start_time
        if total_time <= 0.0:
            return 0.0
        return self.frame_count / total_time

    def compute_percentiles(self, percentiles: List[int] = [50, 95]) -> Dict[int, float]:
        """Compute latency percentiles (e.g. P50, P95) in milliseconds."""
        if not self._total_latencies:
            return {p: 0.0 for p in percentiles}
        arr = np.array(self._total_latencies)
        res = {}
        for p in percentiles:
            res[p] = float(np.percentile(arr, p))
        return res

    def generate_report(self) -> PerformanceReport:
        """Generate final performance report with complete statistical breakdowns."""
        total_elapsed = (
            (self._last_timestamp - self._start_time)
            if (self._start_time and self._last_timestamp)
            else 0.0
        )
        percentiles = self.compute_percentiles([50, 95])

        return PerformanceReport(
            total_frames_processed=self.frame_count,
            total_elapsed_sec=total_elapsed,
            source_fps=self.source_fps,
            actual_average_fps=self.overall_average_fps,
            avg_total_latency_ms=float(np.mean(self._total_latencies)) if self._total_latencies else 0.0,
            min_total_latency_ms=float(np.min(self._total_latencies)) if self._total_latencies else 0.0,
            max_total_latency_ms=float(np.max(self._total_latencies)) if self._total_latencies else 0.0,
            p50_total_latency_ms=percentiles.get(50, 0.0),
            p95_total_latency_ms=percentiles.get(95, 0.0),
            avg_preprocess_ms=float(np.mean(self._preprocess_latencies)) if self._preprocess_latencies else 0.0,
            avg_process_ms=float(np.mean(self._process_latencies)) if self._process_latencies else 0.0
        )
