"""Independent visualization and HUD rendering utilities."""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

def create_side_by_side(
    original_frame: np.ndarray,
    processed_frame: np.ndarray,
    label_left: str = "RAW INPUT",
    label_right: str = "PROCESSED OUTPUT"
) -> np.ndarray:
    """Concatenate original and processed frames horizontally with stylized header tags.
    
    Args:
        original_frame: Unmodified input frame (H, W, 3) uint8 BGR.
        processed_frame: Output frame from vision processor (H, W, 3) uint8 BGR.
        label_left: Header title for left pane.
        label_right: Header title for right pane.
        
    Returns:
        Side-by-side composite image (H, 2*W, 3) uint8 BGR.
    """
    h_left, w_left = original_frame.shape[:2]
    h_right, w_right = processed_frame.shape[:2]

    # Ensure processed frame matches original height
    if (h_right, w_right) != (h_left, w_left):
        processed_resized = cv2.resize(processed_frame, (w_left, h_left))
    else:
        processed_resized = processed_frame

    # Ensure both are 3-channel uint8
    if original_frame.ndim == 2:
        orig_bgr = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
    else:
        orig_bgr = original_frame

    if processed_resized.ndim == 2:
        proc_bgr = cv2.cvtColor(processed_resized, cv2.COLOR_GRAY2BGR)
    else:
        proc_bgr = processed_resized

    # Draw header banners
    orig_annotated = orig_bgr.copy()
    proc_annotated = proc_bgr.copy()

    # Semi-transparent top bar on each pane
    header_h = 32
    cv2.rectangle(orig_annotated, (0, 0), (w_left, header_h), (25, 25, 25), -1)
    cv2.putText(
        orig_annotated, label_left, (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA
    )

    cv2.rectangle(proc_annotated, (0, 0), (w_left, header_h), (25, 25, 25), -1)
    cv2.putText(
        proc_annotated, label_right, (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, cv2.LINE_AA
    )

    # Horizontal stack
    combined = np.hstack([orig_annotated, proc_annotated])
    return combined

def draw_hud_overlay(
    image: np.ndarray,
    frame_idx: int,
    fps: float,
    latency_ms: float,
    preprocess_ms: Optional[float] = None,
    process_ms: Optional[float] = None,
    p95_ms: Optional[float] = None
) -> np.ndarray:
    """Draw an informative semi-transparent HUD banner with live latency and FPS metrics.
    
    Args:
        image: Frame to draw on (mutated copy returned).
        frame_idx: Current frame sequence index.
        fps: Current moving-average FPS.
        latency_ms: Total frame latency in milliseconds.
        preprocess_ms: Optional preprocessing latency.
        process_ms: Optional vision processing latency.
        p95_ms: Optional P95 latency percentile.
        
    Returns:
        Frame with HUD overlay.
    """
    output = image.copy()
    h, w = output.shape[:2]

    # HUD Box parameters (bottom banner)
    hud_h = 36
    hud_y = h - hud_h
    overlay = output.copy()
    cv2.rectangle(overlay, (0, hud_y), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)

    # Top border line for HUD
    cv2.line(output, (0, hud_y), (w, hud_y), (0, 180, 255), 1)

    # Format metric string
    text = f"FRAME: {frame_idx:04d}  |  FPS: {fps:5.1f}  |  LATENCY: {latency_ms:5.1f} ms"
    if p95_ms is not None and p95_ms > 0:
        text += f" (P95: {p95_ms:.1f}ms)"
    if preprocess_ms is not None and process_ms is not None:
        text += f"  [Pre: {preprocess_ms:.1f}ms / Proc: {process_ms:.1f}ms]"

    cv2.putText(
        output,
        text,
        (12, h - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    return output

def display_frame(window_name: str, frame: np.ndarray, delay_ms: int = 1) -> int:
    """Display frame in an OpenCV window and return pressed key code.
    
    Returns:
        Key code pressed (e.g. 27 for ESC, ord('q') for quit).
    """
    cv2.imshow(window_name, frame)
    return cv2.waitKey(delay_ms) & 0xFF
