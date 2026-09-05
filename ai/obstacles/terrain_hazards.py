"""Specialized terrain hazard extractors addressing Phase 3 failure modes.

Specifically handles:
1. Water & specular reflections (RUGD creek case)
2. Low and dark protruding obstacles (rocks, debris, curbs)
3. Negative obstacles (ditches, ruts, drop-offs)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from ai.obstacles.base import HazardType, ObstacleDetection, RiskLevel


class WaterReflectionDetector:
    """Detects standing water, creeks, puddles, and specular ground reflections."""

    def __init__(
        self,
        min_water_area: int = 150,
        specular_v_thresh: int = 225,
        specular_s_thresh: int = 50,
        blue_chroma_margin: float = 0.05
    ) -> None:
        self.min_water_area = min_water_area
        self.specular_v_thresh = specular_v_thresh
        self.specular_s_thresh = specular_s_thresh
        self.blue_chroma_margin = blue_chroma_margin

    def detect(self, rgb_image: np.ndarray) -> List[ObstacleDetection]:
        """Identify water bodies and mirror reflection zones in ground regions."""
        h, w = rgb_image.shape[:2]
        # Restrict ground region (lower 75% of frame)
        ground_start_y = int(h * 0.25)
        ground_rgb = rgb_image[ground_start_y:, :]

        hsv = cv2.cvtColor(ground_rgb, cv2.COLOR_RGB2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        # 1. Specular highlights: high brightness, low saturation
        specular_mask = (v_chan >= self.specular_v_thresh) & (s_chan <= self.specular_s_thresh)

        # 2. Sky reflection in water: elevated blue channel relative to red
        r = ground_rgb[:, :, 0].astype(np.float32)
        g = ground_rgb[:, :, 1].astype(np.float32)
        b = ground_rgb[:, :, 2].astype(np.float32)
        total = r + g + b + 1e-5
        norm_b = b / total
        norm_r = r / total
        sky_reflection_mask = (norm_b > (norm_r + self.blue_chroma_margin)) & (v_chan > 70)

        # 3. Combine water cues
        raw_water = (specular_mask | sky_reflection_mask).astype(np.uint8) * 255

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(raw_water, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        full_mask = np.zeros((h, w), dtype=bool)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_water_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            # Offset y back to full frame
            y_full = y + ground_start_y
            bbox = (x, y_full, x + cw, y_full + ch)

            # Local contour mask in full frame coordinates
            cnt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask[ground_start_y:, :], [cnt], -1, 1, -1)
            full_mask |= (cnt_mask > 0)

            # Calculate confidence from size and intensity
            conf = min(0.99, 0.60 + float(area) / (w * h * 0.15))

            detections.append(
                ObstacleDetection(
                    class_name="water_reflection",
                    confidence=conf,
                    bbox=bbox,
                    hazard_type=HazardType.WATER,
                    risk_level=RiskLevel.CRITICAL,
                    source_model="terrain_hazard_water",
                    mask=(cnt_mask > 0),
                    attributes={"area_pixels": int(area), "specular": bool(specular_mask[y:y+ch, x:x+cw].any())}
                )
            )

        return detections


class LowDarkObstacleDetector:
    """Detects low protruding dark rocks, stones, curbs, and debris on trails."""

    def __init__(
        self,
        min_area: int = 50,
        max_area: int = 25000,
        contrast_threshold_ratio: float = 0.65
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.contrast_threshold_ratio = contrast_threshold_ratio

    def detect(self, rgb_image: np.ndarray) -> List[ObstacleDetection]:
        """Detect localized dark contrast anomalies on the terrain surface."""
        h, w = rgb_image.shape[:2]
        ground_start_y = int(h * 0.35)
        ground_rgb = rgb_image[ground_start_y:, :]

        gray = cv2.cvtColor(ground_rgb, cv2.COLOR_RGB2GRAY)
        med_lum = float(np.median(gray))

        # Identify pixels significantly darker than the terrain baseline
        dark_thresh = int(med_lum * self.contrast_threshold_ratio)
        dark_mask = (gray < dark_thresh).astype(np.uint8) * 255

        # Morphological filter to isolate discrete objects
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = float(cw) / float(ch) if ch > 0 else 1.0

            # Ignore extremely wide flat shadow strips
            if aspect_ratio > 5.0 and ch < 8:
                continue

            y_full = y + ground_start_y
            bbox = (x, y_full, x + cw, y_full + ch)

            # Classify as Rock or Step based on geometry
            h_type = HazardType.ROCK if aspect_ratio < 2.5 else HazardType.STEP
            risk = RiskLevel.HIGH if area > 400 else RiskLevel.MEDIUM
            conf = min(0.95, 0.55 + (med_lum - gray[y:y+ch, x:x+cw].mean()) / (med_lum + 1e-5) * 0.4)

            cnt_mask = np.zeros((h, w), dtype=bool)
            local_cnt_mask = np.zeros((h - ground_start_y, w), dtype=np.uint8)
            cv2.drawContours(local_cnt_mask, [cnt], -1, 1, -1)
            cnt_mask[ground_start_y:, :] = (local_cnt_mask > 0)

            detections.append(
                ObstacleDetection(
                    class_name="low_dark_obstacle",
                    confidence=float(conf),
                    bbox=bbox,
                    hazard_type=h_type,
                    risk_level=risk,
                    source_model="terrain_hazard_contrast",
                    mask=cnt_mask,
                    attributes={"area": int(area), "aspect_ratio": round(aspect_ratio, 2)}
                )
            )

        return detections


class NegativeObstacleDetector:
    """Detects negative obstacles such as ditches, trenches, potholes, and drop-offs."""

    def __init__(
        self,
        min_width: int = 25,
        min_depth_px: int = 15,
        gradient_threshold: int = 35
    ) -> None:
        self.min_width = min_width
        self.min_depth_px = min_depth_px
        self.gradient_threshold = gradient_threshold

    def detect(self, rgb_image: np.ndarray) -> List[ObstacleDetection]:
        """Detect downward depressions and shadow cavities across terrain boundaries."""
        h, w = rgb_image.shape[:2]
        ground_start_y = int(h * 0.40)
        ground_rgb = rgb_image[ground_start_y:, :]

        gray = cv2.cvtColor(ground_rgb, cv2.COLOR_RGB2GRAY)
        # Vertical gradient highlights horizontal shelf edges (ditches and potholes)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobely = np.uint8(np.absolute(sobely))

        # Depressions typically show high downward edge with dark interior
        ret, edge_mask = cv2.threshold(abs_sobely, self.gradient_threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
        dilated = cv2.dilate(edge_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < self.min_width or ch < self.min_depth_px:
                continue

            # Verify if internal region is relatively darker (cavity shadow)
            patch = gray[y:y+ch, x:x+cw]
            surrounding_mean = gray[max(0, y-10):y+ch, max(0, x-10):min(w, x+cw+10)].mean()
            if patch.mean() > surrounding_mean * 0.90:
                continue

            y_full = y + ground_start_y
            bbox = (x, y_full, x + cw, y_full + ch)

            detections.append(
                ObstacleDetection(
                    class_name="ditch_or_pothole",
                    confidence=0.72,
                    bbox=bbox,
                    hazard_type=HazardType.DITCH,
                    risk_level=RiskLevel.CRITICAL,
                    source_model="terrain_hazard_ditch",
                    attributes={"width": cw, "height": ch}
                )
            )

        return detections
