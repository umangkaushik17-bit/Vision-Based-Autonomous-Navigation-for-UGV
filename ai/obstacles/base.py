"""Data contracts and base classes for Explicit Obstacle and Hazard Perception."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class HazardType(str, Enum):
    """Explicit hazard classification types for outdoor UGV navigation."""

    WATER = "WATER"
    DITCH = "DITCH"
    POTHOLE = "POTHOLE"
    ROCK = "ROCK"
    STEP = "STEP"
    CURB = "CURB"
    VEGETATION = "VEGETATION"
    TREE = "TREE"
    VEHICLE = "VEHICLE"
    PERSON = "PERSON"
    ANIMAL = "ANIMAL"
    UNKNOWN_HAZARD = "UNKNOWN_HAZARD"


class RiskLevel(str, Enum):
    """Categorical risk severity levels for navigation cost generation."""

    CRITICAL = "CRITICAL"  # Mandatory emergency stop or lethal collision hazard (cost = 1.0)
    HIGH = "HIGH"          # Severe mobility risk, impassable obstacle (cost = 0.85)
    MEDIUM = "MEDIUM"      # Moderate risk, requires cautious speed reduction (cost = 0.50)
    LOW = "LOW"            # Minor risk or compliant terrain (cost = 0.15)
    UNKNOWN = "UNKNOWN"    # Unresolved perceptual ambiguity (cost = 0.50 exploration penalty)


@dataclass
class ObstacleDetection:
    """Individual obstacle or terrain hazard detection."""

    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    hazard_type: HazardType
    risk_level: RiskLevel
    source_model: str
    mask: Optional[np.ndarray] = None  # Binary boolean mask (H, W) or local patch
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        """Bounding box pixel width."""
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        """Bounding box pixel height."""
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        """Bounding box pixel area."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Center coordinate (cx, cy)."""
        return ((self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize detection to JSON-serializable dictionary."""
        return {
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": list(self.bbox),
            "hazard_type": self.hazard_type.value,
            "risk_level": self.risk_level.value,
            "source_model": self.source_model,
            "area": self.area,
            "has_mask": self.mask is not None,
            "attributes": self.attributes
        }


class ObstacleDetectorBase(ABC):
    """Abstract base class for all obstacle and hazard detection components."""

    @abstractmethod
    def detect(self, image: Union[np.ndarray, str, Path]) -> List[ObstacleDetection]:
        """Detect explicit obstacles and terrain hazards in an input frame."""
        pass

    @abstractmethod
    def get_detector_info(self) -> Dict[str, Any]:
        """Return detector metadata, architecture, and configuration."""
        pass
