"""Vision pipeline configuration schema and loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple
import yaml
from pydantic import BaseModel, Field

class InputConfig(BaseModel):
    width: int = Field(default=640, description="Expected frame width")
    height: int = Field(default=480, description="Expected frame height")
    color_format: str = Field(default="BGR", description="Default capture color format")
    fps: int = Field(default=30, description="Nominal frame rate")

class ResizeConfig(BaseModel):
    enabled: bool = Field(default=True)
    target_width: int = Field(default=640)
    target_height: int = Field(default=480)
    interpolation: str = Field(default="linear")

class CropConfig(BaseModel):
    enabled: bool = Field(default=False)
    top_pixels: int = Field(default=0)
    bottom_pixels: int = Field(default=0)
    left_pixels: int = Field(default=0)
    right_pixels: int = Field(default=0)

class ColorConversionConfig(BaseModel):
    source_format: str = Field(default="BGR")
    target_format: str = Field(default="RGB")

class CLAHEConfig(BaseModel):
    enabled: bool = Field(default=True)
    clip_limit: float = Field(default=2.0)
    tile_grid_size: List[int] = Field(default_factory=lambda: [8, 8])

class DenoisingConfig(BaseModel):
    enabled: bool = Field(default=True)
    method: str = Field(default="gaussian") # 'gaussian', 'bilateral', 'median', 'none'
    kernel_size: int = Field(default=3)
    sigma: float = Field(default=1.0)

class NormalizationConfig(BaseModel):
    enabled: bool = Field(default=True)
    range_min: float = Field(default=0.0)
    range_max: float = Field(default=1.0)
    dtype: str = Field(default="float32") # 'float32' or 'uint8'

class PreprocessingConfig(BaseModel):
    resize: ResizeConfig = Field(default_factory=ResizeConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    color_conversion: ColorConversionConfig = Field(default_factory=ColorConversionConfig)
    clahe: CLAHEConfig = Field(default_factory=CLAHEConfig)
    denoising: DenoisingConfig = Field(default_factory=DenoisingConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)

class ProcessingConfig(BaseModel):
    baseline_processor: str = Field(default="canny_edge")
    canny_low_threshold: int = Field(default=50)
    canny_high_threshold: int = Field(default=150)

class VisualizationConfig(BaseModel):
    window_title: str = Field(default="SIH26126 UGV Vision Pipeline")
    side_by_side: bool = Field(default=True)
    display_fps: bool = Field(default=True)
    display_latency: bool = Field(default=True)
    hud_font_scale: float = Field(default=0.6)
    hud_bg_opacity: float = Field(default=0.6)

class TimingConfig(BaseModel):
    fps_window_size: int = Field(default=30)
    percentiles: List[int] = Field(default_factory=lambda: [50, 95])

class VisionConfig(BaseModel):
    input: InputConfig = Field(default_factory=InputConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)

def load_vision_config(config_path: Optional[str] = None) -> VisionConfig:
    """Load vision configuration from YAML file or return defaults.
    
    Args:
        config_path: Optional path to vision_config.yaml.
        
    Returns:
        Validated VisionConfig instance.
    """
    if config_path is None:
        # Default discovery
        root = Path(__file__).resolve().parent.parent.parent
        default_file = root / "configs" / "vision_config.yaml"
        if default_file.exists():
            config_path = str(default_file)

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return VisionConfig(**data)

    return VisionConfig()
