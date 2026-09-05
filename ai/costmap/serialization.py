"""Costmap serialization and deserialization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from ai.costmap.base import CostmapMetadata
from ai.costmap.bev_costmap import BEVTerrainCostmap


class CostmapSerializer:
    """Exports and imports BEVTerrainCostmap to compressed NPZ and JSON metadata."""

    @staticmethod
    def save(costmap: BEVTerrainCostmap, filepath_prefix: Union[str, Path]) -> Tuple[Path, Path]:
        """Save costmap arrays to .npz and metadata to .json.

        Returns:
            Tuple of (npz_path, json_path).
        """
        prefix = Path(filepath_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        npz_path = prefix.with_suffix(".npz")
        json_path = prefix.with_suffix(".json")

        # Save all numerical arrays into compressed binary archive
        np.savez_compressed(
            npz_path,
            cost=costmap.cost,
            traversability=costmap.traversability,
            obstacle=costmap.obstacle,
            hazard=costmap.hazard,
            unknown=costmap.unknown,
            confidence=costmap.confidence,
            slope_deg=costmap.slope_deg,
            roughness=costmap.roughness,
            height_var=costmap.height_var,
            observation_count=costmap.observation_count
        )

        # Save metadata and statistical breakdown
        meta_dict = costmap.metadata.to_dict()
        meta_dict["stats"] = costmap.summary_stats()

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        return npz_path, json_path

    @staticmethod
    def load(filepath_prefix: Union[str, Path]) -> BEVTerrainCostmap:
        """Load a BEVTerrainCostmap from .npz and .json files."""
        prefix = Path(filepath_prefix)
        npz_path = prefix.with_suffix(".npz")
        json_path = prefix.with_suffix(".json")

        if not npz_path.exists():
            raise FileNotFoundError(f"Costmap binary archive not found: {npz_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Costmap metadata file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        metadata = CostmapMetadata(
            resolution_m=float(meta_dict["resolution_m"]),
            shape=tuple(meta_dict["shape"]),
            x_range_m=tuple(meta_dict["x_range_m"]),
            y_range_m=tuple(meta_dict["y_range_m"]),
            frame_id=meta_dict.get("frame_id", "base_link"),
            timestamp=float(meta_dict.get("timestamp", 0.0)),
            coordinate_convention=meta_dict.get("coordinate_convention", "")
        )

        data = np.load(npz_path)

        return BEVTerrainCostmap(
            cost=data["cost"].astype(np.float32),
            traversability=data["traversability"].astype(np.float32),
            obstacle=data["obstacle"].astype(bool),
            hazard=data["hazard"].astype(np.float32),
            unknown=data["unknown"].astype(bool),
            confidence=data["confidence"].astype(np.float32),
            slope_deg=data["slope_deg"].astype(np.float32),
            roughness=data["roughness"].astype(np.float32),
            height_var=data["height_var"].astype(np.float32),
            observation_count=data["observation_count"].astype(np.int32),
            metadata=metadata
        )
