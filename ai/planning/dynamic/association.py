"""Data association algorithms for multi-frame dynamic obstacle tracking.

Provides modular association interfaces and optimal bipartite matching
(Hungarian algorithm with spatial gating) to associate incoming detections
with active tracks in the world ('odom') frame.
"""

from __future__ import annotations

import abc
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


class BaseAssociation(abc.ABC):
    """Abstract base class for frame-to-frame detection-to-track association."""

    @abc.abstractmethod
    def associate(
        self,
        track_positions: np.ndarray,
        detection_positions: np.ndarray,
        max_distance_m: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections with active tracks under spatial distance gating.

        Args:
            track_positions: (N, 2) array of (x, y) coordinates for N tracks.
            detection_positions: (M, 2) array of (x, y) coordinates for M detections.
            max_distance_m: Maximum Euclidean distance gate in meters. Pairs with
                            distance > max_distance_m must not be matched.

        Returns:
            Tuple of:
            - matches: List of (track_idx, detection_idx) integer index pairs.
            - unmatched_tracks: List of track indices not associated.
            - unmatched_detections: List of detection indices not associated.
        """
        pass


class HungarianAssociation(BaseAssociation):
    """Optimal 1-to-1 bipartite matching using Hungarian algorithm with Euclidean distance gating."""

    def __init__(self, fallback_to_greedy: bool = False) -> None:
        self.fallback_to_greedy = fallback_to_greedy

    def compute_cost_matrix(
        self,
        track_positions: np.ndarray,
        detection_positions: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise Euclidean distance cost matrix between tracks and detections."""
        n_tracks = len(track_positions)
        n_dets = len(detection_positions)
        if n_tracks == 0 or n_dets == 0:
            return np.zeros((n_tracks, n_dets), dtype=np.float64)

        # Vectorized 2D Euclidean distance
        diff = track_positions[:, np.newaxis, :] - detection_positions[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=2)

    def associate(
        self,
        track_positions: np.ndarray,
        detection_positions: np.ndarray,
        max_distance_m: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Perform optimal assignment with spatial gating."""
        if max_distance_m <= 0.0:
            raise ValueError(f"max_distance_m must be positive, got {max_distance_m}")

        n_tracks = len(track_positions)
        n_dets = len(detection_positions)

        # Fast exits for empty sets
        if n_tracks == 0:
            return [], [], list(range(n_dets))
        if n_dets == 0:
            return [], list(range(n_tracks)), []

        track_positions = np.asarray(track_positions, dtype=np.float64)
        detection_positions = np.asarray(detection_positions, dtype=np.float64)

        cost_matrix = self.compute_cost_matrix(track_positions, detection_positions)

        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(n_tracks))
        unmatched_dets = set(range(n_dets))

        if _HAS_SCIPY and not self.fallback_to_greedy:
            # Optimal Hungarian Assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                dist = cost_matrix[r, c]
                if dist <= max_distance_m:
                    matches.append((int(r), int(c)))
                    unmatched_tracks.discard(int(r))
                    unmatched_dets.discard(int(c))
        else:
            # Deterministic greedy matching fallback
            flat_indices = np.argsort(cost_matrix, axis=None)
            for idx in flat_indices:
                r = int(idx // n_dets)
                c = int(idx % n_dets)
                if cost_matrix[r, c] > max_distance_m:
                    break
                if r in unmatched_tracks and c in unmatched_dets:
                    matches.append((r, c))
                    unmatched_tracks.discard(r)
                    unmatched_dets.discard(c)

        # Sort for deterministic output
        matches.sort(key=lambda m: (m[0], m[1]))
        return matches, sorted(list(unmatched_tracks)), sorted(list(unmatched_dets))
