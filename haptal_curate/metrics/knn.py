from __future__ import annotations
from typing import Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseMetric
from ..types import Demo, DemoDataset


class KNNMetric(BaseMetric):
    """Negative mean k-nearest-neighbor distance metric.

    Scores demos by proximity to the fitted reference set. Demos close to
    the reference distribution score higher (less negative distance).
    """

    name = "knn"
    requires_fit = True

    def __init__(self, k: int = 5):
        self.k = k
        self._nn: Optional[NearestNeighbors] = None

    def fit(self, dataset: DemoDataset) -> "KNNMetric":
        """Fit kNN index on action summary features."""
        features = np.array([_extract_features(d) for d in dataset.demos])
        self._nn = NearestNeighbors(n_neighbors=min(self.k, len(features)))
        self._nn.fit(features)
        return self

    def score(self, demo: Demo) -> float:
        """Return negative mean kNN distance (higher = closer = cleaner)."""
        if self._nn is None:
            raise RuntimeError("KNNMetric must be fit before scoring.")
        feat = _extract_features(demo).reshape(1, -1)
        dists, _ = self._nn.kneighbors(feat)
        return float(-np.mean(dists))


def _extract_features(demo: Demo) -> np.ndarray:
    a = demo.actions
    mean = np.mean(a, axis=0)
    std = np.std(a, axis=0)
    return np.concatenate([mean, std])
