from __future__ import annotations
from typing import Optional
import numpy as np
from sklearn.ensemble import IsolationForest as SKIsolationForest

from .base import BaseMetric
from ..types import Demo, DemoDataset


class IsolationForestMetric(BaseMetric):
    """Isolation Forest anomaly detection on action summary features.

    Requires fitting on a reference dataset. Returns the IF decision function
    value (positive for inliers/clean, negative for outliers/defective).
    """

    name = "isolation_forest"
    requires_fit = True

    def __init__(self, n_estimators: int = 100, contamination: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self._model: Optional[SKIsolationForest] = None

    def fit(self, dataset: DemoDataset) -> "IsolationForestMetric":
        """Fit on summary features of the dataset."""
        features = np.array([_extract_features(d) for d in dataset.demos])
        self._model = SKIsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(features)
        return self

    def score(self, demo: Demo) -> float:
        """Return IF decision function (higher = more normal = cleaner)."""
        if self._model is None:
            raise RuntimeError("IsolationForestMetric must be fit before scoring.")
        feat = _extract_features(demo).reshape(1, -1)
        return float(self._model.decision_function(feat)[0])


def _extract_features(demo: Demo) -> np.ndarray:
    """Extract action summary statistics as a fixed-length feature vector."""
    a = demo.actions
    mean = np.mean(a, axis=0)
    std = np.std(a, axis=0)
    rng = np.ptp(a, axis=0)
    return np.concatenate([mean, std, rng])
