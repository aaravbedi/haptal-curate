from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from scipy.stats import rankdata

from .base import BaseMetric
from .smoothness import SmoothnessMetric
from .gripper_timing import GripperTimingMetric
from ..types import Demo, DemoDataset


class EnsembleMetric(BaseMetric):
    """Weighted combination of quality metrics.

    Default: 0.5 * smoothness + 0.5 * gripper_timing (rank-normalized).
    Scores are rank-normalized before weighting to put them on a common scale.
    """

    name = "ensemble"
    requires_fit = False

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Dict mapping metric name to weight. Defaults to
                     {'smoothness': 0.5, 'gripper_timing': 0.5}.
        """
        if weights is None:
            weights = {"smoothness": 0.5, "gripper_timing": 0.5}
        self.weights = weights
        self._metrics: Dict[str, BaseMetric] = {
            "smoothness": SmoothnessMetric(),
            "gripper_timing": GripperTimingMetric(),
        }
        self._dataset_scores: Optional[Dict[str, np.ndarray]] = None
        self._demos: Optional[list] = None

    def fit(self, dataset: DemoDataset) -> "EnsembleMetric":
        """Precompute dataset-wide scores for rank normalization."""
        self._demos = dataset.demos
        self._dataset_scores = {}
        for name, metric in self._metrics.items():
            if name in self.weights:
                metric.fit(dataset)
                raw = np.array([metric.score(d) for d in dataset.demos])
                # Rank normalize to [0, 1]
                ranked = rankdata(raw) / len(raw)
                self._dataset_scores[name] = ranked
        return self

    def score(self, demo: Demo) -> float:
        """Return weighted rank-normalized ensemble score."""
        if self._dataset_scores is None or self._demos is None:
            # Fall back to raw scores without rank normalization
            total, weight_sum = 0.0, 0.0
            for name, metric in self._metrics.items():
                w = self.weights.get(name, 0.0)
                if w > 0:
                    total += w * metric.score(demo)
                    weight_sum += w
            return total / weight_sum if weight_sum > 0 else 0.0

        # Find demo index if it's in the fitted dataset
        try:
            idx = self._demos.index(demo)
            total, weight_sum = 0.0, 0.0
            for name, scores in self._dataset_scores.items():
                w = self.weights.get(name, 0.0)
                total += w * scores[idx]
                weight_sum += w
            return total / weight_sum if weight_sum > 0 else 0.0
        except ValueError:
            # Demo not in fitted set — score directly
            total, weight_sum = 0.0, 0.0
            for name, metric in self._metrics.items():
                w = self.weights.get(name, 0.0)
                if w > 0:
                    total += w * metric.score(demo)
                    weight_sum += w
            return total / weight_sum if weight_sum > 0 else 0.0
