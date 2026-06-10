from __future__ import annotations
import numpy as np
from .base import BaseMetric
from ..types import Demo


class EntropyMetric(BaseMetric):
    """Negative action standard deviation metric.

    Low-variance (predictable) action sequences score higher.
    Higher score = more consistent = cleaner.
    """

    name = "entropy"
    requires_fit = False

    def score(self, demo: Demo) -> float:
        """Return negative mean std across action dimensions."""
        return float(-np.mean(np.std(demo.actions, axis=0)))
