from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from ..types import Demo, DemoDataset


class BaseMetric(ABC):
    """Abstract base for demonstration quality metrics.

    All metrics follow the convention: higher score = cleaner demo.
    """

    name: str = "base"
    requires_fit: bool = False

    def fit(self, dataset: DemoDataset) -> "BaseMetric":
        """Fit the metric on a reference dataset (no-op for standalone metrics)."""
        return self

    @abstractmethod
    def score(self, demo: Demo) -> float:
        """Score a single demonstration. Higher = cleaner."""
        ...

    def score_dataset(self, dataset: DemoDataset) -> list:
        """Score all demos in a dataset."""
        return [self.score(d) for d in dataset.demos]
