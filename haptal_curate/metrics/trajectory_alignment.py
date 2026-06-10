from __future__ import annotations
from typing import Optional
import numpy as np

from .base import BaseMetric
from ..types import Demo, DemoDataset


class TrajectoryAlignmentMetric(BaseMetric):
    """Cosine similarity to the mean clean trajectory.

    Fits the mean action trajectory from a reference dataset, then scores
    each demo by cosine similarity to that mean. Higher = more aligned = cleaner.
    """

    name = "trajectory_alignment"
    requires_fit = True

    def __init__(self):
        self._mean_traj: Optional[np.ndarray] = None
        self._ref_len: int = 0

    def fit(self, dataset: DemoDataset) -> "TrajectoryAlignmentMetric":
        """Compute mean trajectory from the dataset."""
        # Align to shortest length
        min_len = min(d.episode_length for d in dataset.demos)
        trajs = np.array([d.actions[:min_len].flatten() for d in dataset.demos])
        self._mean_traj = np.mean(trajs, axis=0)
        self._ref_len = min_len
        return self

    def score(self, demo: Demo) -> float:
        """Return cosine similarity to the mean trajectory."""
        if self._mean_traj is None:
            raise RuntimeError("TrajectoryAlignmentMetric must be fit before scoring.")
        t = min(demo.episode_length, self._ref_len)
        traj = demo.actions[:t].flatten()
        ref = self._mean_traj[: t * demo.actions.shape[1]]
        # Cosine similarity
        norm_traj = np.linalg.norm(traj)
        norm_ref = np.linalg.norm(ref)
        if norm_traj < 1e-12 or norm_ref < 1e-12:
            return 0.0
        return float(np.dot(traj, ref) / (norm_traj * norm_ref))
