from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import rankdata

from .segmenter import Phase, extract_phase_windows
from ..metrics.base import BaseMetric
from ..types import Demo, DemoDataset


class PhaseGatedScorer:
    """Run a metric per phase and aggregate by rank normalization.

    Instead of scoring a demonstration end-to-end, this scorer splits each
    demo into PREGRASP, GRASP, and POST_CONTACT phases, scores each phase
    independently, then combines the per-phase scores.

    Strategies:
        best_per_phase: max rank-normalized score across phases.
        uniform: mean of rank-normalized per-phase scores.
        global: fall back to scoring the full demo (no phase gating).
    """

    def __init__(
        self,
        metric: BaseMetric,
        strategy: str = "uniform",
        gripper_dim: int = -1,
    ):
        """
        Args:
            metric: The quality metric to apply per phase.
            strategy: Aggregation strategy: 'best_per_phase', 'uniform', or 'global'.
            gripper_dim: Action dimension for gripper state detection.
        """
        self.metric = metric
        self.strategy = strategy
        self.gripper_dim = gripper_dim
        self._phase_demos: Optional[Dict[Phase, List[Demo]]] = None

    def fit(self, dataset: DemoDataset) -> "PhaseGatedScorer":
        """Fit the underlying metric on each phase's sub-demos."""
        if self.strategy == "global":
            self.metric.fit(dataset)
            return self

        # Build per-phase datasets for fitting
        phase_demos: Dict[Phase, List[Demo]] = {p: [] for p in Phase}
        for demo in dataset.demos:
            windows = extract_phase_windows(demo, gripper_dim=self.gripper_dim)
            for phase, indices in windows.items():
                if len(indices) > 0:
                    sub = _slice_demo(demo, indices)
                    phase_demos[phase].append(sub)

        self._phase_demos = phase_demos

        if self.metric.requires_fit:
            from ..types import DemoDataset as DS
            # Fit on the most populated phase
            all_demos = [d for demos in phase_demos.values() for d in demos]
            if all_demos:
                self.metric.fit(DS(demos=all_demos))

        return self

    def score_dataset(self, dataset: DemoDataset) -> np.ndarray:
        """Score all demos, returning rank-normalized combined scores."""
        if self.strategy == "global":
            raw = np.array([self.metric.score(d) for d in dataset.demos])
            return rankdata(raw) / len(raw)

        phases = list(Phase)
        per_phase_raw: Dict[Phase, np.ndarray] = {}
        for phase in phases:
            scores = []
            for demo in dataset.demos:
                windows = extract_phase_windows(demo, gripper_dim=self.gripper_dim)
                indices = windows[phase]
                if len(indices) > 0:
                    sub = _slice_demo(demo, indices)
                    scores.append(self.metric.score(sub))
                else:
                    scores.append(np.nan)
            per_phase_raw[phase] = np.array(scores)

        # Rank-normalize each phase independently
        per_phase_ranked: Dict[Phase, np.ndarray] = {}
        for phase, raw in per_phase_raw.items():
            valid = ~np.isnan(raw)
            ranked = np.zeros(len(raw))
            if valid.any():
                ranked[valid] = rankdata(raw[valid]) / valid.sum()
            per_phase_ranked[phase] = ranked

        # Aggregate
        stacked = np.stack(list(per_phase_ranked.values()), axis=1)  # (N, num_phases)
        if self.strategy == "best_per_phase":
            return np.max(stacked, axis=1)
        else:  # uniform
            return np.nanmean(stacked, axis=1)

    def score(self, demo: Demo) -> float:
        """Score a single demo (uses full demo if phase data unavailable)."""
        return float(self.metric.score(demo))


def _slice_demo(demo: Demo, indices: np.ndarray) -> Demo:
    """Create a sub-demo from selected timestep indices."""
    from ..types import Demo as D
    phase_labels = demo.phase_labels[indices] if demo.phase_labels is not None else None
    return D(
        obs=demo.obs[indices],
        actions=demo.actions[indices],
        episode_length=len(indices),
        demo_id=demo.demo_id,
        phase_labels=phase_labels,
    )
