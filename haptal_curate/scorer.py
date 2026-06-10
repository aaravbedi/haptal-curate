from __future__ import annotations
from typing import List, Optional, Union
import numpy as np

from .types import Demo, DemoDataset, ScoreResult, TRUNC_T
from .metrics.base import BaseMetric
from .metrics.smoothness import SmoothnessMetric
from .metrics.gripper_timing import GripperTimingMetric
from .metrics.ensemble import EnsembleMetric
from .loader import load_demos


def score(
    hdf5_path_or_dataset: Union[str, DemoDataset],
    metrics: Optional[List[BaseMetric]] = None,
    truncate_t: Optional[int] = TRUNC_T,
    phase_gated: bool = False,
    fit_metrics: bool = True,
) -> ScoreResult:
    """Score demonstrations using the specified metrics.

    Args:
        hdf5_path_or_dataset: Path to HDF5 file or a DemoDataset.
        metrics: List of metric instances. Defaults to EnsembleMetric.
        truncate_t: Truncate demos to this length before scoring.
        phase_gated: Whether to use phase-gated scoring (requires PhaseGatedScorer).
        fit_metrics: Whether to fit fittable metrics on the dataset.

    Returns:
        ScoreResult with per-demo scores and metadata.
    """
    if isinstance(hdf5_path_or_dataset, str):
        dataset = load_demos(hdf5_path_or_dataset, truncate_t=truncate_t)
    else:
        dataset = hdf5_path_or_dataset

    if metrics is None:
        metrics = [EnsembleMetric()]

    if fit_metrics:
        for m in metrics:
            if m.requires_fit:
                m.fit(dataset)

    # Aggregate scores from all metrics (rank-normalize each, then average)
    all_scores = []
    per_metric: dict = {}
    for m in metrics:
        raw = np.array([m.score(d) for d in dataset.demos], dtype=float)
        per_metric[m.name] = raw
        all_scores.append(raw)

    if len(all_scores) == 1:
        final_scores = all_scores[0]
    else:
        # Rank-normalize each metric's scores then average
        from scipy.stats import rankdata
        ranked = [rankdata(s) / len(s) for s in all_scores]
        final_scores = np.mean(ranked, axis=0)

    return ScoreResult(
        scores=final_scores,
        demo_ids=[d.demo_id for d in dataset.demos],
        metric_name="+".join(m.name for m in metrics),
        per_metric_scores=per_metric,
        episode_lengths=np.array([d.episode_length for d in dataset.demos]),
    )
