from __future__ import annotations
import numpy as np

from .types import ScoreResult, CurationResult


def curate(
    score_result: ScoreResult,
    fraction: float = 0.5,
    strategy: str = "top_fraction",
) -> CurationResult:
    """Select the highest-quality demonstrations from a scored dataset.

    Args:
        score_result: Output from score().
        fraction: Fraction of demos to keep (0.0 to 1.0).
        strategy: Curation strategy. Options:
            'top_fraction': keep the top-scoring fraction.
            'threshold': keep demos with score above median.

    Returns:
        CurationResult with kept and removed indices.
    """
    scores = score_result.scores
    n = len(scores)

    if strategy == "top_fraction":
        k = max(1, int(np.round(fraction * n)))
        sorted_indices = np.argsort(scores)[::-1]
        kept = sorted_indices[:k]
        removed = sorted_indices[k:]
    elif strategy == "threshold":
        threshold = float(np.median(scores))
        kept = np.where(scores >= threshold)[0]
        removed = np.where(scores < threshold)[0]
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    kept = np.sort(kept)
    removed = np.sort(removed)
    fraction_kept = len(kept) / n

    return CurationResult(
        kept_indices=kept,
        removed_indices=removed,
        fraction_kept=fraction_kept,
        strategy=strategy,
        score_result=score_result,
    )
