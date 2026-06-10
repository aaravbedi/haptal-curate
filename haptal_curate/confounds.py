from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr

from .types import ConfoundReport, TRUNC_T


def detect_length_confound(
    scores: np.ndarray,
    episode_lengths: np.ndarray,
    severity_thresholds: tuple = (0.3, 0.5, 0.7),
) -> ConfoundReport:
    """Detect whether scores are confounded by episode length.

    As shown in arXiv:2606.10229, length-sensitive metrics achieve AUROC≈1.0
    simply because defective demonstrations run the full horizon while clean
    demos finish early. Truncating to TRUNC_T removes this confound.

    Args:
        scores: Quality scores, shape (N,).
        episode_lengths: Episode lengths, shape (N,).
        severity_thresholds: (mild, moderate, severe) |r| thresholds.

    Returns:
        ConfoundReport with severity assessment and truncation recommendation.
    """
    r, p = spearmanr(scores, episode_lengths)
    r = float(r)
    p = float(p)
    abs_r = abs(r)

    mild, moderate, severe = severity_thresholds
    if abs_r < mild:
        severity = "none"
        msg = "No significant length confound detected."
    elif abs_r < moderate:
        severity = "mild"
        msg = f"Mild length confound (|r|={abs_r:.2f}). Consider truncating at TRUNC_T={TRUNC_T}."
    elif abs_r < severe:
        severity = "moderate"
        msg = f"Moderate length confound (|r|={abs_r:.2f}). Recommend truncating at TRUNC_T={TRUNC_T}."
    else:
        severity = "severe"
        msg = (
            f"Severe length confound (|r|={abs_r:.2f}). Scores may be driven purely by "
            f"episode length. Truncate to TRUNC_T={TRUNC_T} before scoring."
        )

    # Recommend truncation at the median length of shorter demos
    median_short = int(np.percentile(episode_lengths, 50))
    recommended_t = min(median_short, TRUNC_T)

    return ConfoundReport(
        spearman_r=r,
        p_value=p,
        severity=severity,
        recommended_truncation_t=recommended_t,
        message=msg,
    )
