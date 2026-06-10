from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union
import numpy as np

from .types import ScoreResult, CurationResult, QualityReport
from .confounds import detect_length_confound


def report(
    score_result: ScoreResult,
    curation_result: Optional[CurationResult] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Generate a quality report from scoring and curation results.

    Args:
        score_result: Output from score().
        curation_result: Output from curate() (optional).
        output_path: If provided, write the report as JSON to this path.

    Returns:
        Dictionary with summary statistics and quality report.
    """
    scores = score_result.scores
    n = len(scores)

    confound_report = None
    if score_result.episode_lengths is not None:
        confound_report = detect_length_confound(scores, score_result.episode_lengths)

    summary: dict = {
        "n_demos": n,
        "metric": score_result.metric_name,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "top5_demo_ids": [score_result.demo_ids[i] for i in np.argsort(scores)[::-1][:5]],
        "bottom5_demo_ids": [score_result.demo_ids[i] for i in np.argsort(scores)[:5]],
    }

    if confound_report is not None:
        summary["confound"] = {
            "severity": confound_report.severity,
            "spearman_r": confound_report.spearman_r,
            "message": confound_report.message,
        }

    if curation_result is not None:
        summary["curation"] = {
            "strategy": curation_result.strategy,
            "fraction_kept": curation_result.fraction_kept,
            "n_kept": int(len(curation_result.kept_indices)),
            "n_removed": int(len(curation_result.removed_indices)),
        }

    quality_report = QualityReport(
        score_result=score_result,
        curation_result=curation_result,
        confound_report=confound_report,
        summary=summary,
    )

    if output_path is not None:
        Path(output_path).write_text(json.dumps(summary, indent=2))

    return summary
