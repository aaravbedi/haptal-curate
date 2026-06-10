from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

TRUNC_T: int = 324

@dataclass
class Demo:
    """Single robot demonstration."""
    obs: np.ndarray          # shape (T, obs_dim)
    actions: np.ndarray      # shape (T, act_dim)
    episode_length: int      # actual length before padding
    demo_id: str = ""
    phase_labels: Optional[np.ndarray] = None  # shape (T,), int phase ids

@dataclass
class DemoDataset:
    """Collection of demonstrations."""
    demos: List[Demo]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.demos)

    def __getitem__(self, idx: int) -> Demo:
        return self.demos[idx]

@dataclass
class ScoreResult:
    """Per-demo scores from one or more metrics."""
    scores: np.ndarray           # shape (N,), higher = cleaner
    demo_ids: List[str]
    metric_name: str
    per_metric_scores: Optional[Dict[str, np.ndarray]] = None
    episode_lengths: Optional[np.ndarray] = None  # shape (N,)

@dataclass
class CurationResult:
    """Output of curate()."""
    kept_indices: np.ndarray     # indices into original dataset
    removed_indices: np.ndarray
    fraction_kept: float
    strategy: str
    score_result: ScoreResult

@dataclass
class ConfoundReport:
    """Length-confound analysis result."""
    spearman_r: float
    p_value: float
    severity: str                # "none", "mild", "moderate", "severe"
    recommended_truncation_t: int
    message: str

@dataclass
class QualityReport:
    """Full quality report combining scores and curation."""
    score_result: ScoreResult
    curation_result: Optional[CurationResult]
    confound_report: Optional[ConfoundReport]
    summary: Dict[str, Any] = field(default_factory=dict)
