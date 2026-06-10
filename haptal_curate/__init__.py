"""haptal-curate: Quality scoring and curation for robot demonstration datasets."""

from .types import (
    Demo,
    DemoDataset,
    ScoreResult,
    CurationResult,
    ConfoundReport,
    QualityReport,
    TRUNC_T,
)
from .loader import load_demos
from .scorer import score
from .curator import curate
from .reporter import report

__version__ = "0.1.0"
__all__ = [
    "score",
    "curate",
    "report",
    "load_demos",
    "Demo",
    "DemoDataset",
    "ScoreResult",
    "CurationResult",
    "ConfoundReport",
    "QualityReport",
    "TRUNC_T",
]
