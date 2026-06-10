# haptal-curate

**Quality scoring and curation for robot demonstration datasets.**

haptal-curate provides a modular toolkit for detecting and removing low-quality
demonstrations from imitation learning datasets. It implements the confound-free
evaluation methodology described in [arXiv:2606.10229](https://arxiv.org/abs/2606.10229),
with built-in length-confound detection, phase-gated scoring, and a composable
metric ensemble.

## Features

- **Multiple quality metrics** — smoothness (SPARC), gripper timing, entropy,
  isolation forest, k-NN distance, trajectory alignment, and a weighted ensemble.
- **Phase-gated scoring** — split demonstrations into PREGRASP / GRASP /
  POST_CONTACT phases and score each independently.
- **Length-confound detection** — Spearman-rank test with automatic severity
  classification and truncation recommendation (truncation at T=324 by default).
- **HDF5 loader** — reads LIBERO-format demonstration files directly.
- **Simple API** — `score()` → `curate()` → `report()`.

## Installation

```bash
pip install haptal-curate
```

For development:

```bash
git clone https://github.com/haptal-ai/haptal-curate
cd haptal-curate
pip install -e ".[dev]"
```

## Quick Start

```python
import haptal_curate as hc

# Score from an HDF5 file
score_result = hc.score("path/to/demos.hdf5")

# Keep the top 50% by quality
curation_result = hc.curate(score_result, fraction=0.5)

# Print a summary report
summary = hc.report(score_result, curation_result)
print(summary)
```

### Using a DemoDataset directly

```python
import numpy as np
from haptal_curate.types import Demo, DemoDataset
import haptal_curate as hc

demos = [
    Demo(obs=np.zeros((100, 9)), actions=np.zeros((100, 7)),
         episode_length=100, demo_id=f"demo_{i}")
    for i in range(20)
]
dataset = DemoDataset(demos=demos)

score_result = hc.score(dataset)
curation_result = hc.curate(score_result, fraction=0.7)
```

### Phase-gated scoring

```python
from haptal_curate.phase import PhaseGatedScorer
from haptal_curate.metrics import SmoothnessMetric

scorer = PhaseGatedScorer(SmoothnessMetric(), strategy="uniform")
scorer.fit(dataset)
scores = scorer.score_dataset(dataset)
```

## Metrics

| Metric | Description | Requires fit |
|--------|-------------|:---:|
| `SmoothnessMetric` | SPARC spectral arc length of action speed profile | no |
| `EntropyMetric` | Negative mean action std (low variance = cleaner) | no |
| `GripperTimingMetric` | Normalized timestep of first gripper open event | no |
| `IsolationForestMetric` | Isolation Forest on action summary features | yes |
| `KNNMetric` | Negative mean k-NN distance to reference set | yes |
| `TrajectoryAlignmentMetric` | Cosine similarity to mean clean trajectory | yes |
| `EnsembleMetric` | Weighted rank-normalized combination | no |

## API Reference

### `score(hdf5_path_or_dataset, metrics=None, truncate_t=324, fit_metrics=True)`

Score demonstrations. Returns a `ScoreResult`.

### `curate(score_result, fraction=0.5, strategy="top_fraction")`

Select the highest-quality subset. Strategies: `top_fraction`, `threshold`.
Returns a `CurationResult`.

### `report(score_result, curation_result=None, output_path=None)`

Generate a quality summary dict (optionally writes JSON). Includes length-confound
analysis via Spearman rank correlation.

### `load_demos(hdf5_path, truncate_t=324, max_demos=None)`

Load demonstrations from an HDF5 file. Returns a `DemoDataset`.

## Background

Length-biased metrics achieve near-perfect AUROC on LIBERO not because they detect
quality, but because defective demonstrations run to the full episode horizon while
clean ones terminate early. Truncating to a fixed length (TRUNC_T=324 steps by
default) eliminates this confound. haptal-curate detects and warns about this
confound automatically via `ConfoundReport`.

## Citation

If you use haptal-curate in your research, please cite:

```bibtex
@software{haptal_curate_2026,
  author    = {Bedi, Aarav},
  title     = {haptal-curate: Quality Scoring and Curation for Robot Demonstration Datasets},
  year      = {2026},
  url       = {https://github.com/haptal-ai/haptal-curate},
  version   = {0.1.0}
}
```

## License

MIT
