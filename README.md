---
title: haptal-curate
emoji: 🤖
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
---

# haptal-curate

Quality scoring and curation for robot demonstration datasets.

haptal-curate implements seven audited quality metrics for imitation learning datasets, backed by findings from arXiv:2606.03134, arXiv:2606.05588, and arXiv:2606.10229. Given a dataset of robot demonstrations (e.g. LIBERO HDF5 files), the library scores each demo for quality and returns a curated subset, helping practitioners improve policy performance by removing defective demonstrations before training.

## Installation

```bash
pip install haptal-curate
```

## Quickstart

```python
from haptal_curate import score, curate, report

# Score all demos in a LIBERO HDF5
result = score("demos.hdf5", truncate_t=324)

# Keep the top 80%
curation = curate(result, fraction=0.8)

# Print a summary report
summary = report(result, curation)
print(f"Kept {summary['curation']['n_kept']} / {summary['n_demos']} demos")
print(f"Top demos: {summary['top5_demo_ids']}")
```

## The Length Confound Warning

A critical finding from arXiv:2606.10229: without truncating trajectories to a fixed length, five of seven quality metrics achieve AUROC ≈ 1.000 on LIBERO datasets — not because they are good metrics, but because defective demonstrations (which contain silent manipulation failures such as premature gripper release, characterized in arXiv:2606.03134) tend to run the full 500-step horizon while clean demonstrations finish in ~325 steps. Any length-sensitive feature trivially separates the two classes. Truncating all demos to **TRUNC_T = 324** timesteps removes this confound and reveals an honest performance ceiling of ~0.91 AUROC. **Always set `truncate_t=324` (the default) when scoring LIBERO data.**

## Available Metrics

| Metric | Class | Description | Requires fit | Paper |
|---|---|---|---|---|
| `smoothness` | `SmoothnessMetric` | SPARC spectral arc length of action speed profile | No | arXiv:2606.05588 |
| `entropy` | `EntropyMetric` | Negative action standard deviation | No | arXiv:2606.05588 |
| `gripper_timing` | `GripperTimingMetric` | Normalized first-open timestep (early = defective) | No | arXiv:2606.03134, arXiv:2606.10229 |
| `isolation_forest` | `IsolationForestMetric` | Isolation Forest on action summary features | Yes | arXiv:2606.05588 |
| `knn` | `KNNMetric` | Negative mean kNN distance to reference demos | Yes | arXiv:2606.05588 |
| `trajectory_alignment` | `TrajectoryAlignmentMetric` | Cosine similarity to dataset mean trajectory | Yes | arXiv:2606.05588 |
| `ensemble` | `EnsembleMetric` | Weighted combination (default: 0.5 smoothness + 0.5 gripper_timing) | No | arXiv:2606.10229 |

## Phase-Gated Curation

Phase-gated scoring runs each metric independently on PREGRASP, GRASP, and POST_CONTACT phases of each demonstration, then aggregates by rank normalization. This disentangles phase-specific quality signals and avoids penalizing naturally short phases.

```python
from haptal_curate import load_demos
from haptal_curate.metrics import SmoothnessMetric
from haptal_curate.phase import PhaseGatedScorer

dataset = load_demos("demos.hdf5", truncate_t=324)
scorer = PhaseGatedScorer(SmoothnessMetric(), strategy="uniform")
scorer.fit(dataset)
scores = scorer.score_dataset(dataset)
```

## Citation

```bibtex
@article{bedi2025failures,
  title={Silent Manipulation Failures in Robot Demonstration Datasets},
  author={Bedi, Aarav},
  journal={arXiv preprint arXiv:2606.03134},
  year={2025}
}

@article{bedi2025audit,
  title={Auditing Curation Metrics for Robot Demonstration Datasets},
  author={Bedi, Aarav},
  journal={arXiv preprint arXiv:2606.05588},
  year={2025}
}

@article{bedi2025confound,
  title={What Demonstration Curation Metrics Do to Your Policy},
  author={Bedi, Aarav},
  journal={arXiv preprint arXiv:2606.10229},
  year={2025}
}
```
