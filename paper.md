---
title: 'haptal-curate: Quality Scoring and Curation for Robot Demonstration Datasets'
tags:
  - Python
  - robotics
  - imitation learning
  - dataset curation
  - quality metrics
authors:
  - name: Aarav Bedi
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Haptal AI
    index: 1
date: 2026-06-10
bibliography: paper.bib
---

# Summary

`haptal-curate` is a Python package for automated quality scoring and curation
of robot demonstration datasets used in imitation learning. It implements a
modular suite of quality metrics, a phase-gated scoring framework, and
length-confound detection, enabling practitioners to identify and remove
low-quality demonstrations before training behavioral cloning or other
imitation learning policies.

The package addresses a critical but often overlooked problem in robot learning:
many existing dataset curation metrics achieve high discriminative accuracy on
standard benchmarks not by detecting genuine quality differences, but by
exploiting a length confound — defective demonstrations tend to run to the full
episode horizon while clean ones terminate early. `haptal-curate` detects this
confound automatically and recommends per-episode truncation to eliminate it,
following the methodology of @haptal2026confound.

# Statement of Need

Large-scale robot demonstration datasets such as LIBERO [@liu2023libero]
frequently contain demonstrations of varying quality: some are clean and task-
completing, while others contain premature gripper releases, jittery motions,
or early terminations. Imitation learning policies trained on uncurated datasets
exhibit degraded performance relative to those trained on curated subsets
[@mandlekar2021matters].

Existing curation approaches are often dataset-specific, use heuristic rules,
or are evaluated under confounded conditions. `haptal-curate` provides:

1. **A confound-aware evaluation framework** that detects and corrects for
   length-based scoring artifacts.
2. **A modular metric library** including spectral smoothness (SPARC), gripper
   timing, distribution-based metrics (Isolation Forest, k-NN), and a
   composable ensemble.
3. **Phase-gated scoring** that evaluates each phase of a manipulation
   trajectory (pre-grasp, grasp, post-contact) independently.
4. **A simple three-step API** (`score → curate → report`) that integrates
   directly with HDF5 dataset formats used by LIBERO and RoboSuite [@zhu2020robosuite].

# Implementation

## Quality Metrics

`haptal-curate` includes seven quality metrics, all following the convention
that higher scores indicate cleaner demonstrations:

- **SmoothnessMetric**: Computes the SPARC (Spectral Arc Length) of the
  end-effector speed profile [@balasubramanian2015sparc]. Smooth, coordinated
  motions have less negative SPARC values.
- **GripperTimingMetric**: Measures the normalized timestep at which the gripper
  first opens. Demonstrations with premature release score low.
- **EntropyMetric**: Uses negative mean action standard deviation as a proxy for
  demonstration predictability.
- **IsolationForestMetric**: Fits a scikit-learn Isolation Forest
  [@pedregosa2011sklearn] on action summary statistics and uses the decision
  function as an inlier score.
- **KNNMetric**: Scores by negative mean distance to the k nearest neighbors
  in the reference dataset, computed on action statistics.
- **TrajectoryAlignmentMetric**: Measures cosine similarity to the mean
  trajectory of the reference set.
- **EnsembleMetric**: Rank-normalizes and combines smoothness and gripper
  timing with configurable weights (default: equal weight).

## Phase-Gated Scoring

The `PhaseGatedScorer` class decomposes each demonstration into three phases
using gripper state transitions or explicit phase labels:
PREGRASP (before first close), GRASP (while gripper is closed), and
POST_CONTACT (after first re-open). Each phase is scored independently and
the scores are combined via rank normalization using one of three strategies:
`uniform` (mean), `best_per_phase` (max), or `global` (full-demo scoring).

## Length-Confound Detection

The `detect_length_confound` function computes the Spearman rank correlation
between quality scores and episode lengths. Correlation magnitude is classified
into four severity levels (none, mild, moderate, severe) and a recommended
truncation horizon is returned. The default truncation `TRUNC_T=324` was chosen
based on the median clean demonstration length in LIBERO-90.

# Example Usage

```python
import haptal_curate as hc

# Score an HDF5 dataset (truncates to 324 steps to remove length confound)
score_result = hc.score("demos.hdf5")

# Keep the top 50% by quality
curation_result = hc.curate(score_result, fraction=0.5)

# Generate a report with confound analysis
summary = hc.report(score_result, curation_result)
```

# Acknowledgements

This work was developed at Haptal AI. The SPARC smoothness metric is based on
@balasubramanian2015sparc. The length-confound analysis methodology follows
the evaluation framework introduced in @haptal2026confound.

# References
