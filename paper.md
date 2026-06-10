---
title: 'haptal-curate: A Python Library for Robot Demonstration Quality Scoring and Curation'
tags:
  - robotics
  - imitation learning
  - data quality
  - behavior cloning
  - demonstration curation
authors:
  - name: Aarav Bedi
    affiliation: 1
affiliations:
  - name: HaptalAI; University of California, Berkeley
    index: 1
date: 9 June 2026
bibliography: paper.bib
---

# Summary

Behavior cloning policies are only as good as the demonstrations they train on.
Large demonstration datasets collected in the wild routinely contain defective
episodes — premature gripper releases, failed grasps, incorrect placements —
that corrupt the training distribution and degrade policy performance.
`haptal-curate` is a Python library that scores each demonstration in a dataset
for quality, detects methodological confounds in the scoring process, and
returns a curated subset for downstream training. It implements seven audited
quality metrics drawn from the imitation learning literature, a phase-gated
scoring mode that applies metrics within task phases rather than globally, and
an automatic episode-length confound detector. The library requires only NumPy,
SciPy, scikit-learn, and h5py — no simulator or robot hardware dependency — and
ships with a Gradio web interface for interactive use.

# Statement of Need

The importance of demonstration quality for imitation learning is well
established [@mandlekar2021]. Large-scale datasets such as Open X-Embodiment
[@oxe2023] aggregate demonstrations across operators and robots, introducing
quality heterogeneity at scale. Despite this, no standard tooling exists for
scoring and curating demonstration datasets before training.

Existing quality metrics — trajectory smoothness [@sparc], isolation forests
[@isoforest], nearest-neighbor distance — are each validated under their own
protocols and are not available through a unified interface. More critically,
recent work has shown that these metrics are routinely evaluated on a flawed
axis. When defective demonstrations run to the episode time limit while clean
demonstrations terminate early, any length-sensitive metric achieves near-perfect
detection accuracy not by identifying harmful content, but by measuring episode
length [@bedi2025confound]. The same work demonstrates that detection accuracy
and curation value are sharply decoupled: the metric with the highest detection
AUROC (0.804) produces the worst downstream policy (13.3% task success), while
a metric with substantially lower AUROC (0.638) nearly matches oracle performance
[@bedi2025confound]. Prior auditing work further shows that global metrics fail
systematically on structural defects localized to specific task phases
[@bedi2025audit], and that silent manipulation failures are difficult to detect
from proprioception and vision alone [@bedi2025failures].

`haptal-curate` addresses these gaps by providing a unified, simulator-free
interface for demonstration quality scoring, exposing the length confound as a
first-class diagnostic, and implementing phase-gated scoring as a principled
alternative to global metric application.

# Installation and Usage

```bash
pip install haptal-curate
```

```python
from haptal_curate import score, curate, report

# score all demos in a LIBERO HDF5 (truncates to 324 steps by default)
result = score("demos.hdf5", truncate_t=324)

# keep the top 75%
curation = curate(result, fraction=0.75)

# print summary
summary = report(result, curation)
print(f"kept {summary['curation']['n_kept']} / {summary['n_demos']} demos")
```

# Key Algorithmic Contributions

**Episode-length confound detection.** `detect_length_confound()` computes the
Spearman rank correlation between each metric's quality scores and raw episode
lengths across the dataset. A high correlation indicates the metric is partially
exploiting length as a proxy signal rather than measuring demonstration content.
The function returns a severity level (none / moderate / severe) and a
recommended truncation timestep. This diagnostic catches the confound documented
in [@bedi2025confound], where five of seven metrics achieved AUROC 1.000 before
truncation on a LIBERO pick-and-place benchmark.

**Phase-gated scoring.** `PhaseGatedScorer` segments each trajectory into
PREGRASP, GRASP, and POST\_CONTACT windows using gripper state transitions, runs
a configurable metric on each window independently, and aggregates scores by
rank normalization. This is scale-invariant across phases of different lengths
and prevents a metric tuned for grasp-phase signals from dominating the global
score during approach or transport. Phase labels from the HDF5 file are used
when available; gripper state heuristics are applied otherwise.

**Unified metric interface.** All seven metrics implement a common
`BaseMetric` interface with `.fit(dataset)` and `.score(demo)` methods,
making it straightforward to add new metrics or swap them in curation
pipelines without changing downstream code.

# Tests

The test suite contains 39 tests across four files, all using synthetic NumPy
data with no simulator dependency. `test_metrics.py` verifies direction-of-effect
for all seven metrics on synthetic clean versus defective demonstrations.
`test_confounds.py` checks that the confound detector fires on correlated
synthetic data and is silent on uncorrelated data. `test_phase.py` tests phase
segmentation and all three `PhaseGatedScorer` strategies. `test_pipeline.py`
runs the full end-to-end pipeline — synthetic HDF5 generation, scoring,
curation, and report generation — and checks output schema correctness.
All 39 tests pass on Python 3.9 through 3.12.

# References
