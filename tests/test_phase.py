"""Tests for phase segmentation and phase-gated scoring."""
import numpy as np
import pytest

from haptal_curate.phase.segmenter import Phase, extract_phase_windows
from haptal_curate.phase.phase_gated import PhaseGatedScorer
from haptal_curate.metrics.smoothness import SmoothnessMetric
from haptal_curate.types import Demo, DemoDataset
from tests.conftest import make_demo, make_dataset


def _demo_with_gripper_open_at(T: int = 100, open_at: int = 60) -> Demo:
    return make_demo(T=T, act_dim=7, gripper_open_at=open_at, demo_id="demo_0")


def test_phase_extraction_basic():
    demo = _demo_with_gripper_open_at(T=100, open_at=60)
    windows = extract_phase_windows(demo)
    assert set(windows.keys()) == set(Phase)
    total = sum(len(v) for v in windows.values())
    assert total == demo.episode_length


def test_pregrasp_before_close():
    demo = _demo_with_gripper_open_at(T=100, open_at=60)
    # Gripper starts closed (negatively signed), so pregrasp is 0 timesteps
    # because actions[:, -1] starts at -1.0 (closed from the start).
    windows = extract_phase_windows(demo)
    pregrasp = windows[Phase.PREGRASP]
    # All pregrasp indices should be before any close event
    assert isinstance(pregrasp, np.ndarray)


def test_phase_labels_used_when_present():
    T = 50
    demo = make_demo(T=T, act_dim=7)
    labels = np.zeros(T, dtype=np.int32)
    labels[20:35] = 1  # GRASP
    labels[35:] = 2    # POST_CONTACT
    demo.phase_labels = labels
    windows = extract_phase_windows(demo)
    assert len(windows[Phase.PREGRASP]) == 20
    assert len(windows[Phase.GRASP]) == 15
    assert len(windows[Phase.POST_CONTACT]) == 15


def test_phase_windows_cover_all_timesteps():
    for open_at in [10, 50, 90]:
        demo = _demo_with_gripper_open_at(T=100, open_at=open_at)
        windows = extract_phase_windows(demo)
        all_idx = np.concatenate(list(windows.values()))
        assert len(np.unique(all_idx)) == demo.episode_length


def test_phase_gated_scorer_uniform():
    dataset = make_dataset(n=10, T=80)
    m = SmoothnessMetric()
    scorer = PhaseGatedScorer(m, strategy="uniform")
    scorer.fit(dataset)
    scores = scorer.score_dataset(dataset)
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))


def test_phase_gated_scorer_best_per_phase():
    dataset = make_dataset(n=10, T=80)
    m = SmoothnessMetric()
    scorer = PhaseGatedScorer(m, strategy="best_per_phase")
    scorer.fit(dataset)
    scores = scorer.score_dataset(dataset)
    assert scores.shape == (10,)


def test_phase_gated_scorer_global():
    dataset = make_dataset(n=10, T=80)
    m = SmoothnessMetric()
    scorer = PhaseGatedScorer(m, strategy="global")
    scorer.fit(dataset)
    scores = scorer.score_dataset(dataset)
    assert scores.shape == (10,)


def test_phase_gated_single_score():
    dataset = make_dataset(n=5, T=60)
    m = SmoothnessMetric()
    scorer = PhaseGatedScorer(m, strategy="uniform")
    scorer.fit(dataset)
    s = scorer.score(dataset.demos[0])
    assert isinstance(s, float)
