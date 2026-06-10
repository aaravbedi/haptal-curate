"""Unit tests for all quality metrics."""
import numpy as np
import pytest

from haptal_curate.types import DemoDataset
from haptal_curate.metrics import (
    SmoothnessMetric,
    EntropyMetric,
    GripperTimingMetric,
    IsolationForestMetric,
    KNNMetric,
    TrajectoryAlignmentMetric,
    EnsembleMetric,
)
from tests.conftest import make_demo, make_dataset


def test_smoothness_returns_float():
    demo = make_demo(T=50, act_dim=7)
    m = SmoothnessMetric()
    s = m.score(demo)
    assert isinstance(s, float)


def test_smoothness_smoother_scores_higher():
    # Smooth demo: constant actions
    smooth = make_demo(T=50, act_dim=7, seed=0)
    smooth.actions[:] = 0.01  # near-constant
    # Jittery demo: random actions
    jittery = make_demo(T=50, act_dim=7, seed=99)
    jittery.actions = np.random.default_rng(99).standard_normal((50, 7)).astype(np.float32) * 5.0
    m = SmoothnessMetric()
    assert m.score(smooth) > m.score(jittery)


def test_entropy_returns_float():
    demo = make_demo(T=50, act_dim=7)
    m = EntropyMetric()
    assert isinstance(m.score(demo), float)


def test_entropy_low_variance_scores_higher():
    low_var = make_demo(T=50, act_dim=7)
    low_var.actions[:] = 0.001
    high_var = make_demo(T=50, act_dim=7)
    high_var.actions = np.random.default_rng(1).standard_normal((50, 7)).astype(np.float32)
    m = EntropyMetric()
    assert m.score(low_var) > m.score(high_var)


def test_gripper_timing_returns_float():
    demo = make_demo(T=100, act_dim=7, gripper_open_at=80)
    m = GripperTimingMetric()
    assert isinstance(m.score(demo), float)


def test_gripper_timing_late_open_scores_higher():
    late_open = make_demo(T=100, act_dim=7, gripper_open_at=90)
    early_open = make_demo(T=100, act_dim=7, gripper_open_at=10)
    m = GripperTimingMetric()
    assert m.score(late_open) > m.score(early_open)


def test_gripper_timing_no_open_scores_one():
    never_open = make_demo(T=100, act_dim=7)
    never_open.actions[:, -1] = -1.0  # always closed
    m = GripperTimingMetric()
    assert m.score(never_open) == 1.0


def test_isolation_forest_fit_and_score():
    dataset = make_dataset(n=20, T=50)
    m = IsolationForestMetric(n_estimators=10)
    m.fit(dataset)
    s = m.score(dataset.demos[0])
    assert isinstance(s, float)


def test_isolation_forest_requires_fit():
    demo = make_demo(T=50)
    m = IsolationForestMetric()
    with pytest.raises(RuntimeError):
        m.score(demo)


def test_knn_fit_and_score():
    dataset = make_dataset(n=20, T=50)
    m = KNNMetric(k=3)
    m.fit(dataset)
    s = m.score(dataset.demos[0])
    assert isinstance(s, float)
    assert s <= 0.0  # negative distance


def test_knn_requires_fit():
    demo = make_demo(T=50)
    m = KNNMetric()
    with pytest.raises(RuntimeError):
        m.score(demo)


def test_trajectory_alignment_fit_and_score():
    dataset = make_dataset(n=20, T=50)
    m = TrajectoryAlignmentMetric()
    m.fit(dataset)
    s = m.score(dataset.demos[0])
    assert isinstance(s, float)
    assert -1.0 <= s <= 1.0


def test_ensemble_returns_float():
    dataset = make_dataset(n=10, T=50)
    m = EnsembleMetric()
    m.fit(dataset)
    s = m.score(dataset.demos[0])
    assert isinstance(s, float)


def test_all_metrics_score_all_demos():
    dataset = make_dataset(n=10, T=60)
    metrics = [
        SmoothnessMetric(),
        EntropyMetric(),
        GripperTimingMetric(),
        IsolationForestMetric(n_estimators=10),
        KNNMetric(k=3),
        TrajectoryAlignmentMetric(),
        EnsembleMetric(),
    ]
    for m in metrics:
        m.fit(dataset)
        scores = m.score_dataset(dataset)
        assert len(scores) == len(dataset)
        assert all(isinstance(s, float) for s in scores)
