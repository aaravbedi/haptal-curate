"""Integration tests for score -> curate -> report pipeline."""
import tempfile
import json
import numpy as np
import pytest

import haptal_curate
from haptal_curate import score, curate, report, load_demos
from haptal_curate.metrics import SmoothnessMetric, GripperTimingMetric, EnsembleMetric
from haptal_curate.types import DemoDataset, TRUNC_T
from tests.conftest import make_dataset, make_hdf5


def test_score_returns_score_result():
    dataset = make_dataset(n=10, T=60)
    result = score(dataset)
    assert result.scores.shape == (10,)
    assert len(result.demo_ids) == 10


def test_score_with_multiple_metrics():
    dataset = make_dataset(n=10, T=60)
    metrics = [SmoothnessMetric(), GripperTimingMetric()]
    result = score(dataset, metrics=metrics)
    assert result.scores.shape == (10,)
    assert "smoothness" in result.per_metric_scores
    assert "gripper_timing" in result.per_metric_scores


def test_score_from_hdf5():
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        path = f.name
    make_hdf5(path, n=5, T=50)
    result = score(path, truncate_t=40)
    assert result.scores.shape == (5,)


def test_curate_top_fraction():
    dataset = make_dataset(n=20, T=60)
    sr = score(dataset)
    cr = curate(sr, fraction=0.5, strategy="top_fraction")
    assert len(cr.kept_indices) == 10
    assert len(cr.removed_indices) == 10
    assert abs(cr.fraction_kept - 0.5) < 0.01


def test_curate_threshold():
    dataset = make_dataset(n=20, T=60)
    sr = score(dataset)
    cr = curate(sr, strategy="threshold")
    assert len(cr.kept_indices) + len(cr.removed_indices) == 20


def test_curate_invalid_strategy():
    dataset = make_dataset(n=5, T=60)
    sr = score(dataset)
    with pytest.raises(ValueError):
        curate(sr, strategy="unknown_strategy")


def test_report_returns_dict():
    dataset = make_dataset(n=10, T=60)
    sr = score(dataset)
    summary = report(sr)
    assert isinstance(summary, dict)
    assert "n_demos" in summary
    assert summary["n_demos"] == 10


def test_report_with_curation():
    dataset = make_dataset(n=10, T=60)
    sr = score(dataset)
    cr = curate(sr, fraction=0.6)
    summary = report(sr, cr)
    assert "curation" in summary
    assert summary["curation"]["n_kept"] == 6


def test_report_writes_json():
    dataset = make_dataset(n=8, T=60)
    sr = score(dataset)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    report(sr, output_path=path)
    with open(path) as f:
        data = json.load(f)
    assert "n_demos" in data


def test_full_pipeline_end_to_end():
    dataset = make_dataset(n=15, T=80)
    sr = score(dataset, metrics=[EnsembleMetric()])
    cr = curate(sr, fraction=0.7)
    summary = report(sr, cr)
    assert summary["curation"]["fraction_kept"] > 0
    assert "confound" in summary


def test_load_demos_from_hdf5():
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        path = f.name
    make_hdf5(path, n=8, T=60)
    dataset = load_demos(path, truncate_t=50, max_demos=5)
    assert len(dataset) == 5
    for d in dataset.demos:
        assert d.episode_length <= 50


def test_package_version():
    assert hasattr(haptal_curate, "__version__")
    assert haptal_curate.__version__ == "0.1.0"
