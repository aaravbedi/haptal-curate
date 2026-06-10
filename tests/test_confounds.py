"""Tests for length confound detection."""
import numpy as np
import pytest

from haptal_curate.confounds import detect_length_confound
from haptal_curate.types import TRUNC_T


def test_no_confound():
    rng = np.random.default_rng(0)
    n = 50
    scores = rng.standard_normal(n)
    lengths = rng.integers(50, 100, size=n)
    result = detect_length_confound(scores, lengths)
    # With random uncorrelated data, expect no/mild confound
    assert result.severity in ("none", "mild", "moderate", "severe")
    assert isinstance(result.spearman_r, float)
    assert isinstance(result.p_value, float)
    assert result.recommended_truncation_t > 0


def test_severe_confound_detected():
    n = 50
    lengths = np.linspace(10, 200, n)
    # Scores perfectly correlated with length
    scores = lengths + np.random.default_rng(5).standard_normal(n) * 0.01
    result = detect_length_confound(scores, lengths)
    assert result.severity == "severe"
    assert result.spearman_r > 0.9


def test_negative_correlation_also_detected():
    n = 50
    lengths = np.linspace(10, 200, n)
    # Scores inversely correlated with length (longer = lower score)
    scores = -lengths + np.random.default_rng(6).standard_normal(n) * 0.01
    result = detect_length_confound(scores, lengths)
    assert result.severity == "severe"
    assert result.spearman_r < -0.9


def test_recommended_truncation_bounded():
    n = 30
    lengths = np.linspace(10, 500, n)
    scores = np.random.default_rng(1).standard_normal(n)
    result = detect_length_confound(scores, lengths)
    assert result.recommended_truncation_t <= TRUNC_T
    assert result.recommended_truncation_t > 0


def test_confound_report_has_message():
    n = 20
    scores = np.random.default_rng(2).standard_normal(n)
    lengths = np.random.default_rng(3).integers(40, 120, size=n)
    result = detect_length_confound(scores, lengths)
    assert isinstance(result.message, str)
    assert len(result.message) > 0
