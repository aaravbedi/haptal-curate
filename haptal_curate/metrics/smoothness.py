from __future__ import annotations
import numpy as np
from .base import BaseMetric
from ..types import Demo


class SmoothnessMetric(BaseMetric):
    """SPARC spectral arc length smoothness metric.

    Computes the spectral arc length of the end-effector speed profile.
    Higher (less negative) values indicate smoother, cleaner demonstrations.
    """

    name = "smoothness"
    requires_fit = False

    def __init__(self, fs: float = 1.0, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05):
        self.fs = fs
        self.padlevel = padlevel
        self.fc = fc
        self.amp_th = amp_th

    def score(self, demo: Demo) -> float:
        """Return SPARC score for the demo's action trajectory."""
        actions = demo.actions  # (T, act_dim)
        # Use translational actions (first 3 dims if available, else all)
        act = actions[:, :3] if actions.shape[1] >= 3 else actions
        # Compute speed profile (norm of action vectors)
        speed = np.linalg.norm(act, axis=1).astype(float)
        return _sparc(speed, fs=self.fs, padlevel=self.padlevel, fc=self.fc, amp_th=self.amp_th)


def _sparc(movement: np.ndarray, fs: float = 1.0, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Compute SPARC (Spectral Arc Length) smoothness."""
    N = len(movement)
    if N < 2:
        return 0.0
    nfft = int(2 ** (np.ceil(np.log2(N)) + padlevel))
    df = fs / nfft
    # One-sided spectrum
    mag = np.abs(np.fft.rfft(movement, n=nfft))
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    # Normalize by DC component; avoid division by zero
    dc = mag[0] if mag[0] > 1e-12 else 1e-12
    mag_norm = mag / dc
    # Find cutoff index
    fc_idx = int(round(fc / df))
    fc_idx = min(fc_idx, len(mag_norm) - 1)
    # Trim to amplitude threshold
    mask = mag_norm[:fc_idx + 1] >= amp_th
    if not np.any(mask):
        return 0.0
    fc_idx = int(np.where(mask)[0][-1])
    # Arc length of normalized spectrum
    seg = mag_norm[:fc_idx + 1]
    dseg = np.diff(seg)
    arc_len = -float(np.sum(np.sqrt(df ** 2 + dseg ** 2)))
    return arc_len
