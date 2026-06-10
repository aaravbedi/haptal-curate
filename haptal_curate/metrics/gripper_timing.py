from __future__ import annotations
import numpy as np
from .base import BaseMetric
from ..types import Demo


class GripperTimingMetric(BaseMetric):
    """Normalized first-open timestep metric.

    Detects premature gripper release. In LIBERO, the last action dimension
    controls the gripper (positive = open, negative = close).

    Demos that open the gripper early score LOW (defective early-release).
    Demos that keep the gripper closed until near the end score HIGH (clean).
    """

    name = "gripper_timing"
    requires_fit = False

    def __init__(self, gripper_dim: int = -1, open_threshold: float = 0.0):
        """
        Args:
            gripper_dim: Action dimension for gripper control.
            open_threshold: Threshold above which gripper is considered open.
        """
        self.gripper_dim = gripper_dim
        self.open_threshold = open_threshold

    def score(self, demo: Demo) -> float:
        """Return normalized timestep of first gripper open event."""
        T = demo.episode_length
        gripper = demo.actions[:T, self.gripper_dim]
        open_steps = np.where(gripper > self.open_threshold)[0]
        if len(open_steps) == 0:
            # Gripper never opens in this window → clean (score = 1.0)
            return 1.0
        first_open = int(open_steps[0])
        return first_open / T
