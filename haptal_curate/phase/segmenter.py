from __future__ import annotations
from enum import IntEnum
from typing import Dict, List, Tuple
import numpy as np

from ..types import Demo


class Phase(IntEnum):
    PREGRASP = 0
    GRASP = 1
    POST_CONTACT = 2


def extract_phase_windows(
    demo: Demo,
    gripper_dim: int = -1,
    open_threshold: float = 0.0,
    close_threshold: float = 0.0,
) -> Dict[Phase, np.ndarray]:
    """Split a demo trajectory into phase windows based on gripper state.

    Phases:
        PREGRASP: timesteps before the first gripper close.
        GRASP: timesteps while gripper is closed.
        POST_CONTACT: timesteps after the first gripper open following a close.

    If explicit phase_labels are present in the demo, they are used directly.

    Args:
        demo: The demonstration to segment.
        gripper_dim: Action dimension for gripper (default: last).
        open_threshold: Threshold above which gripper is open.
        close_threshold: Threshold below which gripper is closed.

    Returns:
        Dict mapping Phase -> index array of timesteps in that phase.
    """
    T = demo.episode_length

    if demo.phase_labels is not None:
        labels = demo.phase_labels[:T]
        return {
            Phase.PREGRASP: np.where(labels == Phase.PREGRASP)[0],
            Phase.GRASP: np.where(labels == Phase.GRASP)[0],
            Phase.POST_CONTACT: np.where(labels == Phase.POST_CONTACT)[0],
        }

    gripper = demo.actions[:T, gripper_dim]
    is_open = gripper > open_threshold

    # Find first close transition
    first_close = _first_transition(is_open, from_val=True, to_val=False)
    if first_close is None:
        first_close = 0

    # Find first open after close
    first_open_after_close = None
    for t in range(first_close, T):
        if is_open[t]:
            first_open_after_close = t
            break

    pregrasp = np.arange(0, first_close, dtype=int)
    if first_open_after_close is not None:
        grasp = np.arange(first_close, first_open_after_close, dtype=int)
        post_contact = np.arange(first_open_after_close, T, dtype=int)
    else:
        grasp = np.arange(first_close, T, dtype=int)
        post_contact = np.array([], dtype=int)

    return {
        Phase.PREGRASP: pregrasp,
        Phase.GRASP: grasp,
        Phase.POST_CONTACT: post_contact,
    }


def _first_transition(arr: np.ndarray, from_val: bool, to_val: bool) -> int | None:
    """Return index of first transition from from_val to to_val."""
    for i in range(len(arr) - 1):
        if arr[i] == from_val and arr[i + 1] == to_val:
            return i + 1
    return None
