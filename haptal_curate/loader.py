from __future__ import annotations
from typing import Optional, List
import numpy as np
import h5py

from .types import Demo, DemoDataset, TRUNC_T


def load_demos(
    hdf5_path: str,
    truncate_t: Optional[int] = TRUNC_T,
    max_demos: Optional[int] = None,
) -> DemoDataset:
    """Load demonstrations from an HDF5 file.

    Supports LIBERO-format HDF5 files with data/demo_N/actions structure.
    Also supports a minimal format with just data/demo_N/actions.

    Args:
        hdf5_path: Path to HDF5 file.
        truncate_t: Truncate all demos to this length. None for no truncation.
        max_demos: Maximum number of demos to load. None for all.

    Returns:
        DemoDataset with loaded demonstrations.
    """
    demos: List[Demo] = []
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        demo_keys = sorted(
            [k for k in data_group.keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]

        for key in demo_keys:
            demo_grp = data_group[key]
            actions = demo_grp["actions"][()]  # (T, act_dim)
            episode_length = len(actions)

            # Build obs array from available obs keys
            obs = _load_obs(demo_grp, episode_length)

            # Load phase labels if present
            phase_labels = None
            if "phase_labels" in demo_grp:
                phase_labels = demo_grp["phase_labels"][()].astype(np.int32)

            if truncate_t is not None:
                t = min(truncate_t, episode_length)
                actions = actions[:t]
                obs = obs[:t]
                if phase_labels is not None:
                    phase_labels = phase_labels[:t]
                episode_length = t

            demos.append(Demo(
                obs=obs,
                actions=actions,
                episode_length=episode_length,
                demo_id=key,
                phase_labels=phase_labels,
            ))

    metadata = {"hdf5_path": hdf5_path, "truncate_t": truncate_t}
    return DemoDataset(demos=demos, metadata=metadata)


def _load_obs(demo_grp: h5py.Group, episode_length: int) -> np.ndarray:
    """Extract a flat observation array from a demo group."""
    if "obs" not in demo_grp:
        # Fall back to states if available
        if "states" in demo_grp:
            return demo_grp["states"][()].astype(np.float32)
        return np.zeros((episode_length, 1), dtype=np.float32)

    obs_grp = demo_grp["obs"]
    # Collect numeric obs keys (skip image keys to keep memory manageable)
    parts = []
    for k in sorted(obs_grp.keys()):
        arr = obs_grp[k][()]
        if arr.ndim == 2 and arr.shape[0] == episode_length:
            parts.append(arr.astype(np.float32))
        elif arr.ndim >= 3:
            # Skip image observations
            continue

    if not parts:
        return np.zeros((episode_length, 1), dtype=np.float32)
    return np.concatenate(parts, axis=1)
