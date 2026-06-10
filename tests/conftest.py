"""Shared fixtures for tests."""
import numpy as np
import pytest
import tempfile
import h5py

from haptal_curate.types import Demo, DemoDataset, TRUNC_T


def make_demo(
    T: int = 100,
    act_dim: int = 7,
    obs_dim: int = 9,
    seed: int = 0,
    gripper_open_at: int = None,
    demo_id: str = "demo_0",
) -> Demo:
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((T, obs_dim)).astype(np.float32)
    actions = rng.standard_normal((T, act_dim)).astype(np.float32)
    # gripper is last dim: negative = closed, positive = open
    actions[:, -1] = -1.0  # start closed
    if gripper_open_at is not None:
        actions[gripper_open_at:, -1] = 1.0
    return Demo(obs=obs, actions=actions, episode_length=T, demo_id=demo_id)


def make_dataset(n: int = 10, T: int = 100, act_dim: int = 7, obs_dim: int = 9) -> DemoDataset:
    demos = [make_demo(T=T, act_dim=act_dim, obs_dim=obs_dim, seed=i, demo_id=f"demo_{i}") for i in range(n)]
    return DemoDataset(demos=demos)


def make_hdf5(path: str, n: int = 10, T: int = 100, act_dim: int = 7, obs_dim: int = 9):
    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        rng = np.random.default_rng(42)
        for i in range(n):
            dg = grp.create_group(f"demo_{i}")
            dg.create_dataset("actions", data=rng.standard_normal((T, act_dim)).astype(np.float32))
            obs_grp = dg.create_group("obs")
            obs_grp.create_dataset("robot0_eef_pos", data=rng.standard_normal((T, 3)).astype(np.float32))
            obs_grp.create_dataset("robot0_joint_pos", data=rng.standard_normal((T, obs_dim)).astype(np.float32))
