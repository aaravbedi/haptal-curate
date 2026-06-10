"""Phase-gated scoring example.

Demonstrates splitting demonstrations into PREGRASP / GRASP / POST_CONTACT
phases and scoring each phase independently with SmoothnessMetric.

Usage:
    python examples/phase_gated_example.py
"""
import numpy as np

from haptal_curate.types import Demo, DemoDataset
from haptal_curate.metrics.smoothness import SmoothnessMetric
from haptal_curate.phase.segmenter import Phase, extract_phase_windows
from haptal_curate.phase.phase_gated import PhaseGatedScorer


def make_demo_with_phases(
    T: int = 120,
    act_dim: int = 7,
    obs_dim: int = 9,
    grasp_at: int = 50,
    release_at: int = 90,
    seed: int = 0,
    demo_id: str = "demo_0",
) -> Demo:
    """Create a demo with explicit phase labels."""
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((T, obs_dim)).astype(np.float32)
    actions = rng.standard_normal((T, act_dim)).astype(np.float32)
    # Set gripper: closed during grasp
    actions[:, -1] = 1.0            # open (pregrasp)
    actions[grasp_at:release_at, -1] = -1.0  # closed (grasp)
    actions[release_at:, -1] = 1.0  # open again (post_contact)

    labels = np.zeros(T, dtype=np.int32)
    labels[grasp_at:release_at] = int(Phase.GRASP)
    labels[release_at:] = int(Phase.POST_CONTACT)

    return Demo(obs=obs, actions=actions, episode_length=T, demo_id=demo_id, phase_labels=labels)


def main():
    print("=== Phase-Gated Scoring Example ===\n")

    # Build a small dataset with heterogeneous phase timing
    demos = [
        make_demo_with_phases(grasp_at=40, release_at=80, seed=i, demo_id=f"demo_{i}")
        for i in range(12)
    ]
    dataset = DemoDataset(demos=demos)

    # 1. Inspect phase windows for one demo
    demo = demos[0]
    windows = extract_phase_windows(demo)
    print("Phase windows for demo_0:")
    for phase, idx in windows.items():
        print(f"  {phase.name:15s}: {len(idx):3d} timesteps")

    print()

    # 2. Phase-gated scoring with uniform strategy
    metric = SmoothnessMetric()
    scorer = PhaseGatedScorer(metric, strategy="uniform")
    scorer.fit(dataset)
    scores = scorer.score_dataset(dataset)

    print("Phase-gated scores (uniform aggregation):")
    for i, (demo, s) in enumerate(zip(demos, scores)):
        print(f"  {demo.demo_id}: {s:.4f}")

    print(f"\nMean score: {scores.mean():.4f} | Std: {scores.std():.4f}")

    # 3. Compare strategies
    print("\nComparing aggregation strategies:")
    for strategy in ("uniform", "best_per_phase", "global"):
        s = PhaseGatedScorer(metric, strategy=strategy)
        s.fit(dataset)
        all_scores = s.score_dataset(dataset)
        print(f"  {strategy:15s}: mean={all_scores.mean():.4f}, std={all_scores.std():.4f}")


if __name__ == "__main__":
    main()
