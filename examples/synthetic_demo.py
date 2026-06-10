"""Synthetic dataset demo: full pipeline with all metrics.

Generates a synthetic dataset with known-clean and known-defective demos,
then runs the full haptal-curate pipeline to verify the metrics discriminate
between them.

Usage:
    python examples/synthetic_demo.py
"""
import numpy as np

from haptal_curate import score, curate, report
from haptal_curate.types import Demo, DemoDataset
from haptal_curate.metrics import (
    SmoothnessMetric,
    EntropyMetric,
    GripperTimingMetric,
    IsolationForestMetric,
    KNNMetric,
    TrajectoryAlignmentMetric,
    EnsembleMetric,
)


def make_clean_demo(T: int = 100, act_dim: int = 7, seed: int = 0, demo_id: str = "clean_0") -> Demo:
    """Clean demo: smooth actions, gripper opens late."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, T)
    actions = np.stack([np.sin(t + i * 0.3) for i in range(act_dim)], axis=1).astype(np.float32)
    actions *= 0.1  # small amplitude = smooth
    actions[:, -1] = -1.0
    actions[int(0.85 * T):, -1] = 1.0  # gripper opens late
    obs = (rng.standard_normal((T, 9)) * 0.01).astype(np.float32)
    return Demo(obs=obs, actions=actions, episode_length=T, demo_id=demo_id)


def make_defective_demo(T: int = 100, act_dim: int = 7, seed: int = 0, demo_id: str = "bad_0") -> Demo:
    """Defective demo: jittery actions, gripper opens early."""
    rng = np.random.default_rng(seed)
    actions = (rng.standard_normal((T, act_dim)) * 2.0).astype(np.float32)
    actions[:, -1] = -1.0
    actions[int(0.15 * T):, -1] = 1.0  # gripper opens early (defective)
    obs = rng.standard_normal((T, 9)).astype(np.float32)
    return Demo(obs=obs, actions=actions, episode_length=T, demo_id=demo_id)


def main():
    print("=== Synthetic Dataset Demo ===\n")

    n_clean = 15
    n_defective = 5
    T = 100

    clean_demos = [make_clean_demo(T=T, seed=i, demo_id=f"clean_{i}") for i in range(n_clean)]
    bad_demos = [make_defective_demo(T=T, seed=i, demo_id=f"bad_{i}") for i in range(n_defective)]
    all_demos = clean_demos + bad_demos
    dataset = DemoDataset(demos=all_demos)

    print(f"Dataset: {n_clean} clean + {n_defective} defective = {len(all_demos)} demos")
    print()

    # Score with each metric individually
    metrics = [
        SmoothnessMetric(),
        EntropyMetric(),
        GripperTimingMetric(),
        IsolationForestMetric(n_estimators=50),
        KNNMetric(k=5),
        TrajectoryAlignmentMetric(),
        EnsembleMetric(),
    ]

    print(f"{'Metric':<25} {'Clean mean':>12} {'Defective mean':>15} {'Separates?':>12}")
    print("-" * 70)
    for metric in metrics:
        metric.fit(dataset)
        scores = np.array([metric.score(d) for d in all_demos])
        clean_scores = scores[:n_clean]
        bad_scores = scores[n_clean:]
        separates = "YES" if clean_scores.mean() > bad_scores.mean() else "no"
        print(f"{metric.name:<25} {clean_scores.mean():>12.4f} {bad_scores.mean():>15.4f} {separates:>12}")

    print()

    # Run full pipeline with EnsembleMetric
    print("Full pipeline with EnsembleMetric:")
    score_result = score(dataset, metrics=[EnsembleMetric()])
    curation_result = curate(score_result, fraction=0.75)
    summary = report(score_result, curation_result)

    kept_ids = set(score_result.demo_ids[i] for i in curation_result.kept_indices)
    n_clean_kept = sum(1 for d in clean_demos if d.demo_id in kept_ids)
    n_bad_kept = sum(1 for d in bad_demos if d.demo_id in kept_ids)

    print(f"  Kept {len(curation_result.kept_indices)}/{len(all_demos)} demos")
    print(f"  Clean demos kept: {n_clean_kept}/{n_clean}")
    print(f"  Defective demos kept: {n_bad_kept}/{n_defective}")
    if "confound" in summary:
        print(f"  Length confound severity: {summary['confound']['severity']}")


if __name__ == "__main__":
    main()
