"""Basic example: score a LIBERO-format HDF5 dataset and print a report.

Usage:
    python examples/basic_scoring.py --hdf5 path/to/demo.hdf5

To run with synthetic data (no HDF5 file needed):
    python examples/basic_scoring.py --synthetic
"""
import argparse
import numpy as np

from haptal_curate import score, curate, report
from haptal_curate.types import Demo, DemoDataset
from haptal_curate.metrics import SmoothnessMetric, GripperTimingMetric, EnsembleMetric


def make_synthetic_dataset(n: int = 20, T: int = 80, seed: int = 42) -> DemoDataset:
    """Generate a synthetic dataset for demonstration purposes."""
    rng = np.random.default_rng(seed)
    demos = []
    for i in range(n):
        actions = rng.standard_normal((T, 7)).astype(np.float32)
        obs = rng.standard_normal((T, 9)).astype(np.float32)
        # Vary gripper timing: some demos open early (defective), some late (clean)
        open_at = rng.integers(T // 4, T)
        actions[:, -1] = -1.0
        actions[open_at:, -1] = 1.0
        demos.append(Demo(obs=obs, actions=actions, episode_length=T, demo_id=f"demo_{i}"))
    return DemoDataset(demos=demos)


def main():
    parser = argparse.ArgumentParser(description="haptal-curate basic scoring example")
    parser.add_argument("--hdf5", type=str, default=None, help="Path to HDF5 demo file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--fraction", type=float, default=0.5, help="Fraction of demos to keep")
    args = parser.parse_args()

    if args.hdf5:
        print(f"Loading demos from {args.hdf5}")
        score_result = score(args.hdf5)
    else:
        print("Using synthetic dataset (20 demos, T=80)")
        dataset = make_synthetic_dataset()
        score_result = score(dataset, metrics=[EnsembleMetric()])

    print(f"\nScored {len(score_result.scores)} demos using metric: {score_result.metric_name}")
    print(f"  Score range: [{score_result.scores.min():.3f}, {score_result.scores.max():.3f}]")
    print(f"  Score mean:  {score_result.scores.mean():.3f} ± {score_result.scores.std():.3f}")

    # Curate: keep top 50%
    curation_result = curate(score_result, fraction=args.fraction)
    print(f"\nCuration (top {args.fraction*100:.0f}%):")
    print(f"  Kept {len(curation_result.kept_indices)} demos, removed {len(curation_result.removed_indices)}")

    # Report
    summary = report(score_result, curation_result)
    print("\nTop 5 demos (highest quality):", summary["top5_demo_ids"])
    print("Bottom 5 demos (lowest quality):", summary["bottom5_demo_ids"])
    if "confound" in summary:
        c = summary["confound"]
        print(f"\nLength confound: severity={c['severity']}, Spearman r={c['spearman_r']:.3f}")
        print(f"  {c['message']}")


if __name__ == "__main__":
    main()
