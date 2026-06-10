"""Gradio app for haptal-curate: demonstration quality scoring and curation."""
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
import h5py

import haptal_curate as hc
from haptal_curate.metrics import (
    EnsembleMetric,
    SmoothnessMetric,
    GripperTimingMetric,
    EntropyMetric,
    IsolationForestMetric,
    KNNMetric,
    TrajectoryAlignmentMetric,
)
from haptal_curate.confounds import detect_length_confound

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

METRIC_CHOICES = [
    "Ensemble (smoothness + gripper timing)",
    "Smoothness (SPARC)",
    "Gripper Timing",
    "Entropy (neg. action std)",
    "Isolation Forest",
    "k-NN Distance",
    "Trajectory Alignment",
]


def _build_metric(name: str):
    return {
        "Ensemble (smoothness + gripper timing)": EnsembleMetric,
        "Smoothness (SPARC)": SmoothnessMetric,
        "Gripper Timing": GripperTimingMetric,
        "Entropy (neg. action std)": EntropyMetric,
        "Isolation Forest": IsolationForestMetric,
        "k-NN Distance": KNNMetric,
        "Trajectory Alignment": TrajectoryAlignmentMetric,
    }[name]()


def _score_bar(scores, sorted_idx, color_fn, title):
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = [color_fn(i) for i in sorted_idx]
    ax.bar(range(len(scores)), scores[sorted_idx], color=colors, width=1.0, linewidth=0)
    ax.set_xlabel("Demo rank (best → worst)")
    ax.set_ylabel("Quality score")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def make_synthetic_hdf5(path: str, n_clean: int = 40, n_defective: int = 7, seed: int = 42):
    """Write a synthetic LIBERO-like HDF5 with clean and defective demos."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        idx = 0
        for _ in range(n_clean):
            T = int(rng.integers(300, 330))
            dg = grp.create_group(f"demo_{idx}")
            actions = (rng.standard_normal((T, 7)) * 0.05).astype(np.float32)
            actions[:, -1] = -1.0
            open_t = int(rng.integers(int(T * 0.8), T))
            actions[open_t:, -1] = 1.0
            dg.create_dataset("actions", data=actions)
            og = dg.create_group("obs")
            og.create_dataset("robot0_eef_pos",
                              data=rng.standard_normal((T, 3)).astype(np.float32))
            idx += 1
        for _ in range(n_defective):
            T = 500
            dg = grp.create_group(f"demo_{idx}")
            actions = (rng.standard_normal((T, 7)) * 0.3).astype(np.float32)
            actions[:, -1] = -1.0
            open_t = int(rng.integers(50, 150))
            actions[open_t:, -1] = 1.0
            dg.create_dataset("actions", data=actions)
            og = dg.create_group("obs")
            og.create_dataset("robot0_eef_pos",
                              data=rng.standard_normal((T, 3)).astype(np.float32))
            idx += 1


# ---------------------------------------------------------------------------
# Tab 1: Score & Curate (upload HDF5)
# ---------------------------------------------------------------------------

def run_score_curate(hdf5_file, metric_name, truncate_t, fraction):
    if hdf5_file is None:
        return "Please upload an HDF5 file.", [], None

    try:
        path = hdf5_file if isinstance(hdf5_file, str) else hdf5_file.name
        metric = _build_metric(metric_name)
        result = hc.score(path, metrics=[metric], truncate_t=int(truncate_t))
        curation = hc.curate(result, fraction=float(fraction))
        summary = hc.report(result, curation)

        kept_set = set(curation.kept_indices.tolist())
        sorted_idx = np.argsort(result.scores)[::-1]
        table = [
            [
                rank + 1,
                result.demo_ids[i],
                f"{result.scores[i]:.4f}",
                "✓ kept" if i in kept_set else "✗ removed",
            ]
            for rank, i in enumerate(sorted_idx)
        ]

        confound_line = ""
        if "confound" in summary:
            c = summary["confound"]
            confound_line = (
                f"\n\nLength Confound: {c['severity'].upper()} "
                f"(Spearman r = {c['spearman_r']:.3f})\n{c['message']}"
            )

        text = (
            f"Metric: {metric_name}\n"
            f"Demos scored: {summary['n_demos']}\n"
            f"Score  mean ± std: {summary['score_mean']:.4f} ± {summary['score_std']:.4f}\n"
            f"Kept: {summary['curation']['n_kept']} / {summary['n_demos']} "
            f"({summary['curation']['fraction_kept']:.0%})\n"
            f"Top 5:    {', '.join(summary['top5_demo_ids'])}\n"
            f"Bottom 5: {', '.join(summary['bottom5_demo_ids'])}"
            + confound_line
        )

        fig = _score_bar(
            result.scores,
            sorted_idx,
            color_fn=lambda i: "#2196F3" if i in kept_set else "#EF5350",
            title=f"{metric_name}  |  blue = kept, red = removed",
        )
        return text, table, fig

    except Exception as exc:
        import traceback
        return f"Error: {exc}\n\n{traceback.format_exc()}", [], None


# ---------------------------------------------------------------------------
# Tab 2: Try with synthetic data
# ---------------------------------------------------------------------------

def run_synthetic(n_clean, n_defective, metric_name, fraction, show_confound):
    n_clean, n_defective = int(n_clean), int(n_defective)
    tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
    path = tmp.name
    tmp.close()

    try:
        make_synthetic_hdf5(path, n_clean=n_clean, n_defective=n_defective)

        lines = [f"Dataset: {n_clean} clean + {n_defective} defective demos"]

        # Optionally show untruncated (confounded) run
        result_raw = None
        if show_confound:
            result_raw = hc.score(
                path, metrics=[_build_metric(metric_name)], truncate_t=None
            )
            cr = detect_length_confound(result_raw.scores, result_raw.episode_lengths)
            lines += [
                "",
                "WITHOUT truncation (length confound demo):",
                f"  Severity: {cr.severity.upper()}  |  Spearman r = {cr.spearman_r:.3f}",
                f"  {cr.message}",
            ]

        # Truncated run
        result = hc.score(path, metrics=[_build_metric(metric_name)], truncate_t=324)
        curation = hc.curate(result, fraction=float(fraction))
        summary = hc.report(result, curation)

        lines += [
            "",
            "WITH truncation at TRUNC_T = 324:",
            f"  Score  mean ± std: {summary['score_mean']:.4f} ± {summary['score_std']:.4f}",
            f"  Kept: {summary['curation']['n_kept']} / {summary['n_demos']} "
            f"({summary['curation']['fraction_kept']:.0%})",
            f"  Top 5:    {', '.join(summary['top5_demo_ids'])}",
            f"  Bottom 5: {', '.join(summary['bottom5_demo_ids'])}",
        ]

        text = "\n".join(lines)

        # Table — colour-coded by ground-truth label
        kept_set = set(curation.kept_indices.tolist())
        sorted_idx = np.argsort(result.scores)[::-1]
        table = [
            [
                rank + 1,
                result.demo_ids[i],
                f"{result.scores[i]:.4f}",
                "✓ kept" if i in kept_set else "✗ removed",
                "defective" if i >= n_clean else "clean",
            ]
            for rank, i in enumerate(sorted_idx)
        ]

        # Plot
        ncols = 2 if show_confound and result_raw is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 3.5))
        if ncols == 1:
            axes = [axes]

        # Truncated subplot (always last)
        s_idx = np.argsort(result.scores)[::-1]
        axes[-1].bar(
            range(len(result.scores)),
            result.scores[s_idx],
            color=["#EF5350" if i >= n_clean else "#2196F3" for i in s_idx],
            width=1.0, linewidth=0,
        )
        axes[-1].set_title("t = 324  |  blue = clean, red = defective")
        axes[-1].set_xlabel("Demo rank")
        axes[-1].set_ylabel("Quality score")
        axes[-1].spines[["top", "right"]].set_visible(False)

        if ncols == 2:
            r_idx = np.argsort(result_raw.scores)[::-1]
            axes[0].bar(
                range(len(result_raw.scores)),
                result_raw.scores[r_idx],
                color=["#EF5350" if i >= n_clean else "#2196F3" for i in r_idx],
                width=1.0, linewidth=0,
            )
            axes[0].set_title("No truncation (length confound)")
            axes[0].set_xlabel("Demo rank")
            axes[0].set_ylabel("Quality score")
            axes[0].spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        return text, table, fig

    except Exception as exc:
        import traceback
        return f"Error: {exc}\n\n{traceback.format_exc()}", [], None

    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# haptal-curate

**Quality scoring and curation for robot demonstration datasets.**

Backed by [arXiv:2606.03134](https://arxiv.org/abs/2606.03134),
[arXiv:2606.05588](https://arxiv.org/abs/2606.05588), and
[arXiv:2606.10229](https://arxiv.org/abs/2606.10229).

> **Length-confound warning:** without truncating trajectories, 5 of 7 metrics achieve AUROC ≈ 1.0
> simply because defective demos run the full episode horizon. Always use `truncate_t = 324`.
"""

with gr.Blocks(title="haptal-curate") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ── Tab 1: Score & Curate ──────────────────────────────────────────
        with gr.Tab("Score & Curate"):
            gr.Markdown(
                "Upload a LIBERO-format HDF5 file, choose a quality metric, "
                "and curate the dataset."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    t1_file = gr.File(label="HDF5 demo file", file_types=[".hdf5", ".h5"])
                    t1_metric = gr.Dropdown(
                        choices=METRIC_CHOICES,
                        value=METRIC_CHOICES[0],
                        label="Quality metric",
                    )
                    t1_trunc = gr.Slider(
                        minimum=50, maximum=500, value=324, step=1,
                        label="Truncate to T timesteps",
                    )
                    t1_frac = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.8, step=0.05,
                        label="Fraction to keep",
                    )
                    t1_btn = gr.Button("Run", variant="primary")

                with gr.Column(scale=2):
                    t1_summary = gr.Textbox(label="Summary", lines=10)
                    t1_plot = gr.Plot(label="Score distribution")
                    t1_table = gr.Dataframe(
                        headers=["Rank", "Demo ID", "Score", "Decision"],
                        label="All demos (sorted by score)",
                        wrap=True,
                    )

            t1_btn.click(
                fn=run_score_curate,
                inputs=[t1_file, t1_metric, t1_trunc, t1_frac],
                outputs=[t1_summary, t1_table, t1_plot],
            )

        # ── Tab 2: Try with synthetic data ─────────────────────────────────
        with gr.Tab("Try with synthetic data"):
            gr.Markdown(
                "Generate synthetic demonstrations (no LIBERO needed) and run "
                "the full scoring pipeline. Toggle **Show length confound** to "
                "see what happens *without* truncation."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    t2_n_clean = gr.Slider(
                        minimum=5, maximum=100, value=40, step=1,
                        label="Number of clean demos",
                    )
                    t2_n_def = gr.Slider(
                        minimum=1, maximum=30, value=7, step=1,
                        label="Number of defective demos",
                    )
                    t2_metric = gr.Dropdown(
                        choices=METRIC_CHOICES,
                        value=METRIC_CHOICES[0],
                        label="Quality metric",
                    )
                    t2_frac = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.85, step=0.05,
                        label="Fraction to keep",
                    )
                    t2_confound = gr.Checkbox(
                        label="Show length confound (no truncation vs. t=324)",
                        value=True,
                    )
                    t2_btn = gr.Button("Run", variant="primary")

                with gr.Column(scale=2):
                    t2_summary = gr.Textbox(label="Summary", lines=12)
                    t2_plot = gr.Plot(label="Score distribution")
                    t2_table = gr.Dataframe(
                        headers=["Rank", "Demo ID", "Score", "Decision", "Ground truth"],
                        label="All demos (sorted by score)",
                        wrap=True,
                    )

            t2_btn.click(
                fn=run_synthetic,
                inputs=[t2_n_clean, t2_n_def, t2_metric, t2_frac, t2_confound],
                outputs=[t2_summary, t2_table, t2_plot],
            )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
