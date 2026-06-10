"""haptal-curate  —  dark, monospace, minimal Gradio UI."""
import json
import os
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
import h5py

import haptal_curate as hc
from haptal_curate.metrics import (
    EnsembleMetric, SmoothnessMetric, GripperTimingMetric,
    EntropyMetric, IsolationForestMetric, KNNMetric, TrajectoryAlignmentMetric,
)
from haptal_curate.phase import PhaseGatedScorer
from haptal_curate.confounds import detect_length_confound
from haptal_curate.types import ScoreResult

# ── chart theme ──────────────────────────────────────────────────────────────
ACCENT  = "#00ff88"
RED     = "#ff4444"
SURFACE = "#111111"
MUTED   = "#555555"
BORDER  = "#222222"

plt.rcParams.update({
    "figure.facecolor":  SURFACE,
    "axes.facecolor":    "#0d0d0d",
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "axes.titlecolor":   "#888888",
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "text.color":        "#cccccc",
    "grid.color":        "#1a1a1a",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         9,
    "figure.dpi":        130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── metric registry ───────────────────────────────────────────────────────────
METRIC_CHOICES = [
    "ensemble",
    "smoothness",
    "gripper_timing",
    "entropy",
    "isolation_forest",
    "knn",
    "trajectory_alignment",
]

_METRIC_MAP = {
    "ensemble":             EnsembleMetric,
    "smoothness":           SmoothnessMetric,
    "gripper_timing":       GripperTimingMetric,
    "entropy":              EntropyMetric,
    "isolation_forest":     IsolationForestMetric,
    "knn":                  KNNMetric,
    "trajectory_alignment": TrajectoryAlignmentMetric,
}


def _build_metric(name: str):
    return _METRIC_MAP[name]()


# ── synthetic data ────────────────────────────────────────────────────────────
def _make_synthetic_hdf5(path: str, n_clean: int, n_defective: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        g = f.create_group("data")
        for k in range(n_clean):
            T = int(rng.integers(300, 330))
            dg = g.create_group(f"demo_{k}")
            a = (rng.standard_normal((T, 7)) * 0.05).astype(np.float32)
            a[:, -1] = -1.0
            a[int(rng.integers(int(T * 0.8), T)):, -1] = 1.0
            dg.create_dataset("actions", data=a)
            dg.create_group("obs").create_dataset(
                "robot0_eef_pos", data=rng.standard_normal((T, 3)).astype(np.float32))
        for k in range(n_defective):
            T = 500
            dg = g.create_group(f"demo_{n_clean + k}")
            a = (rng.standard_normal((T, 7)) * 0.3).astype(np.float32)
            a[:, -1] = -1.0
            a[int(rng.integers(50, 150)):, -1] = 1.0
            dg.create_dataset("actions", data=a)
            dg.create_group("obs").create_dataset(
                "robot0_eef_pos", data=rng.standard_normal((T, 3)).astype(np.float32))


# ── charts ────────────────────────────────────────────────────────────────────
def _bar_chart(scores, sorted_idx, colors, xlabel, title, figsize=(7, 2.8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(scores)), scores[sorted_idx],
           color=colors, width=1.0, linewidth=0)
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel("score", labelpad=6)
    ax.set_title(title, pad=8)
    ax.set_xlim(-0.5, len(scores) - 0.5)
    fig.tight_layout(pad=1.2)
    return fig


def _score_chart(result, curation):
    kept_set = set(curation.kept_indices.tolist())
    idx = np.argsort(result.scores)[::-1]
    colors = [ACCENT if i in kept_set else RED for i in idx]
    return _bar_chart(result.scores, idx, colors,
                      "demo rank  (best → worst)",
                      f"{result.metric_name}   |   green = kept   red = removed")


def _confound_chart(result_raw, result_trunc, n_clean):
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.8))
    for ax, res, title in [
        (axes[0], result_raw,   "no truncation  (length confound)"),
        (axes[1], result_trunc, "truncated at t=324"),
    ]:
        idx = np.argsort(res.scores)[::-1]
        colors = [RED if i >= n_clean else ACCENT for i in idx]
        ax.bar(range(len(res.scores)), res.scores[idx],
               color=colors, width=1.0, linewidth=0)
        ax.set_title(title, pad=8)
        ax.set_xlabel("demo rank", labelpad=6)
        ax.set_ylabel("score", labelpad=6)
        ax.set_xlim(-0.5, len(res.scores) - 0.5)
    fig.tight_layout(pad=1.4)
    return fig


def _single_chart(result, n_clean):
    idx = np.argsort(result.scores)[::-1]
    colors = [RED if i >= n_clean else ACCENT for i in idx]
    return _bar_chart(result.scores, idx, colors,
                      "demo rank  (best → worst)",
                      "truncated at t=324   |   green = clean   red = defective")


# ── tab 1 handler ─────────────────────────────────────────────────────────────
def run_score_curate(hdf5_file, metric_name, phase_gated, truncate_t, fraction):
    if hdf5_file is None:
        return "[error] no file uploaded", [], None, None

    try:
        path = hdf5_file if isinstance(hdf5_file, str) else hdf5_file.name
        metric = _build_metric(metric_name)
        dataset = hc.load_demos(path, truncate_t=int(truncate_t))

        if phase_gated:
            scorer = PhaseGatedScorer(metric, strategy="uniform")
            scorer.fit(dataset)
            scores_arr = scorer.score_dataset(dataset)
            result = ScoreResult(
                scores=scores_arr,
                demo_ids=[d.demo_id for d in dataset.demos],
                metric_name=f"phase-gated:{metric_name}",
                episode_lengths=np.array([d.episode_length for d in dataset.demos]),
            )
        else:
            result = hc.score(dataset, metrics=[metric])

        curation = hc.curate(result, fraction=float(fraction))
        summary  = hc.report(result, curation)

        # confound line
        confound_lines = []
        if "confound" in summary:
            c = summary["confound"]
            sev = c["severity"]
            prefix = "[warn]" if sev in ("mild", "moderate") else "[critical]" if sev == "severe" else "[ok]"
            confound_lines = [
                "",
                f"{prefix} length confound: {sev}  (spearman r={c['spearman_r']:.3f})",
                f"         {c['message']}",
            ]

        lines = [
            f"metric         {result.metric_name}",
            f"demos          {summary['n_demos']}",
            f"score mean±std {summary['score_mean']:.4f} ± {summary['score_std']:.4f}",
            f"score range    [{summary['score_min']:.4f}, {summary['score_max']:.4f}]",
            f"kept           {summary['curation']['n_kept']} / {summary['n_demos']}"
            f"  ({summary['curation']['fraction_kept']:.0%})",
            f"top-5          {', '.join(summary['top5_demo_ids'])}",
            f"bottom-5       {', '.join(summary['bottom5_demo_ids'])}",
        ] + confound_lines

        text = "\n".join(lines)

        # table
        kept_set = set(curation.kept_indices.tolist())
        sorted_idx = np.argsort(result.scores)[::-1]
        table = [
            [rank + 1, result.demo_ids[i], f"{result.scores[i]:.4f}",
             "kept" if i in kept_set else "removed"]
            for rank, i in enumerate(sorted_idx)
        ]

        # download json
        download_data = {
            "metric": result.metric_name,
            "truncate_t": int(truncate_t),
            "kept_fraction": float(fraction),
            "n_kept": int(len(curation.kept_indices)),
            "kept_demo_ids": [result.demo_ids[i] for i in curation.kept_indices],
            "scores": {result.demo_ids[i]: float(result.scores[i])
                       for i in range(len(result.scores))},
        }
        tmp_json = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="haptal_curate_")
        json.dump(download_data, tmp_json, indent=2)
        tmp_json.close()

        fig = _score_chart(result, curation)
        return text, table, fig, tmp_json.name

    except Exception as exc:
        msg = f"[error] {exc}\n\n{traceback.format_exc()}"
        return msg, [], None, None


# ── tab 2 handler ─────────────────────────────────────────────────────────────
def run_synthetic(n_clean, n_defective, metric_name, fraction, compare_confound):
    n_clean, n_defective = int(n_clean), int(n_defective)
    tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
    path = tmp.name
    tmp.close()

    try:
        _make_synthetic_hdf5(path, n_clean=n_clean, n_defective=n_defective)

        result_raw = None
        confound_lines = []
        if compare_confound:
            result_raw = hc.score(
                path, metrics=[_build_metric(metric_name)], truncate_t=None)
            cr = detect_length_confound(result_raw.scores, result_raw.episode_lengths)
            confound_lines = [
                "",
                f"without truncation:",
                f"  length confound  {cr.severity}  (r={cr.spearman_r:.3f})",
                f"  {cr.message}",
            ]

        result  = hc.score(path, metrics=[_build_metric(metric_name)], truncate_t=324)
        curation = hc.curate(result, fraction=float(fraction))
        summary  = hc.report(result, curation)

        lines = [
            f"dataset        {n_clean} clean + {n_defective} defective",
            f"metric         {metric_name}",
            f"score mean±std {summary['score_mean']:.4f} ± {summary['score_std']:.4f}",
            f"kept           {summary['curation']['n_kept']} / {summary['n_demos']}"
            f"  ({summary['curation']['fraction_kept']:.0%})",
            f"top-5          {', '.join(summary['top5_demo_ids'])}",
            f"bottom-5       {', '.join(summary['bottom5_demo_ids'])}",
        ] + confound_lines

        text = "\n".join(lines)

        # table
        kept_set = set(curation.kept_indices.tolist())
        sorted_idx = np.argsort(result.scores)[::-1]
        table = [
            [rank + 1, result.demo_ids[i], f"{result.scores[i]:.4f}",
             "kept" if i in kept_set else "removed",
             "defective" if i >= n_clean else "clean"]
            for rank, i in enumerate(sorted_idx)
        ]

        if compare_confound and result_raw is not None:
            fig = _confound_chart(result_raw, result, n_clean)
        else:
            fig = _single_chart(result, n_clean)

        return text, table, fig

    except Exception as exc:
        return f"[error] {exc}\n\n{traceback.format_exc()}", [], None

    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500&display=swap');

/* ── variables ── */
:root {
    --bg:       #0a0a0a;
    --s1:       #111111;
    --s2:       #161616;
    --border:   #1e1e1e;
    --border2:  #2a2a2a;
    --accent:   #00ff88;
    --red:      #ff4444;
    --text:     #cccccc;
    --muted:    #555555;
    --mono:     'JetBrains Mono', 'Fira Code', monospace;
    --sans:     'Inter', system-ui, sans-serif;
}

/* ── reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body {
    background: var(--bg) !important;
    color: var(--text) !important;
    margin: 0 !important;
}

.gradio-container {
    background: var(--bg) !important;
    font-family: var(--sans) !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    min-height: 100vh !important;
}

/* hide gradio branding */
footer, .built-with { display: none !important; }
.share-btn-container { display: none !important; }

/* ── header ── */
.hc-header {
    padding: 18px 28px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 14px;
}
.hc-header-name {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: -0.02em;
}
.hc-header-sub {
    font-family: var(--sans);
    font-size: 12px;
    color: var(--muted);
}

/* ── tabs ── */
.tabs > .tab-nav {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 24px !important;
    gap: 0 !important;
    margin-bottom: 0 !important;
}
.tabs > .tab-nav > button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    letter-spacing: 0.06em !important;
    padding: 11px 20px !important;
    margin: 0 !important;
    margin-bottom: -1px !important;
    text-transform: lowercase !important;
    transition: color 0.12s !important;
}
.tabs > .tab-nav > button:hover { color: var(--text) !important; }
.tabs > .tab-nav > button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
.tabitem { padding: 0 !important; }

/* ── all panels ── */
.block, .panel, .form, .contain, .wrap, .box, .gap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    gap: 0 !important;
}

/* ── inner section padding ── */
.gradio-container .row { gap: 0 !important; }
.gradio-container .col { padding: 0 !important; }

/* ── labels ── */
label > span,
.label-wrap span,
.block > label > span {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    font-weight: 400 !important;
    padding-bottom: 5px !important;
    display: block !important;
}

/* ── text inputs ── */
input[type=text], input[type=number], textarea {
    background: var(--s1) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    padding: 8px 10px !important;
    transition: border-color 0.1s !important;
}
input[type=text]:focus, textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ── sliders ── */
input[type=range] { accent-color: var(--accent) !important; cursor: pointer !important; }
.wrap .numeral, .wrap .number, input[type=number] {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: var(--text) !important;
}

/* ── checkbox / radio ── */
input[type=checkbox], input[type=radio] {
    accent-color: var(--accent) !important;
    cursor: pointer !important;
}

/* ── select / dropdown ── */
select, .dropdown {
    background: var(--s1) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
}
.options-wrap {
    background: var(--s1) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
}
.item { color: var(--text) !important; }
.item:hover, .item.selected {
    background: var(--s2) !important;
    color: var(--accent) !important;
}

/* ── file upload ── */
.upload-container, .file-upload, .dnd-container,
.file-preview-holder, .wrap .wrap {
    background: var(--s1) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 0 !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    transition: border-color 0.12s !important;
}
.upload-container:hover { border-color: var(--accent) !important; }
.file-name, .file-name span { color: var(--accent) !important; font-family: var(--mono) !important; }

/* ── primary button ── */
button.primary {
    background: transparent !important;
    border: 2px solid var(--accent) !important;
    border-radius: 0 !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    padding: 11px 0 !important;
    width: 100% !important;
    text-transform: lowercase !important;
    transition: background 0.1s !important;
    cursor: pointer !important;
    margin-top: 4px !important;
}
button.primary:hover  { background: rgba(0,255,136,0.06) !important; }
button.primary:active { background: rgba(0,255,136,0.14) !important; }

/* secondary (download, clear, etc.) */
button.secondary {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.06em !important;
    text-transform: lowercase !important;
}
button.secondary:hover { border-color: var(--border2) !important; color: var(--text) !important; }

/* icon-only buttons */
button.icon { color: var(--muted) !important; background: transparent !important; }

/* ── textbox output ── */
.output-text-area textarea,
textarea.scroll-hide,
.block textarea {
    background: var(--s1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 11.5px !important;
    line-height: 1.75 !important;
    padding: 12px !important;
}

/* ── dataframe / table ── */
table { border-collapse: collapse !important; width: 100% !important; }
thead th {
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 9.5px !important;
    letter-spacing: 0.12em !important;
    padding: 7px 12px !important;
    text-transform: uppercase !important;
    font-weight: 400 !important;
}
tbody td {
    background: var(--s1) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    padding: 5px 12px !important;
}
tbody tr:hover td { background: var(--s2) !important; }

/* ── plot wrapper ── */
.plot-container, .matplotlib-plot, .plot-container > div {
    background: var(--s1) !important;
    border-radius: 0 !important;
    border: 1px solid var(--border) !important;
}

/* ── markdown ── */
.prose, .markdown {
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 13px !important;
}
.prose code, .markdown code {
    background: var(--s2) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    border-radius: 0 !important;
    padding: 1px 5px !important;
    font-size: 11px !important;
}
.prose a, .markdown a { color: var(--accent) !important; text-decoration: none !important; }
.prose a:hover { text-decoration: underline !important; }

/* ── custom note ── */
.hc-note {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    padding: 10px 0 16px;
    line-height: 1.5;
}

/* ── section padding wrapper ── */
.hc-col-inputs  { padding: 24px 20px 24px 28px !important; border-right: 1px solid var(--border); }
.hc-col-outputs { padding: 24px 28px 24px 20px !important; }

/* ── footer ── */
.hc-footer {
    padding: 14px 28px;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--muted);
    display: flex;
    gap: 0;
    align-items: center;
    line-height: 1;
}
.hc-footer a { color: var(--muted); text-decoration: none; }
.hc-footer a:hover { color: var(--accent); }
.hc-footer .sep { color: var(--border2); margin: 0 10px; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: #333; }

/* ── input row spacing ── */
.input-stack > * + * { margin-top: 16px !important; }

/* ── no blue focus rings ── */
*:focus { outline: none !important; box-shadow: none !important; }
*:focus-visible { outline: 1px solid var(--accent) !important; }
"""

HEADER_HTML = """
<div class="hc-header">
  <span class="hc-header-name">haptal-curate</span>
  <span class="hc-header-sub">demonstration quality scoring + curation</span>
</div>
"""

FOOTER_HTML = """
<div class="hc-footer">
  haptal-curate
  <span class="sep">|</span>
  <a href="https://arxiv.org/abs/2606.03134" target="_blank">arxiv:2606.03134</a>
  <span class="sep">&nbsp;</span>
  <a href="https://arxiv.org/abs/2606.05588" target="_blank">arxiv:2606.05588</a>
  <span class="sep">&nbsp;</span>
  <a href="https://arxiv.org/abs/2606.10229" target="_blank">arxiv:2606.10229</a>
  <span class="sep">|</span>
  <a href="https://github.com/aaravbedi/haptal-curate" target="_blank">github</a>
</div>
"""

NOTE_SYNTHETIC = """
<div class="hc-note">
  // no robot data needed — generates synthetic demos on the fly
</div>
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="haptal-curate") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── tab 1: score & curate ─────────────────────────────────────────
        with gr.Tab("score & curate"):
            with gr.Row():

                # left: inputs
                with gr.Column(scale=4, elem_classes=["hc-col-inputs"]):
                    with gr.Column(elem_classes=["input-stack"]):
                        t1_file = gr.File(
                            label="hdf5 file",
                            file_types=[".hdf5", ".h5"],
                        )
                        t1_metric = gr.Dropdown(
                            choices=METRIC_CHOICES,
                            value="ensemble",
                            label="metric",
                        )
                        t1_phase = gr.Checkbox(
                            label="phase-gated scoring",
                            value=False,
                        )
                        t1_trunc = gr.Slider(
                            minimum=50, maximum=500, value=324, step=1,
                            label="truncate t",
                        )
                        t1_frac = gr.Slider(
                            minimum=0.05, maximum=1.0, value=0.8, step=0.05,
                            label="keep fraction",
                        )
                        t1_btn = gr.Button("run", variant="primary")

                # right: outputs
                with gr.Column(scale=6, elem_classes=["hc-col-outputs"]):
                    t1_plot    = gr.Plot(label="score distribution", show_label=False)
                    t1_summary = gr.Textbox(label="stats", lines=9, interactive=False)
                    t1_table   = gr.Dataframe(
                        headers=["rank", "demo id", "score", "decision"],
                        label="scores",
                        wrap=False,
                        row_count=(10, "dynamic"),
                    )
                    t1_dl = gr.File(label="download curation json", visible=True)

            t1_btn.click(
                fn=run_score_curate,
                inputs=[t1_file, t1_metric, t1_phase, t1_trunc, t1_frac],
                outputs=[t1_summary, t1_table, t1_plot, t1_dl],
            )

        # ── tab 2: synthetic ──────────────────────────────────────────────
        with gr.Tab("synthetic"):
            with gr.Row():

                # left: inputs
                with gr.Column(scale=4, elem_classes=["hc-col-inputs"]):
                    with gr.Column(elem_classes=["input-stack"]):
                        gr.HTML(NOTE_SYNTHETIC)
                        t2_clean = gr.Slider(
                            minimum=5, maximum=120, value=40, step=1,
                            label="clean demos",
                        )
                        t2_defective = gr.Slider(
                            minimum=1, maximum=40, value=7, step=1,
                            label="defective demos",
                        )
                        t2_metric = gr.Dropdown(
                            choices=METRIC_CHOICES,
                            value="ensemble",
                            label="metric",
                        )
                        t2_frac = gr.Slider(
                            minimum=0.05, maximum=1.0, value=0.85, step=0.05,
                            label="keep fraction",
                        )
                        t2_confound = gr.Checkbox(
                            label="compare without truncation",
                            value=True,
                        )
                        t2_btn = gr.Button("run", variant="primary")

                # right: outputs
                with gr.Column(scale=6, elem_classes=["hc-col-outputs"]):
                    t2_plot    = gr.Plot(label="score distribution", show_label=False)
                    t2_summary = gr.Textbox(label="stats", lines=9, interactive=False)
                    t2_table   = gr.Dataframe(
                        headers=["rank", "demo id", "score", "decision", "ground truth"],
                        label="scores",
                        wrap=False,
                        row_count=(10, "dynamic"),
                    )

            t2_btn.click(
                fn=run_synthetic,
                inputs=[t2_clean, t2_defective, t2_metric, t2_frac, t2_confound],
                outputs=[t2_summary, t2_table, t2_plot],
            )

    gr.HTML(FOOTER_HTML)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(), css=CSS)
