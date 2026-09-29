"""'GHD Results' page: the Experiment 1 convergence-speed comparison only.

Rendered by pages/1_📊_GHD_Results.py as four tabs:
  Exp 1a (gain 1.0) from results/archive_exp1a_gain1/,
  Exp 1b (gain 4.0) from results/ (files with init_gain == 4.0), and
  Exp 2 / 2b (7-layer, gain 1.5 / 1.0) from results/exp2/ (by file prefix).
Auto-refreshes every 30 s while runs are in progress.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ghd.experiment import CONFIGS, RUNNING_DIR, SEEDS, config_label
from ghd.summary import aggregate, best_indices, load_results


RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    results_dir: Path
    init_gain: float
    note: str
    prefix: str | None = None  # only result files with this prefix (see ghd.paths.result_files)
    num_layers: int = 10
    controller_note: str = ""
    gain_card: tuple[str, str] | None = None  # (label, subtitle) override for the GHD Improvement card
    table_extra: tuple[tuple[str, str], ...] = ()  # ablation configs shown in the tables only, not the charts

    @property
    def table_configs(self) -> list[tuple[str, str]]:
        """CONFIGS with `table_extra` inserted after the same optimizer's GHD-Rules row."""
        out = list(CONFIGS)
        for cfg in self.table_extra:
            out.insert(out.index((cfg[0], "rules")) + 1, cfg)
        return out

    @property
    def controller_path(self) -> Path:
        return self.results_dir / "ghd_controller.pt"


EXPERIMENTS = [
    Experiment("exp1a", "Exp 1a — Gain 1.0 (severe vanishing)", RESULTS_DIR / "archive_exp1a_gain1", 1.0,
               "Xavier gain 1.0: the forward signal collapses ~10¹⁰× across 10 layers. Detection is the "
               "contribution here; no optimizer trains SGD/LARS past chance. Run with the pre-weight-decay-split "
               "hook and lrs SGD 0.001 / LARS 0.01 / LNGD 1.0 / AdamW 0.001, 5000 steps.",
               gain_card=("GHD vs Baseline",
                          "Detection only — no convergence improvement at gain 1.0 (gap too large to correct)")),
    Experiment("exp1b", "Exp 1b — Gain 4.0 (moderate vanishing)", RESULTS_DIR, 4.0,
               "Xavier gain 4.0: all optimizers can train; both detection and correction are evaluated.",
               controller_note="Note: 100% reflects ~100% healthy training examples at gain 4.0. "
                               "Primary evidence is convergence speed match."),
    Experiment("exp2", "Exp 2 — Main Result (gain 1.5)", RESULTS_DIR / "exp2", 1.5,
               "Main result: SGD+GHD-AI reaches 80% in 300 steps vs 473 (37% faster).  \n"
               "SGD+GHD-Rules reaches 80% in 313 steps (34% faster).  \n"
               "Vanishing correction alone (197 interventions) produces the full speed-up.",
               prefix="mlp_mnist_l7_g1.5", num_layers=7, table_extra=(("sgd", "rules_vanish_only"),)),
    Experiment("exp2b", "Exp 2b — Rescue Case (gain 1.0)", RESULTS_DIR / "exp2", 1.0,
               "SGD: 9.99% final accuracy (never trains). SGD+GHD-Rules: 95.56% final accuracy (rescued).",
               prefix="mlp_mnist_l7_g1", num_layers=7),
]
MARKER_MAX_AGE = 3 * 3600  # ignore stale markers from killed runs
REFRESH_SECONDS = 30

COLORS = {
    ("sgd", "none"): "#64748B", ("sgd", "rules"): "#94A3B8", ("sgd", "ai"): "#1E40AF",
    ("adamw", "none"): "#0F766E", ("adamw", "rules"): "#5EEAD4", ("adamw", "ai"): "#0D9488",
    ("lars", "none"): "#92400E", ("lars", "ai"): "#F59E0B",
    ("lngd", "none"): "#6B21A8", ("lngd", "ai"): "#A855F7",
    ("sgd", "rules_vanish_only"): "#CBD5E1",
}
MODE_COLORS = {
    "healthy": "#14B8A6", "vanishing": "#EF4444", "exploding": "#F97316", "oscillating": "#EAB308",
    "noisy": "#A855F7", "recovering": "#6B7280", "warmup": "#D1D5DB",
}
# Delta rows: (after this config, [(ghd config, baseline config), ...]).
DELTA_ROWS = {
    ("sgd", "ai"): [(("sgd", "rules"), ("sgd", "none")), (("sgd", "ai"), ("sgd", "none"))],
    ("adamw", "ai"): [(("adamw", "rules"), ("adamw", "none")), (("adamw", "ai"), ("adamw", "none"))],
    ("lars", "ai"): [(("lars", "ai"), ("lars", "none"))],
    ("lngd", "ai"): [(("lngd", "ai"), ("lngd", "none"))],
}
THRESHOLDS = ["70", "80", "90"]
GOOD, BAD = "#22C55E", "#EF4444"

CSS = """
<style>
.ghd-sub { color: #94A3B8; margin-top: -0.6rem; margin-bottom: 1.2rem; font-size: 1.02rem; }
.ghd-card {
    background: #111827; border: 1px solid #1F2937; border-radius: 14px;
    padding: 18px 20px; height: 100%; color: #E5E7EB;
}
.ghd-card .label { font-size: 0.8rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.04em; }
.ghd-card .value { font-size: 1.9rem; font-weight: 700; margin: 6px 0 2px 0; font-variant-numeric: tabular-nums; }
.ghd-card .note { font-size: 0.86rem; color: #9CA3AF; }
.ghd-table { width: 100%; border-collapse: separate; border-spacing: 0; border-radius: 12px; overflow: hidden;
             border: 1px solid #1F2937; font-variant-numeric: tabular-nums; font-size: 0.92rem; }
.ghd-table th { background: #0B1220; color: #CBD5E1; text-align: left; padding: 10px 12px; font-weight: 600; }
.ghd-table td { background: #111827; color: #E5E7EB; padding: 8px 12px; border-top: 1px solid #1F2937; }
.ghd-table tr.ai td { background: #EFF6FF; color: #1E293B; }
.ghd-table tr.delta td { background: #0B1220; color: #94A3B8; font-size: 0.84rem; padding: 5px 12px; }
.ghd-table tr.pending td { color: #6B7280; font-style: italic; }
.ghd-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 8px; }
.ghd-chart-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 0; }
.ghd-chart-sub { color: #94A3B8; font-size: 0.86rem; margin-bottom: 0.4rem; }
</style>
"""


# ---------------------------------------------------------------- data
def _fingerprint(exp: Experiment) -> tuple:
    files = sorted(exp.results_dir.glob("mlp_mnist_*_seed*.json"))
    ckpt = exp.controller_path.stat().st_mtime if exp.controller_path.exists() else 0
    return tuple((p.name, p.stat().st_mtime) for p in files) + (("ckpt", ckpt),)


@st.cache_data(show_spinner="Loading GHD results...")
def _load(results_dir: str, init_gain: float, prefix: str | None, _fp: tuple) -> list[dict]:
    """Results in `results_dir` whose model used `init_gain` (files without the field are gain 1.0)."""
    return [r for r in load_results(results_dir, with_logs=True, prefix=prefix)
            if float(r["config"].get("init_gain", 1.0)) == init_gain]


def _load_exp(exp: Experiment) -> list[dict]:
    return _load(str(exp.results_dir), exp.init_gain, exp.prefix, _fingerprint(exp))


@st.cache_data
def _controller_info(path: str, _fp: tuple) -> dict | None:
    if not Path(path).exists():
        return None
    import torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    history = ckpt.get("history") or []
    best = min(history, key=lambda h: h["val_loss"]) if history else None
    return {"val_acc": best["val_acc"] if best else None, "class_counts": ckpt.get("class_counts", {})}


def runs_in_progress() -> bool:
    now = time.time()
    markers = (RESULTS_DIR / RUNNING_DIR).glob("*") if (RESULTS_DIR / RUNNING_DIR).exists() else []
    return any(now - m.stat().st_mtime < MARKER_MAX_AGE for m in markers)


def _runs(results: list[dict], cfg: tuple[str, str]) -> list[dict]:
    return [r for r in results if (r["config"]["optimizer"], r["config"]["ghd_mode"]) == cfg]


def _label(cfg: tuple[str, str]) -> str:
    return config_label(*cfg).replace(" + ", "+").replace("GHD-Vanish-Only", "Vanish-Only")


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _log_frame(runs: list[dict]) -> pd.DataFrame:
    rows = [
        {"seed": r["config"]["seed"], "step": e["step"], "layer": int(name[1:]),
         "mode": rec["mode"], "strength": rec["strength"], "s_depth": rec["s_depth"]}
        for r in runs for e in r.get("ghd_logs", []) for name, rec in e["layers"].items()
    ]
    return pd.DataFrame(rows)


def _card(label: str, value: str, note: str = "", color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ""
    return (f'<div class="ghd-card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value"{style}>{html.escape(value)}</div><div class="note">{html.escape(note)}</div></div>')


def _sci(x: float | None) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.1e}"


# ---------------------------------------------------------------- section 1 + 2: comparisons
FAMILIES = ["sgd", "adamw", "lars", "lngd"]
BASELINES = [c for c in CONFIGS if c[1] == "none"]


def _max_step(results: list[dict]) -> int:
    return max((r["val_steps"][-1] for r in results if r["val_steps"]), default=5000)


def _add_config_traces(fig: go.Figure, results: list[dict], cfg: tuple[str, str], showlegend: bool = True,
                       **pos) -> bool:
    runs = _runs(results, cfg)
    if not runs:
        return False
    df = pd.DataFrame([
        {"step": s, "acc": a} for r in runs for s, a in zip(r["val_steps"], r["val_acc_curve"])
    ]).groupby("step")["acc"].agg(["mean", "min", "max"]).reset_index()
    color, label, is_ghd = COLORS[cfg], _label(cfg), cfg[1] != "none"
    fig.add_trace(go.Scatter(
        x=np.concatenate([df["step"], df["step"][::-1]]), y=np.concatenate([df["max"], df["min"][::-1]]),
        fill="toself", fillcolor=_rgba(color, 0.14), line=dict(width=0), hoverinfo="skip",
        legendgroup=label, showlegend=False,
    ), **pos)
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["mean"], mode="lines", name=label, legendgroup=label, showlegend=showlegend,
        line=dict(color=color, width=2.5 if is_ghd else 1.5, dash="solid" if is_ghd else "dash"),
        customdata=np.stack([df["min"], df["max"]], axis=-1),
        hovertemplate=f"<b>{label}</b><br>step %{{x}}<br>mean %{{y:.1f}}%"
                      "<br>min–max %{customdata[0]:.1f}–%{customdata[1]:.1f}%<extra></extra>",
    ), **pos)
    return True


def _threshold_lines(fig: go.Figure, **pos) -> None:
    for th in (70, 80, 90):
        fig.add_hline(y=th, line=dict(color="#9CA3AF", width=1, dash="dot"),
                      annotation_text=f"{th}%", annotation_position="top left",
                      annotation_font=dict(color="#9CA3AF", size=11), **pos)


def _missing_note(results: list[dict], configs: list[tuple[str, str]]) -> None:
    note = "Mean across seeds; shaded band = min/max across seeds. Solid = GHD, dashed = baseline."
    missing = [_label(c) for c in configs if not _runs(results, c)]
    if missing:
        note += f" Running… (no results yet): {', '.join(missing)}."
    st.caption(note)


def chart_race(results: list[dict], configs: list[tuple[str, str]], key: str) -> None:
    fig = go.Figure()
    for cfg in configs:
        _add_config_traces(fig, results, cfg)
    _threshold_lines(fig)
    fig.update_layout(
        height=450, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Training step", range=[0, _max_step(results)]),
        yaxis=dict(title="Validation accuracy (%)", range=[0, 100]),
        legend=dict(x=1.02, xanchor="left", y=1, yanchor="top"),
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch", theme="streamlit", key=key)
    _missing_note(results, configs)


def chart_facets(results: list[dict], key: str) -> None:
    """One panel per optimizer: the baseline against its GHD variants."""
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True, vertical_spacing=0.14, horizontal_spacing=0.05,
                        subplot_titles=[_label((f, "none")) + " vs +GHD" for f in FAMILIES])
    for i, fam in enumerate(FAMILIES):
        pos = dict(row=i // 2 + 1, col=i % 2 + 1)
        for cfg in [c for c in CONFIGS if c[0] == fam]:
            _add_config_traces(fig, results, cfg, **pos)
        _threshold_lines(fig, **pos)
        fam_runs = [r for r in results if r["config"]["optimizer"] == fam]
        fig.update_xaxes(range=[0, _max_step(fam_runs) if fam_runs else 5000], **pos)
        fig.update_yaxes(range=[0, 100], **pos)
    fig.update_xaxes(title_text="Training step", row=2)
    fig.update_yaxes(title_text="Val accuracy (%)", col=1)
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(x=1.02, xanchor="left", y=1, yanchor="top"), hovermode="closest")
    st.plotly_chart(fig, width="stretch", theme="streamlit", key=key)
    _missing_note(results, CONFIGS)


def _steps_cell(row: pd.Series, th: str) -> str:
    reached, n = int(row[f"s{th}_reached"]), int(row["n_seeds"])
    if reached == 0:
        return "—"
    txt = f"{row[f's{th}_mean']:.0f} ± {row[f's{th}_std']:.0f}"
    return txt if reached == n else f"{txt} ({reached}/{n})"


def _full_steps(row: pd.Series | None, th: str) -> float | None:
    if row is None or int(row[f"s{th}_reached"]) != int(row["n_seeds"]):
        return None
    return float(row[f"s{th}_mean"])


def _delta_steps(ghd: pd.Series | None, base: pd.Series | None, th: str) -> str:
    if ghd is None or base is None:
        return "—"
    g, b = _full_steps(ghd, th), _full_steps(base, th)
    if g is None and b is None:
        return "—"
    if b is None:
        return f'<span style="color:{GOOD}">reached (baseline did not)</span>'
    if g is None:
        return f'<span style="color:{BAD}">not reached (baseline did)</span>'
    d = g - b
    color = GOOD if d < 0 else BAD if d > 0 else "#94A3B8"
    return f'<span style="color:{color}">{d:+.0f} steps</span>'.replace("-", "−")


def _delta_value(ghd: pd.Series | None, base: pd.Series | None, col: str) -> str:
    if ghd is None or base is None:
        return "—"
    d = ghd[f"{col}_mean"] - base[f"{col}_mean"]
    color = GOOD if d > 0 else BAD if d < 0 else "#94A3B8"
    return f'<span style="color:{color}">{d:+.1f}</span>'.replace("-", "−")


def table_threshold(results: list[dict], configs: list[tuple[str, str]], deltas: bool) -> None:
    """Steps-to-threshold table for `configs`; bold = best among the rows shown."""
    shown = [r for r in results if (r["config"]["optimizer"], r["config"]["ghd_mode"]) in configs]
    df = aggregate(shown)
    by_cfg = {(r["optimizer"], r["ghd_mode"]): (idx, r) for idx, r in df.iterrows()}
    best = best_indices(df) if not df.empty else {}

    head = "".join(f"<th>{h}</th>" for h in
                   ["Optimizer", "Steps→70%", "Steps→80%", "Steps→90%", "AUC", "Final Val Acc"])
    body = []
    for cfg in configs:
        swatch = f'<span class="ghd-swatch" style="background:{COLORS[cfg]}"></span>'
        name = swatch + html.escape(_label(cfg))
        if cfg not in by_cfg:
            body.append(f'<tr class="pending"><td>{name}</td>' + "<td>Running…</td>" * 5 + "</tr>")
        else:
            idx, row = by_cfg[cfg]
            cells = []
            for th in THRESHOLDS:
                c = _steps_cell(row, th)
                cells.append(f"<b>{c}</b>" if best.get(th) == idx else c)
            cells.append(f"{row['auc_mean']:.1f} ± {row['auc_std']:.1f}")
            cells.append(f"{row['final_val_mean']:.1f} ± {row['final_val_std']:.1f}%")
            cls = ' class="ai"' if cfg[1] == "ai" else ""
            body.append(f"<tr{cls}><td>{name}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        for ghd_cfg, base_cfg in (DELTA_ROWS.get(cfg, []) if deltas else []):
            g = by_cfg.get(ghd_cfg, (None, None))[1]
            b = by_cfg.get(base_cfg, (None, None))[1]
            cells = [_delta_steps(g, b, th) for th in THRESHOLDS]
            cells += [_delta_value(g, b, "auc"), _delta_value(g, b, "final_val")]
            label = f"Δ {_label(ghd_cfg)} vs {_label(base_cfg)}"
            body.append(f'<tr class="delta"><td>{html.escape(label)}</td>'
                        + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    st.markdown(f'<table class="ghd-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>',
                unsafe_allow_html=True)
    caption = ("Mean ± std across seeds. Steps average only the seeds that reached the threshold; (k/n) = k of n "
               "seeds reached it; — = never reached. Bold = fastest per threshold among the rows shown that reached "
               "it on every seed.")
    if deltas:
        caption += " Δ rows: negative steps (green) = GHD converged faster."
    st.caption(caption)


def section_comparisons(results: list[dict], k: str, table_configs: list[tuple[str, str]]) -> None:
    st.subheader("Convergence Race & Steps to Threshold")
    alone, versus, everything = st.tabs(["Optimizers alone", "Optimizer vs Optimizer + GHD", "All configurations"])
    with alone:
        st.caption("Plain optimizers only, no GHD: SGD vs AdamW vs LARS vs LNGD.")
        chart_race(results, BASELINES, key=f"{k}_race_alone")
        table_threshold(results, BASELINES, deltas=False)
    with versus:
        st.caption("Each optimizer against the same optimizer with GHD-Rules / GHD-AI.")
        chart_facets(results, key=f"{k}_race_versus")
        table_threshold(results, table_configs, deltas=True)
    with everything:
        chart_race(results, CONFIGS, key=f"{k}_race_all")
        table_threshold(results, table_configs, deltas=True)


# ---------------------------------------------------------------- section 3
def section_gradient_health(results: list[dict], cfg: tuple[str, str], k: str) -> None:
    runs = _runs(results, cfg)
    logs = _log_frame(runs)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="ghd-chart-title">Per-Layer Gradient Depth Signal</div>'
                    '<div class="ghd-chart-sub">Values below 0.01 indicate structural vanishing</div>',
                    unsafe_allow_html=True)
        post = logs[logs["step"] > 100] if not logs.empty else logs
        if post.empty:
            st.info("No gradient logs for this configuration yet.")
        else:
            ladder = post.groupby("layer")["s_depth"].mean().clip(lower=1e-12).sort_index()
            colors = ["#EF4444" if v < 0.01 else "#EAB308" if v < 0.1 else "#22C55E" for v in ladder]
            fig = go.Figure(go.Bar(
                x=ladder.values, y=[f"Layer {i}" for i in ladder.index], orientation="h", marker_color=colors,
                hovertemplate="%{y}: S_depth %{x:.1e}<extra></extra>",
            ))
            fig.add_vline(x=0.01, line=dict(color="#EF4444", width=2))
            fig.update_layout(
                height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                xaxis=dict(type="log", title="Mean S_depth (log scale)", exponentformat="power"),
                yaxis=dict(categoryorder="array", categoryarray=[f"Layer {i}" for i in ladder.index]),
            )
            st.plotly_chart(fig, width="stretch", theme="streamlit", key=f"{k}_sdepth")
    with right:
        st.markdown('<div class="ghd-chart-title">Failure Mode Distribution per Layer</div>'
                    '<div class="ghd-chart-sub">Share of logged steps in each mode</div>',
                    unsafe_allow_html=True)
        if cfg[1] == "none":
            st.info("Mode distribution is shown for GHD configurations only.")
        elif logs.empty:
            st.info("No gradient logs for this configuration yet.")
        else:
            share = (logs.groupby(["layer", "mode"]).size() / logs.groupby("layer").size() * 100).unstack(fill_value=0)
            fig = go.Figure()
            for mode, color in MODE_COLORS.items():
                if mode in share:
                    fig.add_trace(go.Bar(
                        x=share[mode], y=[f"Layer {i}" for i in share.index], orientation="h", name=mode,
                        marker_color=color, hovertemplate=f"%{{y}} · {mode}: %{{x:.1f}}%<extra></extra>",
                    ))
            fig.update_layout(
                barmode="stack", height=380, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="% of steps", range=[0, 100]),
                yaxis=dict(categoryorder="array", categoryarray=[f"Layer {i}" for i in share.index]),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, width="stretch", theme="streamlit", key=f"{k}_modes")


# ---------------------------------------------------------------- section 4
def section_ai_controller(results: list[dict], cfg: tuple[str, str], exp: Experiment, k: str) -> None:
    st.subheader("AI Controller")
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="ghd-chart-title">GHD-AI Correction Strength</div>'
                    '<div class="ghd-chart-sub">Mean across layers and seeds; band = min/max across layers</div>',
                    unsafe_allow_html=True)
        logs = _log_frame(_runs(results, cfg))
        logs = logs[logs["mode"] != "warmup"] if not logs.empty else logs
        if logs.empty:
            st.info("No post-warm-up logs yet.")
        else:
            s = logs.groupby("step")["strength"].agg(["mean", "min", "max"]).reset_index()
            color = COLORS[cfg]
            fig = go.Figure([
                go.Scatter(x=np.concatenate([s["step"], s["step"][::-1]]),
                           y=np.concatenate([s["max"], s["min"][::-1]]), fill="toself",
                           fillcolor=_rgba(color, 0.18), line=dict(width=0), hoverinfo="skip", showlegend=False),
                go.Scatter(x=s["step"], y=s["mean"], mode="lines", line=dict(color=color, width=2),
                           name="mean", hovertemplate="step %{x}<br>mean strength %{y:.2f}<extra></extra>"),
            ])
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                              xaxis=dict(title="Training step"), yaxis=dict(title="Strength", range=[0, 1]))
            st.plotly_chart(fig, width="stretch", theme="streamlit", key=f"{k}_strength")
    with right:
        info = _controller_info(str(exp.controller_path), _fingerprint(exp))
        if info is None or info["val_acc"] is None:
            st.markdown(_card("Controller Accuracy", "Running…", "controller not trained yet"),
                        unsafe_allow_html=True)
            return
        st.markdown(_card("Controller Accuracy", f"{info['val_acc'] * 100:.1f}%",
                          "vs rule-based labels on validation set"), unsafe_allow_html=True)
        if exp.controller_note:
            st.caption(exp.controller_note)
        counts = info["class_counts"]
        if counts:
            st.markdown('<div class="ghd-chart-title" style="margin-top:14px">Training Data Distribution</div>',
                        unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=list(counts), y=list(counts.values()), marker_color=[MODE_COLORS[m] for m in counts],
                text=[f"{v:,}" for v in counts.values()], textposition="outside",
                hovertemplate="%{x}: %{y:,} layer-steps<extra></extra>",
            ))
            fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis=dict(type="log", title="Layer-steps (log)", exponentformat="power"))
            st.plotly_chart(fig, width="stretch", theme="streamlit", key=f"{k}_classes")


# ---------------------------------------------------------------- section 5
def section_summary(results: list[dict], df: pd.DataFrame, cfg: tuple[str, str], exp: Experiment) -> None:
    st.subheader("Summary")
    by_cfg = {(r["optimizer"], r["ghd_mode"]): r for _, r in df.iterrows()}

    best_label, best_steps = None, None
    for c, row in by_cfg.items():
        v = _full_steps(row, "80")
        if v is not None and (best_steps is None or v < best_steps):
            best_label, best_steps = _label(c), v

    best_gain, gain_label = None, None
    for ghd_cfg, base_cfg in [p for pairs in DELTA_ROWS.values() for p in pairs]:
        g, b = _full_steps(by_cfg.get(ghd_cfg), "80"), _full_steps(by_cfg.get(base_cfg), "80")
        if g is not None and b is not None and (best_gain is None or b - g > best_gain):
            best_gain, gain_label = b - g, f"{_label(ghd_cfg)} vs {_label(base_cfg)}"

    runs = _runs(results, cfg)
    sums = [r.get("ghd_summary") or {} for r in runs]
    sdepth = [s["mean_sdepth_early"] for s in sums if s.get("mean_sdepth_early") is not None]
    fa = [s["false_alarm_rate"] for s in sums if s.get("false_alarm_rate") is not None]

    gain_name = "GHD Improvement"
    gain_note = f"Δ steps→80% · {gain_label}" if gain_label else "needs a GHD config and its baseline at 80%"
    if exp.gain_card:
        gain_name, gain_note = exp.gain_card

    cols = st.columns(4)
    cards = [
        _card("Best Convergence", f"{best_steps:.0f} steps" if best_steps is not None else "—",
              f"to 80% · {best_label}" if best_label else "no config reached 80% on every seed"),
        _card(gain_name,
              "—" if best_gain is None else f"{-best_gain:+.0f} steps".replace("-", "−"),
              gain_note,
              None if best_gain is None else GOOD if best_gain > 0 else BAD),
        _card("Detection Rate", _sci(float(np.mean(sdepth))) if sdepth else "—",
              f"Avg S_depth layers 1-5 · {_label(cfg)}"),
        _card("False Alarm Rate", f"{np.mean(fa) * 100:.1f}%" if fa and cfg[1] != "none" else "—",
              f"Healthy layer-steps flagged · {_label(cfg)}"),
    ]
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


# ---------------------------------------------------------------- page
def _steps_note(results: list[dict]) -> str:
    """e.g. 'SGD/LARS/LNGD 10000 steps · AdamW 5000 steps', from the results present."""
    steps: dict[int, list[str]] = {}
    for r in results:
        opt = config_label(r["config"]["optimizer"], "none")
        names = steps.setdefault(r["config"]["n_steps"], [])
        if opt not in names:
            names.append(opt)
    if not steps:
        return "steps pending"
    if len(steps) == 1:
        return f"{next(iter(steps))} steps"
    return " · ".join(f"{'/'.join(v)} {k} steps" for k, v in sorted(steps.items(), reverse=True))


def render_experiment(exp: Experiment, in_progress: bool) -> None:
    results = _load_exp(exp)
    k = exp.key
    st.markdown(f'<div class="ghd-sub">{exp.num_layers}-layer Sigmoid MLP on MNIST · Xavier gain {exp.init_gain:g} · '
                f'{_steps_note(results)} · 3 seeds</div>', unsafe_allow_html=True)
    st.caption(exp.note)
    if in_progress and exp.results_dir == RESULTS_DIR:
        st.info(f"Experiment runs in progress: {len(results)} of {len(CONFIGS) * len(SEEDS)} results available. "
                f"This page refreshes every {REFRESH_SECONDS} s.", icon="⏳")
    if not results:
        cols = st.columns(4)
        for col, name in zip(cols, ["Best Convergence", "GHD Improvement", "Detection Rate", "False Alarm Rate"]):
            col.markdown(_card(name, "Running…"), unsafe_allow_html=True)
        st.caption("No results yet. Start the experiment with `python main.py --exp 1 --workers 8`.")
        return

    with st.container(border=True):
        section_comparisons(results, k, exp.table_configs)
    df = aggregate(results)

    available = [c for c in CONFIGS if _runs(results, c)]
    with st.container(border=True):
        st.subheader("Gradient Health Evidence")
        default = next((i for i, c in enumerate(available) if c[1] == "ai"), 0)
        by_label = {_label(c): c for c in available}
        cfg = by_label[st.selectbox("Configuration", list(by_label), index=default, key=f"{k}_cfg")]
        section_gradient_health(results, cfg, k)
    if cfg[1] == "ai":
        with st.container(border=True):
            section_ai_controller(results, cfg, exp, k)
    with st.container(border=True):
        section_summary(results, df, cfg, exp)


def _body(auto_refresh: bool) -> None:
    in_progress = runs_in_progress()
    if auto_refresh and not in_progress:
        st.rerun()  # runs finished: full rerun turns auto-refresh off
    for tab, exp in zip(st.tabs([e.label for e in EXPERIMENTS]), EXPERIMENTS):
        with tab:
            render_experiment(exp, in_progress)


def render() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("GHD Experiment Results — Convergence Speed Comparison")
    auto = runs_in_progress()
    st.fragment(run_every=REFRESH_SECONDS if auto else None)(_body)(auto)
