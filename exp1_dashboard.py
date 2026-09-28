"""Streamlit panels for Experiment 1 (10-layer sigmoid MLP, optimizers x GHD).

Rendered as a tab inside ``streamlit_app.py``; reads results/mlp_mnist_*.json.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ghd.experiment import CONFIGS, config_label
from ghd.summary import TABLE_HEADER, aggregate, load_results, table_rows


RESULTS_DIR = Path("results")

# Base / GHD-Rules (lighter) / GHD-AI (darker) per optimizer family.
SHADES = {
    "sgd": {"none": "#6b7280", "rules": "#b9bec7", "ai": "#1f2937"},
    "adamw": {"none": "#2563eb", "rules": "#93c5fd", "ai": "#1e3a8a"},
    "lars": {"none": "#16a34a", "rules": "#86efac", "ai": "#14532d"},
    "lngd": {"none": "#ea580c", "rules": "#fdba74", "ai": "#7c2d12"},
}
MODE_COLORS = {
    "healthy": "#14b8a6",
    "vanishing": "#dc2626",
    "exploding": "#f97316",
    "oscillating": "#eab308",
    "noisy": "#9333ea",
    "recovering": "#60a5fa",
    "warmup": "#9ca3af",
}
LABELS = [config_label(o, g) for o, g in CONFIGS]
COLOR_SCALE = alt.Scale(domain=LABELS, range=[SHADES[o][g] for o, g in CONFIGS])
MODE_SCALE = alt.Scale(domain=list(MODE_COLORS), range=list(MODE_COLORS.values()))


def _fingerprint(results_dir: Path) -> tuple:
    return tuple((p.name, p.stat().st_mtime) for p in sorted(results_dir.glob("mlp_mnist_*_seed*.json")))


@st.cache_data(show_spinner="Loading Experiment 1 results...")
def _load(results_dir: str, _fp: tuple) -> list[dict]:
    return load_results(results_dir, with_logs=True)


def _curves_frame(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        label = config_label(r["config"]["optimizer"], r["config"]["ghd_mode"])
        for step, acc in zip(r["val_steps"], r["val_acc_curve"]):
            rows.append({"config": label, "seed": r["config"]["seed"], "step": step, "val_acc": acc})
    df = pd.DataFrame(rows)
    return df.groupby(["config", "step"], as_index=False).agg(
        val_acc=("val_acc", "mean"), val_std=("val_acc", "std"), seeds=("seed", "nunique")
    ).fillna({"val_std": 0.0})


def _log_frame(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        seed = r["config"]["seed"]
        for entry in r.get("ghd_logs", []):
            for name, rec in entry["layers"].items():
                rows.append({
                    "seed": seed, "step": entry["step"], "layer": int(name[1:]), "mode": rec["mode"],
                    "strength": rec["strength"], "s_depth": rec["s_depth"],
                })
    return pd.DataFrame(rows)


def _runs_for(results: list[dict], opt: str, ghd: str) -> list[dict]:
    return [r for r in results if r["config"]["optimizer"] == opt and r["config"]["ghd_mode"] == ghd]


def _available(results: list[dict]) -> list[tuple[str, str]]:
    have = {(r["config"]["optimizer"], r["config"]["ghd_mode"]) for r in results}
    return [c for c in CONFIGS if c in have]


# ---------------------------------------------------------------- panels
def panel_convergence(results: list[dict]) -> None:
    st.subheader("1 · Convergence Race")
    curves = _curves_frame(results)
    present = [l for l in LABELS if l in set(curves["config"])]
    chosen = st.multiselect("Configurations", present, default=present, key="exp1_race_configs")
    data = curves[curves["config"].isin(chosen)]
    if data.empty:
        st.info("Select at least one configuration.")
        return
    lines = alt.Chart(data).mark_line(strokeWidth=2).encode(
        x=alt.X("step:Q", title="Training step"),
        y=alt.Y("val_acc:Q", title="Validation accuracy (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("config:N", scale=COLOR_SCALE, sort=LABELS, title=None,
                        legend=alt.Legend(orient="bottom", columns=5)),
        tooltip=["config", "step", alt.Tooltip("val_acc:Q", format=".2f", title="mean acc %"),
                 alt.Tooltip("val_std:Q", format=".2f", title="std"), "seeds"],
    )
    thresholds = alt.Chart(pd.DataFrame({"y": [70, 80, 90]})).mark_rule(
        strokeDash=[6, 4], color="#9ca3af"
    ).encode(y="y:Q")
    labels = alt.Chart(pd.DataFrame({"y": [70, 80, 90], "t": ["70%", "80%", "90%"]})).mark_text(
        align="left", dx=4, dy=-6, color="#9ca3af"
    ).encode(y="y:Q", x=alt.value(0), text="t:N")
    st.altair_chart((thresholds + labels + lines).properties(height=420), width="stretch")
    st.caption("Mean across seeds. Lighter shade = GHD-Rules, darker shade = GHD-AI.")


def panel_threshold_table(results: list[dict]) -> None:
    st.subheader("2 · Steps to Threshold")
    df = aggregate(results)
    rows = table_rows(df, bold="**")
    md = ["| " + " | ".join(TABLE_HEADER) + " |", "|" + "---|" * len(TABLE_HEADER)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    st.markdown("\n".join(md))
    seeds = sorted({s for ss in df["seeds"] for s in ss})
    st.caption(
        f"Mean ± std over seeds {seeds}. Steps count only the seeds that reached the threshold; "
        "(k/n) marks configs where only k of n seeds did. AUC = normalised area under the validation-accuracy "
        "curve. Bold = best per column (for steps: fastest among configs that reached it on every seed)."
    )


def _seed_picker(runs: list[dict], key: str) -> list[dict]:
    seeds = sorted(r["config"]["seed"] for r in runs)
    seed = st.selectbox("Seed", seeds, key=key)
    return [r for r in runs if r["config"]["seed"] == seed]


def panel_heatmap(results: list[dict]) -> None:
    st.subheader("3 · Gradient Health Heatmap")
    ghd_opts = [o for o in ["sgd", "adamw", "lars", "lngd"]
                if _runs_for(results, o, "rules") or _runs_for(results, o, "ai")]
    if not ghd_opts:
        st.info("No GHD runs yet.")
        return
    c1, c2, c3 = st.columns([1, 1, 1])
    opt = c1.selectbox("Optimizer", ghd_opts, format_func=lambda o: config_label(o, "none"), key="exp1_hm_opt")
    modes = [g for g in ["rules", "ai"] if _runs_for(results, opt, g)]
    with c2:
        ghd = st.radio("Detector", modes, format_func=lambda g: {"rules": "GHD-Rules", "ai": "GHD-AI"}[g],
                       horizontal=True, key="exp1_hm_mode")
    with c3:
        runs = _seed_picker(_runs_for(results, opt, ghd), "exp1_hm_seed")
    logs = _log_frame(runs)
    if logs.empty:
        st.info("This run has no GHD logs.")
        return
    chart = alt.Chart(logs).mark_rect().encode(
        x=alt.X("step:O", title="Training step",
                axis=alt.Axis(labelExpr="datum.value % 500 === 0 ? datum.value : ''", labelAngle=0, ticks=False)),
        y=alt.Y("layer:O", title="Layer", sort="ascending"),
        color=alt.Color("mode:N", scale=MODE_SCALE, title="Mode"),
        tooltip=["step", "layer", "mode", alt.Tooltip("strength:Q", format=".3f"),
                 alt.Tooltip("s_depth:Q", format=".2e")],
    ).properties(height=320)
    st.altair_chart(chart, width="stretch")
    st.caption("Sampled every 10 steps. Gray = warm-up (no detection before step 100). "
               "Light blue = recovering (spike right after a vanishing episode).")


def panel_sdepth(results: list[dict]) -> None:
    st.subheader("4 · S_depth Ladder")
    configs = [c for c in _available(results) if any(r.get("ghd_logs") for r in _runs_for(results, *c))]
    if not configs:
        st.info("No GHD logs yet.")
        return
    chosen = st.selectbox("Configuration", configs, format_func=lambda c: config_label(*c), key="exp1_sd_cfg")
    logs = _log_frame(_runs_for(results, *chosen))
    logs = logs[logs["step"] > 100]
    ladder = logs.groupby("layer", as_index=False)["s_depth"].mean()
    ladder["s_depth"] = ladder["s_depth"].clip(lower=1e-12)
    floor = float(min(ladder["s_depth"].min(), 1e-2) / 3)
    ladder["base"] = floor
    x_scale = alt.Scale(type="log", domain=[floor, max(1.5, float(ladder["s_depth"].max()) * 1.5)])
    bars = alt.Chart(ladder).mark_bar(color="#6366f1", cornerRadiusEnd=3).encode(
        y=alt.Y("layer:O", title="Layer", sort="ascending"),
        x=alt.X("s_depth:Q", scale=x_scale, title="Mean S_depth (log scale)"),
        x2="base:Q",
        tooltip=["layer", alt.Tooltip("s_depth:Q", format=".2e")],
    )
    rule = alt.Chart(pd.DataFrame({"x": [0.01]})).mark_rule(color="#dc2626", strokeWidth=2).encode(x="x:Q")
    st.altair_chart((bars + rule).properties(height=320), width="stretch")
    st.caption("S_depth = per-parameter gradient norm of the layer relative to the output layer, averaged over "
               "logged steps after warm-up and over seeds. Red line: structural-vanishing threshold (0.01). "
               "Baselines are logged by a passive (observe-only) hook.")


def panel_ai_controller(results: list[dict]) -> None:
    ai_opts = [o for o in ["sgd", "adamw", "lars", "lngd"] if _runs_for(results, o, "ai")]
    if not ai_opts:
        return
    st.subheader("5 · AI Controller")
    c1, c2 = st.columns([1, 2])
    opt = c1.selectbox("GHD-AI run", ai_opts, format_func=lambda o: config_label(o, "ai"), key="exp1_ai_opt")
    runs = _runs_for(results, opt, "ai")
    if any(r["config"].get("controller") == "fallback_rules" for r in runs):
        st.warning("Some of these runs had no controller checkpoint and fell back to rule-based decisions.")
    logs = _log_frame(runs)
    logs = logs[logs["mode"] != "warmup"]
    if logs.empty:
        st.info("No post-warm-up logs.")
        return
    layers = sorted(logs["layer"].unique())
    picked = c2.multiselect("Layers", layers, default=layers[:3], key="exp1_ai_layers")

    strength = logs[logs["layer"].isin(picked)].groupby(["layer", "step"], as_index=False)["strength"].mean()
    line = alt.Chart(strength).mark_line().encode(
        x=alt.X("step:Q", title="Training step"),
        y=alt.Y("strength:Q", title="Correction strength", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("layer:N", title="Layer"),
        tooltip=["layer", "step", alt.Tooltip("strength:Q", format=".3f")],
    ).properties(height=280)

    dist = logs.groupby(["layer", "mode"], as_index=False).size()
    bars = alt.Chart(dist).mark_bar().encode(
        y=alt.Y("layer:O", title="Layer", sort="ascending"),
        x=alt.X("size:Q", stack="normalize", title="Share of logged steps", axis=alt.Axis(format="%")),
        color=alt.Color("mode:N", scale=MODE_SCALE, title="Mode"),
        tooltip=["layer", "mode", alt.Tooltip("size:Q", title="logged steps")],
    ).properties(height=280)

    a, b = st.columns(2)
    with a:
        st.markdown("**Correction strength over steps** (mean over seeds)")
        st.altair_chart(line, width="stretch")
    with b:
        st.markdown("**Mode distribution per layer**")
        st.altair_chart(bars, width="stretch")


def render_experiment1(results_dir: Path = RESULTS_DIR) -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Experiment 1 &mdash; Convergence Race</h1>
            <p>10-layer sigmoid MLP (784&rarr;256&times;9&rarr;10) on MNIST. SGD, AdamW, LARS and LNGD, each alone
            and with the Gradient Health Detector (GHD-Rules / GHD-AI) correcting per-layer gradients.
            Primary metric: training steps to reach 70 / 80 / 90% validation accuracy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    results = _load(str(results_dir), _fingerprint(results_dir))
    if not results:
        st.info("No Experiment 1 results yet. Run the experiment, then refresh this page.")
        st.code("python main.py --exp 1 --workers 8", language="bash")
        return
    n_expected = len(CONFIGS) * 3
    if len(results) < n_expected:
        st.caption(f"{len(results)} of {n_expected} runs found. Missing runs are resumed by `python main.py --exp 1`.")

    for panel in (panel_convergence, panel_threshold_table, panel_heatmap, panel_sdepth, panel_ai_controller):
        with st.container(border=True):
            panel(results)
