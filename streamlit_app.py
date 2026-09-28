"""Streamlit dashboard for the Layer-Aware AI Optimizer.

Run:
    streamlit run app.py

Shows the real experiment output only: how the AI-hybrid optimizer modes
compare against Adam / AdamW / RMSProp / SGD on the vanishing-gradient
MNIST task, aggregated mean +/- std across seeds.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="AI Optimizer Results",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CSS = """
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        visibility: hidden;
        height: 0;
    }

    .hero {
        background: linear-gradient(120deg, #4338ca 0%, #6d28d9 55%, #9333ea 100%);
        border-radius: 18px;
        padding: 32px 36px;
        color: #ffffff;
        margin-bottom: 1.75rem;
        box-shadow: 0 10px 30px rgba(76, 29, 149, 0.25);
    }
    .hero h1 {
        margin: 0 0 6px 0;
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: -0.01em;
    }
    .hero p {
        margin: 0;
        font-size: 1.02rem;
        opacity: 0.92;
        max-width: 760px;
    }

    div[data-testid="stMetric"] {
        padding: 6px 4px;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.02);
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


MULTISEED_RESULTS_PATH = Path("results/multiseed_1000/summary_mean_std.csv")
FALLBACK_RESULTS_PATH = Path("results/metrics_report.csv")


def _controller_label(name: str) -> str:
    if name.startswith("AI-MLP-FT"):
        return "Fine-Tuned MLP"
    if name.startswith("AI-MLP"):
        return "Distilled MLP"
    if name.startswith("AI"):
        return "AI Optimizer"
    return "Classical"


def _mode_label(name: str) -> str:
    parts = name.split("-")
    if name.startswith("AI-MLP-FT") and len(parts) >= 4:
        return parts[3].title()
    if name.startswith("AI-MLP") and len(parts) >= 3:
        return parts[2].title()
    if name.startswith("AI-") and len(parts) >= 2:
        return parts[1].title()
    return "Baseline"


def _mean_std(df: pd.DataFrame, mean_col: str, std_col: str, fmt: str) -> pd.Series:
    return df.apply(
        lambda row: f"{row[mean_col]:{fmt}} ± {row[std_col]:{fmt}}"
        if pd.notna(row[mean_col])
        else "N/A",
        axis=1,
    )


def load_multiseed_report(path: Path = MULTISEED_RESULTS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df.insert(1, "controller", df["optimizer"].map(_controller_label))
    df.insert(2, "mode", df["optimizer"].map(_mode_label))
    df["rank_accuracy"] = df["final_accuracy_mean"].rank(method="min", ascending=False).astype(int)
    df["rank_vanishing"] = df["log10_vanishing_ratio_mean"].rank(method="min", ascending=True).astype(int)
    return df.sort_values(["rank_vanishing", "rank_accuracy"])


def load_fallback_report(path: Path = FALLBACK_RESULTS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df.insert(1, "controller", df["optimizer"].map(_controller_label))
    df.insert(2, "mode", df["optimizer"].map(_mode_label))
    for metric, mean_col in [
        ("final_accuracy", "final_accuracy_mean"),
        ("final_loss", "final_loss_mean"),
        ("log10_vanishing_ratio", "log10_vanishing_ratio_mean"),
        ("trust_activations", "trust_activations_mean"),
        ("starvation_activation_rate", "starvation_activation_rate_mean"),
        ("recovery_success_rate", "recovery_success_rate_mean"),
    ]:
        df[mean_col] = pd.to_numeric(df[metric], errors="coerce")
        df[mean_col.replace("_mean", "_std")] = np.nan
    df["rank_accuracy"] = df["final_accuracy_mean"].rank(method="min", ascending=False).astype(int)
    df["rank_vanishing"] = df["log10_vanishing_ratio_mean"].rank(method="min", ascending=True).astype(int)
    return df.sort_values(["rank_vanishing", "rank_accuracy"])


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Layer-Aware AI Optimizer &mdash; Results</h1>
            <p>Real MNIST measurements: a 12-layer sigmoid MLP deliberately built to vanish gradients,
            optimized with Adam / AdamW / RMSProp / SGD versus the AI-hybrid controller modes
            (Safe, Balanced, Aggressive, Extreme).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_headline_metrics(metrics: pd.DataFrame, seed_note: str) -> None:
    baselines = metrics[metrics["controller"] == "Classical"]
    ai_rows = metrics[metrics["controller"] != "Classical"]
    if ai_rows.empty or baselines.empty:
        return

    best_ai = ai_rows.loc[ai_rows["final_accuracy_mean"].idxmax()]
    best_baseline = baselines.loc[baselines["final_accuracy_mean"].idxmax()]
    acc_gain = best_ai["final_accuracy_mean"] / max(best_baseline["final_accuracy_mean"], 1e-8)
    vr_gain = best_baseline["log10_vanishing_ratio_mean"] - best_ai["log10_vanishing_ratio_mean"]

    cols = st.columns(4)
    cols[0].metric(
        "Best AI Optimizer",
        best_ai["mode"],
        f"{best_ai['final_accuracy_mean']:.1%} accuracy",
        help=best_ai["optimizer"],
    )
    cols[1].metric(
        "Best Classical Optimizer",
        best_baseline["optimizer"],
        f"{best_baseline['final_accuracy_mean']:.1%} accuracy",
    )
    cols[2].metric("Accuracy Gain", f"{acc_gain:.2f}x", "AI vs best classical")
    cols[3].metric(
        "Gradient-Flow Gain",
        f"{vr_gain:+.2f}",
        "log10, lower vanishing ratio",
        help="Reduction in log10(late-layer / early-layer gradient norm) vs the best classical optimizer",
    )
    st.caption(seed_note)


def render_comparison_table(metrics: pd.DataFrame, has_std: bool) -> None:
    if has_std:
        display = pd.DataFrame(
            {
                "Optimizer": metrics["optimizer"],
                "Type": metrics["controller"],
                "Mode": metrics["mode"],
                "Accuracy": _mean_std(metrics, "final_accuracy_mean", "final_accuracy_std", ".2%"),
                "Final Loss": _mean_std(metrics, "final_loss_mean", "final_loss_std", ".4f"),
                "Log10 Vanishing Ratio": _mean_std(
                    metrics, "log10_vanishing_ratio_mean", "log10_vanishing_ratio_std", ".3f"
                ),
                "Trust Clips": _mean_std(metrics, "trust_activations_mean", "trust_activations_std", ".1f"),
                "Starvation Rate": _mean_std(
                    metrics, "starvation_activation_rate_mean", "starvation_activation_rate_std", ".3f"
                ),
                "Recovery Rate": _mean_std(
                    metrics, "recovery_success_rate_mean", "recovery_success_rate_std", ".3f"
                ),
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        display = pd.DataFrame(
            {
                "Optimizer": metrics["optimizer"],
                "Type": metrics["controller"],
                "Mode": metrics["mode"],
                "Accuracy": metrics["final_accuracy_mean"],
                "Final Loss": metrics["final_loss_mean"],
                "Log10 Vanishing Ratio": metrics["log10_vanishing_ratio_mean"],
                "Trust Clips": metrics["trust_activations_mean"],
                "Starvation Rate": metrics["starvation_activation_rate_mean"],
                "Recovery Rate": metrics["recovery_success_rate_mean"],
            }
        )
        styled = display.style.format(
            {
                "Accuracy": "{:.2%}",
                "Final Loss": "{:.4f}",
                "Log10 Vanishing Ratio": "{:.3f}",
                "Trust Clips": "{:.0f}",
                "Starvation Rate": "{:.3f}",
                "Recovery Rate": "{:.2%}",
            },
            na_rep="N/A",
        )
        styled = styled.highlight_max(subset=["Accuracy"], color="#dcfce7")
        styled = styled.highlight_min(subset=["Log10 Vanishing Ratio"], color="#dbeafe")
        st.dataframe(styled, use_container_width=True, hide_index=True)


def render_visualizations(metrics: pd.DataFrame) -> None:
    chart_df = metrics[["optimizer", "controller", "final_accuracy_mean", "log10_vanishing_ratio_mean"]].copy()
    chart_df.columns = ["optimizer", "controller", "accuracy", "log10_vr"]

    color_scale = alt.Scale(
        domain=["Classical", "AI Optimizer", "Distilled MLP", "Fine-Tuned MLP"],
        range=["#94a3b8", "#7c3aed", "#0ea5e9", "#059669"],
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Final Accuracy by Optimizer**")
        acc_chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("optimizer:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-35)),
                y=alt.Y("accuracy:Q", title="Accuracy", axis=alt.Axis(format="%")),
                color=alt.Color("controller:N", scale=color_scale, title="Type"),
                tooltip=["optimizer", alt.Tooltip("accuracy:Q", format=".2%")],
            )
            .properties(height=320)
        )
        st.altair_chart(acc_chart, use_container_width=True)

    with col_b:
        st.markdown("**Gradient Vanishing Ratio by Optimizer (lower = better)**")
        vr_chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("optimizer:N", sort="y", title=None, axis=alt.Axis(labelAngle=-35)),
                y=alt.Y("log10_vr:Q", title="log10(late / early gradient norm)"),
                color=alt.Color("controller:N", scale=color_scale, title="Type"),
                tooltip=["optimizer", alt.Tooltip("log10_vr:Q", format=".3f")],
            )
            .properties(height=320)
        )
        st.altair_chart(vr_chart, use_container_width=True)


def main() -> None:
    from exp1_dashboard import render_experiment1

    tab_exp1, tab_legacy = st.tabs(["Experiment 1 · Convergence Race", "Layer-Aware Optimizer"])
    with tab_exp1:
        render_experiment1()
    with tab_legacy:
        render_legacy()


def render_legacy() -> None:
    render_hero()

    metrics = load_multiseed_report()
    has_std = not metrics.empty
    seed_note = "Aggregated across 3 seeds (42, 123, 999), 1000 training steps each."
    if metrics.empty:
        metrics = load_fallback_report()
        seed_note = "Single seed (42), 300 training steps. Run `python multi_seed_experiments.py` for the full mean ± std comparison."

    if metrics.empty:
        st.info("No results found yet. Run the experiment first, then refresh this page.")
        st.code("python multi_seed_experiments.py", language="bash")
        return

    with st.container(border=True):
        render_headline_metrics(metrics, seed_note)

    with st.container(border=True):
        st.subheader("Optimizer Comparison")
        render_comparison_table(metrics, has_std)
        st.caption(
            "Lower final loss and lower log10 vanishing ratio are better; higher accuracy and recovery rate are better."
        )

    with st.container(border=True):
        st.subheader("Visualization")
        render_visualizations(metrics)


if __name__ == "__main__":
    main()
