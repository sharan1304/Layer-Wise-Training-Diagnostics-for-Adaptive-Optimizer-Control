"""Run rigorous multi-seed MNIST optimizer comparisons.

This is the reproducibility runner for paper-style results:
- seeds: 42, 123, 999
- longer training: 1000 steps by default
- baselines plus all four rule-controller modes
- reports mean +/- std for every numeric metric
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from metrics import full_report
from train_mnist import train_one_optimizer

DEFAULT_SEEDS = [42, 123, 999]
DEFAULT_OPTIMIZERS = [
    "Adam",
    "AdamW",
    "RMSProp",
    "SGD",
    "AI-SAFE-HYBRID",
    "AI-BALANCED-HYBRID",
    "AI-AGGRESSIVE-HYBRID",
    "AI-EXTREME-HYBRID",
]
METRIC_ORDER = [
    "final_loss",
    "final_accuracy",
    "early_grad_mean",
    "vanishing_ratio",
    "log10_vanishing_ratio",
    "gradient_stability",
    "convergence_step",
    "early_gain_vs_adam",
    "early_gain_vs_sgd",
    "vanishing_gain_vs_adam",
    "vanishing_gain_vs_sgd",
    "trust_activations",
    "trust_clip_utilization_max",
    "trust_clip_utilization_mean",
    "starvation_events",
    "starvation_recoveries",
    "starvation_activation_rate",
    "recovery_success_rate",
]


def numeric_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float, np.floating)):
        out = float(value)
        if np.isfinite(out):
            return out
    return None


def summarize(seed_reports: dict[int, dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, float]]]:
    optimizers = sorted({name for report in seed_reports.values() for name in report})
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for optimizer in optimizers:
        summary[optimizer] = {}
        for metric in METRIC_ORDER:
            values = [
                numeric_or_none(seed_reports[seed][optimizer].get(metric))
                for seed in seed_reports
                if optimizer in seed_reports[seed]
            ]
            clean = [v for v in values if v is not None]
            if not clean:
                continue
            summary[optimizer][metric] = {
                "mean": mean(clean),
                "std": pstdev(clean) if len(clean) > 1 else 0.0,
                "n": float(len(clean)),
            }
    return summary


def write_summary_csv(summary: dict[str, dict[str, dict[str, float]]], path: Path) -> None:
    headers = ["optimizer"]
    for metric in METRIC_ORDER:
        headers.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for optimizer, metrics in summary.items():
            row: list[Any] = [optimizer]
            for metric in METRIC_ORDER:
                if metric in metrics:
                    row.extend([metrics[metric]["mean"], metrics[metric]["std"]])
                else:
                    row.extend(["N/A", "N/A"])
            writer.writerow(row)


def print_compact_summary(summary: dict[str, dict[str, dict[str, float]]]) -> None:
    print("\nMEAN +/- STD ACROSS SEEDS")
    print("=" * 118)
    print(
        f"{'Optimizer':<24} {'Loss':>16} {'Acc':>16} {'Log10VR':>16} "
        f"{'EarlyGrad':>20} {'TrustClip':>12} {'MaxUtil':>12}"
    )
    print("-" * 118)
    for optimizer, metrics in summary.items():
        def fmt(metric: str, prec: int = 4) -> str:
            if metric not in metrics:
                return "N/A"
            m = metrics[metric]["mean"]
            s = metrics[metric]["std"]
            if metric == "early_grad_mean":
                return f"{m:.2e} +/- {s:.1e}"
            return f"{m:.{prec}f} +/- {s:.{prec}f}"
        print(
            f"{optimizer:<24} "
            f"{fmt('final_loss'):>16} "
            f"{fmt('final_accuracy'):>16} "
            f"{fmt('log10_vanishing_ratio'):>16} "
            f"{fmt('early_grad_mean'):>20} "
            f"{fmt('trust_activations', 1):>12} "
            f"{fmt('trust_clip_utilization_max', 3):>12}"
        )
    print("=" * 118)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", default="results/multiseed_1000")
    parser.add_argument("--include-mlp", action="store_true")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    optimizers = list(DEFAULT_OPTIMIZERS)
    if args.include_mlp:
        optimizers.extend([
            "AI-MLP-BALANCED-HYBRID",
            "AI-MLP-AGGRESSIVE-HYBRID",
            "AI-MLP-FT-BALANCED-HYBRID",
            "AI-MLP-FT-AGGRESSIVE-HYBRID",
        ])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_reports: dict[int, dict[str, dict[str, Any]]] = {}

    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Steps: {args.steps} | Layers: {args.layers} | Hidden: {args.hidden_dim}")
    print(f"Optimizers: {', '.join(optimizers)}")

    for seed in args.seeds:
        print(f"\n===== SEED {seed} =====")
        results = {}
        for optimizer in optimizers:
            print(f"\n-- {optimizer} | seed {seed} --")
            results[optimizer] = train_one_optimizer(
                optimizer,
                num_steps=args.steps,
                batch_size=args.batch_size,
                device=device,
                num_layers=args.layers,
                hidden_dim=args.hidden_dim,
                seed=seed,
                log_interval=max(args.steps // 4, 1),
            )
        report = full_report(results)
        seed_reports[seed] = report
        (out / f"metrics_seed_{seed}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = summarize(seed_reports)
    (out / "summary_mean_std.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_csv(summary, out / "summary_mean_std.csv")
    print_compact_summary(summary)
    print(f"\nSaved multi-seed reports to {out}")


if __name__ == "__main__":
    main()
