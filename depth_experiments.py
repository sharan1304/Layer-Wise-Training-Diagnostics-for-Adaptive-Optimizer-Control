"""Run depth sweep experiments for gradient-starvation recovery."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from metrics import full_report
from train_mnist import train_one_optimizer


def main() -> None:
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    depths = [8, 12, 16, 20]
    optimizers = ["Adam", "SGD", "RMSProp", "AI-AGGRESSIVE-HYBRID"]
    all_reports = {}

    for depth in depths:
        print(f"\n===== DEPTH {depth} =====")
        results = {}
        for name in optimizers:
            print(f"\n-- {name} --")
            results[name] = train_one_optimizer(
                name,
                num_steps=220,
                device=device,
                num_layers=depth,
                hidden_dim=96,
            )
        ai_log = results["AI-AGGRESSIVE-HYBRID"].get("layer_log", [])
        report = full_report(results, ai_log)
        all_reports[str(depth)] = report
        for name, metrics in report.items():
            print(
                f"{name:<22} loss={metrics['final_loss']:.4f} "
                f"early={metrics['early_grad_mean']:.3e} "
                f"log10VR={metrics['log10_vanishing_ratio']:.2f} "
                f"gainAdam={metrics['early_gain_vs_adam']:.2f}"
            )

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "depth_experiments.json").write_text(
        json.dumps(all_reports, indent=2),
        encoding="utf-8",
    )
    print("\nSaved depth sweep to results/depth_experiments.json")


if __name__ == "__main__":
    main()
