"""Compare AIOptimizer against common optimizers on a vanishing-gradient task."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ai_optimizer import AIOptimizer
from metrics import full_report, format_float
from mlp_policy import checkpoint_is_compatible


class DeepSigmoidMLP(nn.Module):
    """20-layer sigmoid MLP deliberately chosen to expose vanishing gradients."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        num_classes: int = 10,
        num_layers: int = 20,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()])
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.reshape(x.size(0), -1))

    def get_layer_grad_norms(self) -> list[float]:
        norms: list[float] = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                if module.weight.grad is None:
                    norms.append(0.0)
                else:
                    norms.append(float(module.weight.grad.norm().item()))
        return norms


MLP_CHECKPOINT = "results/mlp_policy.pt"
FINETUNED_MLP_CHECKPOINT = "results/mlp_policy_finetuned.pt"


def make_optimizer(name: str, params):
    if name == "AI Optimizer":
        return AIOptimizer(params, lr=1e-3, mode="balanced", detector="hybrid")
    if name.startswith("AI-MLP-FT-"):
        parts = name.split("-")
        mode = parts[3].lower()
        detector = parts[4].lower() if len(parts) > 4 else "hybrid"
        return AIOptimizer(
            params,
            lr=1e-3,
            mode=mode,  # type: ignore[arg-type]
            detector=detector,  # type: ignore[arg-type]
            controller="mlp",
            mlp_checkpoint=FINETUNED_MLP_CHECKPOINT,
        )
    if name.startswith("AI-MLP-"):
        parts = name.split("-")
        mode = parts[2].lower()
        detector = parts[3].lower() if len(parts) > 3 else "hybrid"
        return AIOptimizer(
            params,
            lr=1e-3,
            mode=mode,  # type: ignore[arg-type]
            detector=detector,  # type: ignore[arg-type]
            controller="mlp",
            mlp_checkpoint=MLP_CHECKPOINT,
        )
    if name.startswith("AI-"):
        parts = name.split("-")
        mode = parts[1].lower()
        detector = parts[2].lower() if len(parts) > 2 else "hybrid"
        return AIOptimizer(
            params,
            lr=1e-3,
            mode=mode,  # type: ignore[arg-type]
            detector=detector,  # type: ignore[arg-type]
        )
    if name == "Adam":
        return torch.optim.Adam(params, lr=1e-3)
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    if name == "RMSProp":
        return torch.optim.RMSprop(params, lr=1e-3)
    if name == "SGD":
        return torch.optim.SGD(params, lr=1e-2, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def train_one_optimizer(
    optimizer_name: str,
    num_steps: int = 200,
    batch_size: int = 64,
    device: str = "cpu",
    num_layers: int = 12,
    hidden_dim: int = 96,
    seed: int = 42,
    log_interval: int = 50,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepSigmoidMLP(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    opt = make_optimizer(optimizer_name, model.parameters())

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    data_iter = iter(loader)

    loss_history: list[float] = []
    accuracy_history: list[float] = []
    grad_norm_history: list[list[float]] = []

    model.train()
    for step in range(num_steps):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, labels = next(data_iter)

        images = images.to(device)
        labels = labels.to(device)

        opt.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        if isinstance(opt, AIOptimizer):
            opt.step(loss=loss.item())
        else:
            opt.step()

        loss_val = float(loss.item())
        acc = float((logits.argmax(1) == labels).float().mean().item())
        grad_norms = model.get_layer_grad_norms()
        loss_history.append(loss_val)
        accuracy_history.append(acc)
        grad_norm_history.append(grad_norms)

        if log_interval > 0 and (step + 1) % log_interval == 0:
            early_norm = float(np.mean(grad_norms[:5])) if grad_norms else 0.0
            print(
                f"{optimizer_name:<20} Step {step + 1:3d} | "
                f"Loss: {loss_val:.4f} | Acc: {acc:.3f} | "
                f"EarlyGrad: {early_norm:.3e}"
            )

    return_data = {
        "loss": loss_history,
        "accuracy": accuracy_history,
        "grad_norms": grad_norm_history,
        "seed": seed,
    }
    if isinstance(opt, AIOptimizer):
        return_data["layer_log"] = opt.layer_log
    return return_data


def print_results_table(results: dict) -> None:
    print("\n" + "=" * 65)
    print(f"{'Optimizer':<20} {'FinalLoss':>10} {'Accuracy':>10} {'EarlyGrad':>12}")
    print("=" * 65)
    for name, data in results.items():
        final_loss = data["loss"][-1]
        final_acc = data["accuracy"][-1]
        early_norm = (
            float(np.mean([np.mean(norms[:5]) for norms in data["grad_norms"][-10:]]))
            if data["grad_norms"]
            else 0.0
        )
        marker = " <- AI" if name == "AI Optimizer" else ""
        print(
            f"{name:<20} {final_loss:>10.4f} {final_acc:>10.4f} "
            f"{early_norm:>12.3e}{marker}"
        )
    print("=" * 65)


def print_metrics_report(results: dict) -> None:
    ai_key = next((name for name in results if name.startswith("AI")), "AI Optimizer")
    ai_log = results.get(ai_key, {}).get("layer_log", [])
    report = full_report(results, ai_log)

    print("\nCOMPLETE METRICS REPORT")
    print("=" * 140)
    print(
        f"{'Optimizer':<22} {'Loss':>8} {'Acc':>8} {'EarlyGrad':>12} "
        f"{'VanishRatio':>14} {'Log10VR':>8} {'GainAdam':>9} {'GainSGD':>8} "
        f"{'ActRate':>8} {'RecRate':>8} {'Starved':>8} {'Recovered':>10}"
    )
    print("-" * 140)
    for name, m in report.items():
        print(
            f"{name:<22} "
            f"{m['final_loss']:>8.4f} "
            f"{m['final_accuracy']:>8.4f} "
            f"{m['early_grad_mean']:>12.3e} "
            f"{format_float(m['vanishing_ratio'], 2):>14} "
            f"{format_float(m['log10_vanishing_ratio'], 2):>8} "
            f"{format_float(m['early_gain_vs_adam'], 2):>9} "
            f"{format_float(m['early_gain_vs_sgd'], 2):>8} "
            f"{format_float(m['starvation_activation_rate'], 3) if name.startswith('AI') else 'N/A':>8} "
            f"{format_float(m['recovery_success_rate'], 3) if name.startswith('AI') else 'N/A':>8} "
            f"{str(m['starvation_events']):>8} "
            f"{str(m['starvation_recoveries']):>10}"
        )
    print("=" * 140)


def save_metrics_report(results: dict, output_dir: str = "results") -> None:
    ai_key = next((name for name in results if name.startswith("AI")), "AI Optimizer")
    ai_log = results.get(ai_key, {}).get("layer_log", [])
    report = full_report(results, ai_log)
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    serializable = {
        name: {
            key: (float(value) if isinstance(value, np.floating) else value)
            for key, value in metrics.items()
        }
        for name, metrics in report.items()
    }
    (out / "metrics_report.json").write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )

    headers = [
        "optimizer",
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
    lines = [",".join(headers)]
    for name, metrics in report.items():
        lines.append(",".join([name] + [str(metrics[h]) for h in headers[1:]]))
    (out / "metrics_report.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if ai_log:
        log_headers = [
            "step",
            "layer",
            "grad_norm",
            "relative_norm",
            "inter_layer_ratio",
            "confidence",
            "update_norm",
            "pre_clip_update_norm",
            "trust_clipped",
            "starved",
            "recovered",
            "trust",
            "effective_trust",
            "amp",
            "recovery_scale",
            "starved_steps",
        ]
        log_lines = [",".join(log_headers)]
        for row in ai_log:
            log_lines.append(",".join(str(row.get(h, "")) for h in log_headers))
        (out / "ai_layer_log.csv").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def main() -> dict:
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    num_layers = 12
    hidden_dim = 96
    num_steps = 300
    print(f"Device: {device}")
    print(f"Training {num_layers}-layer Sigmoid MLP on MNIST")
    print("This is still a vanishing-gradient task, but less saturated than 20 layers.\n")

    results = {}
    optimizers = [
        "Adam",
        "AdamW",
        "RMSProp",
        "SGD",
        "AI-SAFE-HYBRID",
        "AI-BALANCED-HYBRID",
        "AI-AGGRESSIVE-HYBRID",
        "AI-EXTREME-HYBRID",
    ]
    if checkpoint_is_compatible(MLP_CHECKPOINT):
        optimizers += [
            "AI-MLP-BALANCED-HYBRID",
            "AI-MLP-AGGRESSIVE-HYBRID",
        ]
    else:
        print(
            f"\nNo compatible temporal MLP controller checkpoint found at {MLP_CHECKPOINT}; "
            "run train_mlp_policy.py first to include MLP variants.\n"
        )
    if checkpoint_is_compatible(FINETUNED_MLP_CHECKPOINT):
        optimizers += [
            "AI-MLP-FT-BALANCED-HYBRID",
            "AI-MLP-FT-AGGRESSIVE-HYBRID",
        ]
    else:
        print(
            f"\nNo compatible fine-tuned MLP checkpoint found at {FINETUNED_MLP_CHECKPOINT}; "
            "run train_mlp_finetune.py first to include fine-tuned MLP variants.\n"
        )
    for name in optimizers:
        print(f"\n-- {name} --")
        results[name] = train_one_optimizer(
            name,
            num_steps=num_steps,
            device=device,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            seed=42,
        )

    print_results_table(results)
    print_metrics_report(results)
    save_metrics_report(results)
    print("\nSaved metrics to results/metrics_report.csv and results/metrics_report.json")
    return results


if __name__ == "__main__":
    main()
