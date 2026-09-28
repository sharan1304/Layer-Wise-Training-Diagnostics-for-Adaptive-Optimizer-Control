"""CIFAR-10 comparison for AIOptimizer vs classical and research-inspired baselines.

NOVAKInspired and LNGDLite are local proxy baselines, not official paper code.
Use this script to test whether our layer-aware optimizer still helps outside
MNIST on a small convolutional CIFAR-10 model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ai_optimizer import AIOptimizer
from metrics import full_report
from research_baselines import LNGDLite, NOVAKInspired


class SmallCIFARConvNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def get_layer_grad_norms(self) -> list[float]:
        norms: list[float] = []
        for module in list(self.features) + list(self.classifier):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if module.weight.grad is None:
                    norms.append(0.0)
                else:
                    norms.append(float(module.weight.grad.norm().item()))
        return norms


def make_optimizer(name: str, params):
    if name == "Adam":
        return torch.optim.Adam(params, lr=1e-3)
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    if name == "RMSProp":
        return torch.optim.RMSprop(params, lr=1e-3)
    if name == "SGD":
        return torch.optim.SGD(params, lr=0.03, momentum=0.9, weight_decay=5e-4)
    if name == "NOVAK-inspired":
        return NOVAKInspired(params, lr=1e-3, weight_decay=0.01)
    if name == "LNGD-lite":
        return LNGDLite(params, lr=3e-3, weight_decay=5e-4)
    if name.startswith("AI-"):
        parts = name.split("-")
        mode = parts[1].lower()
        detector = parts[2].lower() if len(parts) > 2 else "hybrid"
        return AIOptimizer(params, lr=1e-3, mode=mode, detector=detector, weight_decay=5e-4)
    raise ValueError(f"Unknown optimizer: {name}")


def build_loaders(batch_size: int, seed: int, train_subset: int | None, test_subset: int | None):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    rng = np.random.default_rng(seed)
    if train_subset is not None and train_subset < len(train_ds):
        train_ds = Subset(train_ds, rng.choice(len(train_ds), size=train_subset, replace=False).tolist())
    if test_subset is not None and test_subset < len(test_ds):
        test_ds = Subset(test_ds, rng.choice(len(test_ds), size=test_subset, replace=False).tolist())
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, max_batches: int | None = None) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(1) == labels).sum().item())
        total += int(labels.numel())
    model.train()
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_one(
    optimizer_name: str,
    steps: int,
    batch_size: int,
    seed: int,
    device: str,
    train_subset: int | None,
    test_subset: int | None,
    log_interval: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallCIFARConvNet().to(device)
    optimizer = make_optimizer(optimizer_name, model.parameters())
    train_loader, test_loader = build_loaders(batch_size, seed, train_subset, test_subset)
    data_iter = iter(train_loader)
    loss_history: list[float] = []
    accuracy_history: list[float] = []
    grad_norm_history: list[list[float]] = []

    model.train()
    for step in range(steps):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        if isinstance(optimizer, AIOptimizer):
            optimizer.step(loss=loss.item())
        else:
            optimizer.step()

        batch_acc = float((logits.argmax(1) == labels).float().mean().item())
        loss_history.append(float(loss.item()))
        accuracy_history.append(batch_acc)
        grad_norm_history.append(model.get_layer_grad_norms())

        if log_interval > 0 and (step + 1) % log_interval == 0:
            test_loss, test_acc = evaluate(model, test_loader, device, max_batches=20)
            print(
                f"{optimizer_name:<18} step {step + 1:4d} | "
                f"train loss {loss.item():.4f} | batch acc {batch_acc:.3f} | "
                f"test acc {test_acc:.3f}"
            )

    test_loss, test_acc = evaluate(model, test_loader, device)
    loss_history[-1] = test_loss
    accuracy_history[-1] = test_acc
    result: dict[str, Any] = {
        "loss": loss_history,
        "accuracy": accuracy_history,
        "grad_norms": grad_norm_history,
        "seed": seed,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }
    if isinstance(optimizer, AIOptimizer):
        result["layer_log"] = optimizer.layer_log
    return result


def save_report(report: dict[str, dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cifar10_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    headers = ["optimizer", *next(iter(report.values())).keys()]
    with (output_dir / "cifar10_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for name, metrics in report.items():
            writer.writerow([name, *[metrics[h] for h in headers[1:]]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--test-subset", type=int, default=2000)
    parser.add_argument("--output-dir", default="results/cifar10_comparison")
    parser.add_argument("--optimizers", nargs="+", default=[
        "Adam", "AdamW", "RMSProp", "SGD", "NOVAK-inspired", "LNGD-lite",
        "AI-BALANCED-HYBRID", "AI-AGGRESSIVE-HYBRID",
    ])
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("CIFAR-10 comparison: NOVAK-inspired and LNGD-lite are proxy baselines, not official implementations.")

    results = {}
    for name in args.optimizers:
        print(f"\n-- {name} --")
        results[name] = train_one(
            name,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            train_subset=args.train_subset,
            test_subset=args.test_subset,
            log_interval=max(args.steps // 5, 1),
        )

    report = full_report(results)
    print("\nCIFAR-10 REPORT")
    print("=" * 110)
    print(f"{'Optimizer':<20} {'TestLoss':>9} {'TestAcc':>8} {'Log10VR':>10} {'EarlyGrad':>12} {'TrustClips':>10}")
    print("-" * 110)
    for name, m in report.items():
        print(
            f"{name:<20} {m['final_loss']:>9.4f} {m['final_accuracy']:>8.3f} "
            f"{m['log10_vanishing_ratio']:>10.3f} {m['early_grad_mean']:>12.3e} {str(m['trust_activations']):>10}"
        )
    print("=" * 110)
    save_report(report, Path(args.output_dir))
    print(f"Saved CIFAR-10 report to {args.output_dir}")


if __name__ == "__main__":
    main()
