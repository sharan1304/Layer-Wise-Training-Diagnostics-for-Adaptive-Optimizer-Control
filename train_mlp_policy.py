"""Distill the RuleBasedController into the shared MLPController (Phase 2).

Runs AIOptimizer in rule-based mode on the same MNIST vanishing-gradient task
used by train_mnist.py, recording (feature_vector -> rule-based control
signals) pairs at every layer/step where the rule-based policy fires. The
ControlMLP is then trained by supervised regression to reproduce those
signals, so it becomes a learned, drop-in replacement for the formulas.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ai_optimizer import AIOptimizer
from config import DEFAULT_CONFIG
from mlp_policy import ControlMLP, build_feature_vector
from train_mnist import DeepSigmoidMLP


def collect_distillation_data(
    num_steps: int = 1000,
    batch_size: int = 64,
    num_layers: int = 12,
    hidden_dim: int = 96,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = DeepSigmoidMLP(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    opt = AIOptimizer(model.parameters(), lr=1e-3, mode="balanced", detector="hybrid")

    samples_x: list[torch.Tensor] = []
    samples_y: list[torch.Tensor] = []

    def record(**kwargs) -> None:
        x = build_feature_vector(
            kwargs["features"],
            kwargs["state"],
            kwargs["layer_idx"],
            kwargs["total_layers"],
            kwargs["global_step"],
            DEFAULT_CONFIG.max_training_steps,
            kwargs["loss_plateau"],
        )
        signals = kwargs["signals"]
        alpha_scale = signals.alpha / kwargs["alpha_base"]
        y = torch.tensor(
            [alpha_scale, signals.confidence, signals.amplification, kwargs["effective_trust_radius"]]
            + list(signals.fusion_weights),
            dtype=torch.float32,
        )
        samples_x.append(x)
        samples_y.append(y)

    opt.record_callback = record

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    data_iter = iter(loader)

    model.train()
    for _ in range(num_steps):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)

        opt.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        opt.step(loss=loss.item())

    return torch.stack(samples_x), torch.stack(samples_y)


def decode_targets(net: ControlMLP, x: torch.Tensor) -> torch.Tensor:
    """Apply the same sigmoid/softmax transforms as MLPController.decode, batched."""
    cfg = DEFAULT_CONFIG
    z = net(x)
    alpha_scale = cfg.alpha_max_scale * torch.sigmoid(z[:, 0])
    confidence = torch.sigmoid(z[:, 1])
    amplification = 1.0 + (cfg.max_amp - 1.0) * torch.sigmoid(z[:, 2])
    tau = cfg.tau_min + (cfg.tau_max - cfg.tau_min) * torch.sigmoid(z[:, 3])
    weights = torch.softmax(z[:, 4:8], dim=-1)
    return torch.cat(
        [alpha_scale[:, None], confidence[:, None], amplification[:, None], tau[:, None], weights],
        dim=-1,
    )


def train_mlp(
    x: torch.Tensor,
    y: torch.Tensor,
    hidden_dim: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
) -> ControlMLP:
    net = ControlMLP(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = decode_targets(net, x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch + 1:4d} | distillation MSE: {loss.item():.6f}")

    return net


def main() -> None:
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print("Collecting rule-based control signals from MNIST training...")
    x, y = collect_distillation_data(device=device)
    print(f"Collected {x.shape[0]} samples ({x.shape[1]} features -> {y.shape[1]} targets)")

    net = train_mlp(x, y)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    checkpoint_path = out_dir / "mlp_policy.pt"
    torch.save(net.state_dict(), checkpoint_path)
    print(f"Saved distilled MLP controller to {checkpoint_path}")


if __name__ == "__main__":
    main()
