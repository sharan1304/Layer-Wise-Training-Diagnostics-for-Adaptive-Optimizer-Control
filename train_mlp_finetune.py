"""Fine-tune the distilled MLP control policy with a differentiable one-step task loss.

Phase 2B-v0:
    L_total = CE(model(theta + delta_mlp), batch)
              + lambda_anchor * MSE(action_mlp, action_rule)

This script intentionally does not call AIOptimizer.step() for the MLP update.
AIOptimizer.step() is a normal no_grad optimizer path; here we need gradients
through a temporary parameter update back into the MLP controller. Temporary
parameters are evaluated with torch.func.functional_call so the graph stays
connected to the MLP outputs.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ai_optimizer import AIOptimizer
from config import DEFAULT_CONFIG, OptimizerConfig
from control_policy import RuleBasedController
from grad_analyzer import GradientStateAnalyzer
from layer_memory import LayerStateMemory
from mlp_policy import ControlMLP, build_current_feature_vector, build_feature_vector, checkpoint_is_compatible
from train_mnist import DeepSigmoidMLP

DISTILLED_CHECKPOINT = Path("results/mlp_policy.pt")
FINETUNED_CHECKPOINT = Path("results/mlp_policy_finetuned.pt")
EPS = 1e-8


def decode_action_tensor(z: torch.Tensor, cfg: OptimizerConfig, alpha_base: float) -> dict[str, torch.Tensor]:
    """Differentiable decode matching MLPController.decode."""
    alpha_scale = cfg.alpha_max_scale * torch.sigmoid(z[0])
    confidence = torch.sigmoid(z[1])
    amplification = 1.0 + (cfg.max_amp - 1.0) * torch.sigmoid(z[2])
    clip_radius = cfg.tau_min + (cfg.tau_max - cfg.tau_min) * torch.sigmoid(z[3])
    weights = torch.softmax(z[4:8], dim=-1)
    return {
        "alpha": alpha_base * alpha_scale,
        "alpha_scale": alpha_scale,
        "confidence": confidence,
        "amplification": amplification,
        "clip_radius": clip_radius,
        "weights": weights,
        "flat": torch.cat(
            [
                alpha_scale.reshape(1),
                confidence.reshape(1),
                amplification.reshape(1),
                clip_radius.reshape(1),
                weights.reshape(-1),
            ]
        ),
    }


def rule_target_tensor(rule_signals: Any, alpha_base: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            rule_signals.alpha / alpha_base,
            rule_signals.confidence,
            rule_signals.amplification,
            rule_signals.trust_radius,
            *rule_signals.fusion_weights,
        ],
        dtype=torch.float32,
        device=device,
    )


def _moment_like(buffer: torch.Tensor | None, g_flat: torch.Tensor) -> torch.Tensor:
    if buffer is None or buffer.shape != g_flat.shape:
        return torch.zeros_like(g_flat)
    return buffer.detach().to(device=g_flat.device, dtype=g_flat.dtype)


def differentiable_candidates(
    g: torch.Tensor,
    param: torch.Tensor,
    state: Any,
    action: dict[str, torch.Tensor],
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate candidate deltas without mutating LayerState.

    State buffers are treated as constants; g and action tensors remain in the
    graph. Returned deltas follow Phase 1 sign convention: theta_temp = theta + delta.
    """
    alpha = action["alpha"]
    confidence = action["confidence"]
    amplification = action["amplification"]
    g_flat = g.reshape(-1).float()
    p_flat = param.reshape(-1).float()

    beta1, beta2 = 0.9, 0.999
    t = max(state.step_count + 1, 1)
    m_prev = _moment_like(state.m_adamw, g_flat)
    v_prev = _moment_like(state.v_adamw, g_flat)
    m_adamw = beta1 * m_prev + (1.0 - beta1) * g_flat
    v_adamw = beta2 * v_prev + (1.0 - beta2) * (g_flat * g_flat)
    m_hat = m_adamw / (1.0 - beta1**t)
    v_hat = v_adamw / (1.0 - beta2**t)
    adamw = -alpha * (m_hat / (v_hat.sqrt() + EPS) + weight_decay * p_flat)

    rho = 0.99
    v_rms_prev = _moment_like(state.v_rmsprop, g_flat)
    v_rms = rho * v_rms_prev + (1.0 - rho) * (g_flat * g_flat)
    rmsprop = -alpha * g_flat / (v_rms.sqrt() + EPS)

    momentum = 0.9
    m_sgd_prev = _moment_like(state.m_sgd, g_flat)
    m_sgd = momentum * m_sgd_prev + g_flat
    sgd = -alpha * m_sgd

    direction = m_sgd if m_sgd.norm() > EPS else g_flat
    unit = direction / (direction.norm() + EPS)
    normalized = -alpha * unit * confidence * amplification * float(state.recovery_scale)

    return (
        adamw.reshape_as(param).to(dtype=param.dtype),
        rmsprop.reshape_as(param).to(dtype=param.dtype),
        sgd.reshape_as(param).to(dtype=param.dtype),
        normalized.reshape_as(param).to(dtype=param.dtype),
    )


def clip_delta(delta: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    norm = delta.norm()
    scale = torch.minimum(torch.ones_like(norm), radius / (norm + EPS))
    return delta * scale


def functional_logits(model: torch.nn.Module, updates: dict[str, torch.Tensor], images: torch.Tensor) -> torch.Tensor:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    temp_params = {name: param + updates.get(name, torch.zeros_like(param)) for name, param in params.items()}
    return functional_call(model, (temp_params, buffers), (images,))


def fine_tune_step(
    model: DeepSigmoidMLP,
    mlp: ControlMLP,
    rule_policy: RuleBasedController,
    analyzer: GradientStateAnalyzer,
    memory: LayerStateMemory,
    images: torch.Tensor,
    labels: torch.Tensor,
    global_step: int,
    cfg: OptimizerConfig,
    alpha_base: float,
    lambda_anchor: float,
    weight_decay: float,
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    logits_before = model(images)
    loss_before = F.cross_entropy(logits_before, labels)

    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    params = [param for _, param in named_params]
    grads = torch.autograd.grad(loss_before, params, create_graph=True, retain_graph=True)

    loss_history = deque([float(loss_before.detach().item())], maxlen=100)
    loss_plateau = memory.get_loss_plateau(loss_history)
    weight_like_indices = [i for i, (_, p) in enumerate(named_params) if p.ndim >= 2]
    weight_like_count = max(len(weight_like_indices), 1)
    weight_grads = {i: grads[i] for i in weight_like_indices}

    updates: dict[str, torch.Tensor] = {}
    anchor_losses: list[torch.Tensor] = []
    action_count = 0

    for param_index, (name, param) in enumerate(named_params):
        g = grads[param_index]
        if param.ndim < 2:
            updates[name] = torch.zeros_like(param)
            continue

        state = memory.get_or_create(param_index)
        next_grad = None
        for next_idx in weight_like_indices:
            if next_idx > param_index:
                next_grad = weight_grads[next_idx]
                break

        with torch.no_grad():
            features = analyzer.compute_all(param_index, g.detach(), state, next_grad.detach() if next_grad is not None else None, loss_plateau)
            effective_idx = sum(1 for idx in weight_like_indices if idx <= param_index) - 1
            current_feature = build_current_feature_vector(
                features,
                state,
                effective_idx,
                weight_like_count,
                global_step,
                cfg.max_training_steps,
                loss_plateau,
                alpha_base,
            )
            rule_signals = rule_policy.compute_all(
                features,
                state,
                effective_idx,
                weight_like_count,
                alpha_base,
                loss_plateau=loss_plateau,
                global_step=global_step,
            )
            rule_target = rule_target_tensor(rule_signals, alpha_base, param.device)

        x = build_feature_vector(
            features,
            state,
            effective_idx,
            weight_like_count,
            global_step,
            cfg.max_training_steps,
            loss_plateau,
            alpha_base,
            cfg.mlp_history_window,
        ).to(param.device)
        action = decode_action_tensor(mlp(x), cfg, alpha_base)
        anchor_losses.append(F.mse_loss(action["flat"], rule_target))

        candidates = differentiable_candidates(g, param, state, action, weight_decay)
        weights = action["weights"]
        delta = sum(weights[i] * candidates[i] for i in range(4))
        updates[name] = clip_delta(delta, action["clip_radius"])
        memory.append_feature_vector(param_index, current_feature, maxlen=cfg.mlp_history_window)
        action_count += 1

    logits_after = functional_logits(model, updates, images)
    loss_after = F.cross_entropy(logits_after, labels)
    anchor = torch.stack(anchor_losses).mean() if anchor_losses else torch.zeros((), device=images.device)
    total = loss_after + lambda_anchor * anchor
    total.backward()

    return {
        "loss_before": float(loss_before.detach().item()),
        "loss_after": float(loss_after.detach().item()),
        "anchor": float(anchor.detach().item()),
        "total": float(total.detach().item()),
        "actions": float(action_count),
    }


def main() -> None:
    cfg = DEFAULT_CONFIG
    if not checkpoint_is_compatible(str(DISTILLED_CHECKPOINT), cfg):
        raise FileNotFoundError(
            f"No compatible distilled checkpoint at {DISTILLED_CHECKPOINT}. Run train_mlp_policy.py first."
        )

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)
    model = DeepSigmoidMLP(hidden_dim=64, num_layers=8).to(device)
    mlp = ControlMLP(hidden_dim=cfg.mlp_hidden_dim).to(device)
    mlp.load_state_dict(torch.load(DISTILLED_CHECKPOINT, map_location=device))
    mlp.train()

    # The model is updated by the stable rule optimizer to keep trajectories realistic;
    # only the temporary functional update is used to train the MLP controller.
    model_optimizer = AIOptimizer(model.parameters(), lr=1e-3, mode="balanced", detector="hybrid")
    mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=3e-4)
    rule_policy = RuleBasedController(cfg)
    analyzer = GradientStateAnalyzer(cfg)
    memory = LayerStateMemory()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    steps = 120
    lambda_anchor_start = 1.0
    lambda_anchor_end = 0.15
    for step, (images, labels) in enumerate(loader, start=1):
        if step > steps:
            break
        images, labels = images.to(device), labels.to(device)
        progress = (step - 1) / max(steps - 1, 1)
        lambda_anchor = lambda_anchor_start * (1.0 - progress) + lambda_anchor_end * progress

        mlp_optimizer.zero_grad(set_to_none=True)
        stats = fine_tune_step(
            model,
            mlp,
            rule_policy,
            analyzer,
            memory,
            images,
            labels,
            step,
            cfg,
            alpha_base=1e-3,
            lambda_anchor=lambda_anchor,
            weight_decay=0.01,
        )
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        mlp_optimizer.step()

        # Advance the actual model with the stable non-differentiable rule controller.
        model_optimizer.zero_grad(set_to_none=True)
        real_loss = F.cross_entropy(model(images), labels)
        real_loss.backward()
        model_optimizer.step(loss=real_loss.item())

        if step == 1 or step % 20 == 0:
            print(
                f"step {step:03d} | before {stats['loss_before']:.4f} | "
                f"after {stats['loss_after']:.4f} | anchor {stats['anchor']:.5f} | "
                f"lambda {lambda_anchor:.3f} | actions {int(stats['actions'])}"
            )

    FINETUNED_CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save(mlp.state_dict(), FINETUNED_CHECKPOINT)
    print(f"Saved fine-tuned MLP controller to {FINETUNED_CHECKPOINT}")


if __name__ == "__main__":
    main()
