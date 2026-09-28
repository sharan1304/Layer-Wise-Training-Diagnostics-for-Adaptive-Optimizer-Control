"""Shared MLP control policy for Phase 2 (see AI_Optimizer_Algorithm_Documentation.md, section 8).

Drop-in replacement for RuleBasedController: same ControlSignals output, but
the mapping from gradient-state features to control values is learned by a
single MLP shared across all layers instead of hand-coded formulas.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from config import DEFAULT_CONFIG, OptimizerConfig
from control_policy import ControlSignals
from grad_analyzer import GradientFeatures
from layer_memory import LayerState

BASE_FEATURE_NAMES = [
    "relative_grad_norm",
    "inter_layer_ratio",
    "cosine_similarity",
    "gradient_variance",
    "gradient_entropy",
    "moving_average_grad",
    "noise_level",
    "oscillation_score",
    "loss_plateau_flag",
    "stable_flag",
    "depth_normalized",
    "activation_variance",
    "starved_flag",
    "success_ema",
    "layer_lr",
    "epoch_progress",
]
FEATURE_NAMES = [
    f"t-{offset}:{name}"
    for offset in range(DEFAULT_CONFIG.mlp_history_window - 1, -1, -1)
    for name in BASE_FEATURE_NAMES
]
BASE_FEATURE_DIM = len(BASE_FEATURE_NAMES)
FEATURE_DIM = BASE_FEATURE_DIM * DEFAULT_CONFIG.mlp_history_window
OUTPUT_DIM = 8  # z_alpha, z_conf, z_amp, z_clip_radius, z_w1..z_w4
EPS = 1e-8


def _normalized_log(value: float) -> float:
    return float(max(min(math.log(value + EPS), 5.0), -20.0) / 20.0)


def _gradient_entropy(values: list[float]) -> float:
    if not values:
        return 0.0
    weights = torch.tensor([max(float(v), 0.0) for v in values], dtype=torch.float32)
    total = float(weights.sum().item())
    if total <= EPS or len(values) <= 1:
        return 0.0
    probs = weights / total
    entropy = -torch.sum(probs * torch.log(probs + EPS))
    return float((entropy / math.log(len(values))).item())


def build_current_feature_vector(
    features: GradientFeatures,
    state: LayerState,
    layer_idx: int,
    total_layers: int,
    global_step: int,
    max_training_steps: int,
    loss_plateau: bool,
    alpha_base: float = 1e-3,
) -> list[float]:
    grad_history = list(state.grad_norms)
    grad_variance = float(torch.tensor(grad_history, dtype=torch.float32).var(unbiased=False).item()) if grad_history else 0.0
    moving_average_grad = float(sum(grad_history) / len(grad_history)) if grad_history else features.grad_norm
    depth_l = layer_idx / max(total_layers - 1, 1)
    epoch_progress = min(global_step / max(max_training_steps, 1), 1.0)
    return [
        float(features.relative_norm),
        float(features.inter_layer_ratio),
        float(features.consistency),
        _normalized_log(grad_variance),
        _gradient_entropy(grad_history),
        _normalized_log(moving_average_grad),
        float(features.noise),
        float(features.oscillation),
        float(loss_plateau),
        float(features.stable),
        float(depth_l),
        0.0,  # activation variance is reserved for future forward-hook instrumentation
        float(features.starved),
        float(state.success_ema),
        _normalized_log(alpha_base),
        float(epoch_progress),
    ]


def build_feature_vector(
    features: GradientFeatures,
    state: LayerState,
    layer_idx: int,
    total_layers: int,
    global_step: int,
    max_training_steps: int,
    loss_plateau: bool,
    alpha_base: float = 1e-3,
    history_window: int | None = None,
) -> torch.Tensor:
    history_window = history_window or DEFAULT_CONFIG.mlp_history_window
    current = build_current_feature_vector(
        features,
        state,
        layer_idx,
        total_layers,
        global_step,
        max_training_steps,
        loss_plateau,
        alpha_base,
    )
    previous = list(state.feature_history)[-(history_window - 1):]
    frames = previous + [current]
    while len(frames) < history_window:
        frames.insert(0, current)
    flat = [value for frame in frames[-history_window:] for value in frame]
    return torch.tensor(flat, dtype=torch.float32)

def checkpoint_is_compatible(path: str, config: OptimizerConfig = DEFAULT_CONFIG) -> bool:
    checkpoint = Path(path)
    if not checkpoint.exists():
        return False
    try:
        state_dict = torch.load(checkpoint, map_location="cpu")
    except Exception:
        return False
    first_weight = state_dict.get("net.0.weight")
    last_weight = state_dict.get("net.4.weight")
    expected_feature_dim = BASE_FEATURE_DIM * config.mlp_history_window
    return (
        first_weight is not None
        and last_weight is not None
        and tuple(first_weight.shape) == (config.mlp_hidden_dim, expected_feature_dim)
        and tuple(last_weight.shape) == (OUTPUT_DIM, config.mlp_hidden_dim)
    )


class ControlMLP(nn.Module):
    """Single MLP shared across all layers: x_l,t -> z_l,t."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPController:
    """Learned analogue of RuleBasedController (same compute_all interface)."""

    def __init__(
        self,
        config: OptimizerConfig = DEFAULT_CONFIG,
        checkpoint: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = device
        self.net = ControlMLP(hidden_dim=config.mlp_hidden_dim).to(device)
        if checkpoint is not None:
            self.load(checkpoint)
        self.net.eval()

    def load(self, path: str) -> None:
        state_dict = torch.load(Path(path), map_location=self.device)
        self.net.load_state_dict(state_dict)

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), Path(path))

    def raw_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """z_l,t for a batch of feature vectors, shape (..., OUTPUT_DIM)."""
        return self.net(x)

    def decode(self, z: torch.Tensor, alpha_base: float) -> ControlSignals:
        cfg = self.config
        alpha_scale = cfg.alpha_max_scale * torch.sigmoid(z[0])
        confidence = torch.sigmoid(z[1])
        amplification = 1.0 + (cfg.max_amp - 1.0) * torch.sigmoid(z[2])
        clip_radius = cfg.tau_min + (cfg.tau_max - cfg.tau_min) * torch.sigmoid(z[3])
        weights = torch.softmax(z[4:8], dim=-1)
        return ControlSignals(
            alpha=float(alpha_base * alpha_scale.item()),
            confidence=float(confidence.item()),
            amplification=float(amplification.item()),
            trust_radius=float(clip_radius.item()),
            fusion_weights=[float(w) for w in weights.tolist()],
        )

    @torch.no_grad()
    def compute_all(
        self,
        features: GradientFeatures,
        state: LayerState,
        layer_idx: int,
        total_layers: int,
        alpha_base: float = 1e-3,
        *,
        loss_plateau: bool = False,
        global_step: int = 0,
    ) -> ControlSignals:
        x = build_feature_vector(
            features,
            state,
            layer_idx,
            total_layers,
            global_step,
            self.config.max_training_steps,
            loss_plateau,
            alpha_base,
            self.config.mlp_history_window,
        ).to(self.device)
        z = self.net(x)
        return self.decode(z, alpha_base)
