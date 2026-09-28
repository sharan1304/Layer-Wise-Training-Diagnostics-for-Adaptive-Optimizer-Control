"""Experiment configuration for Phase 1.1 optimizer modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OptimizerMode = Literal["safe", "balanced", "aggressive", "extreme"]
DetectorMode = Literal["strict", "early", "hybrid"]


@dataclass(frozen=True)
class OptimizerConfig:
    mode: OptimizerMode = "balanced"
    detector: DetectorMode = "hybrid"
    warmup_steps: int = 35
    blend_steps: int = 35
    relative_threshold_strict: float = 0.10
    ratio_threshold_strict: float = 0.20
    success_threshold_strict: float = 0.40
    relative_threshold_early: float = 0.30
    ratio_threshold_early: float = 0.40
    success_threshold_early: float = 0.85
    loss_plateau_window: int = 35
    min_loss_improvement: float = 0.003
    max_amp: float = 5.0
    depth_boost: float = 0.75
    starvation_lr_multiplier: float = 3.0
    cold_start_amp: float = 1.75
    stability_noise_factor: float = 0.50
    stability_oscillation_factor: float = 0.50
    recovery_grow: float = 1.20
    recovery_shrink: float = 0.70
    recovery_min: float = 1.0
    recovery_max: float = 10.0
    depth_trust_boost: float = 0.5
    starvation_patience: int = 3
    alpha_max_scale: float = 5.25
    tau_min: float = 0.005
    tau_max: float = 0.05
    max_training_steps: int = 2000
    mlp_hidden_dim: int = 64
    mlp_history_window: int = 5
    mlp_blend_base: float = 0.7
    allow_untrained_mlp: bool = False

    @property
    def starved_fusion_weights(self) -> list[float]:
        if self.mode == "safe":
            return [0.20, 0.20, 0.10, 0.50]
        if self.mode == "balanced":
            return [0.10, 0.10, 0.10, 0.70]
        if self.mode == "aggressive":
            return [0.05, 0.05, 0.05, 0.85]
        if self.mode == "extreme":
            return [0.00, 0.00, 0.00, 1.00]
        raise ValueError(f"Unknown optimizer mode: {self.mode}")


DEFAULT_CONFIG = OptimizerConfig()
