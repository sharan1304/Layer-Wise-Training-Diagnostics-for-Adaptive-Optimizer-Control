"""Rule-based control policy for Phase 1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from config import DEFAULT_CONFIG, OptimizerConfig
from grad_analyzer import GradientFeatures
from layer_memory import LayerState


@dataclass(frozen=True)
class ControlSignals:
    alpha: float
    confidence: float
    amplification: float
    trust_radius: float
    fusion_weights: List[float]


def blend_signals(rule: ControlSignals, mlp: ControlSignals, blend: float) -> ControlSignals:
    """Confidence-weighted mix of a rule-based and an MLP control signal.

    Used as a safety net so an under-confident or out-of-distribution MLP
    cannot dominate the update; blend should already be scaled by the MLP's
    own confidence (e.g. mlp_blend_base * mlp.confidence) before calling this.
    """
    blend = float(np.clip(blend, 0.0, 1.0))
    return ControlSignals(
        alpha=(1.0 - blend) * rule.alpha + blend * mlp.alpha,
        confidence=(1.0 - blend) * rule.confidence + blend * mlp.confidence,
        amplification=(1.0 - blend) * rule.amplification + blend * mlp.amplification,
        trust_radius=(1.0 - blend) * rule.trust_radius + blend * mlp.trust_radius,
        fusion_weights=[
            (1.0 - blend) * rw + blend * mw
            for rw, mw in zip(rule.fusion_weights, mlp.fusion_weights)
        ],
    )


class RuleBasedController:
    EPS = 1e-8

    def __init__(self, config: OptimizerConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def compute_alpha(
        self,
        features: GradientFeatures,
        layer_idx: int,
        total_layers: int,
        alpha_base: float = 1e-3,
    ) -> float:
        depth_l = layer_idx / max(total_layers - 1, 1)
        depth_factor = 1.0 + self.config.depth_boost * (1.0 - depth_l)
        starvation_factor = self.config.starvation_lr_multiplier if features.starved else 1.0
        stability_factor = float(
            np.clip(
                1.0
                - self.config.stability_noise_factor * features.noise
                - self.config.stability_oscillation_factor * features.oscillation,
                0.25,
                1.0,
            )
        )
        return float(alpha_base * depth_factor * starvation_factor * stability_factor)

    def compute_confidence(self, features: GradientFeatures, success_ema: float) -> float:
        raw = (
            0.25 * features.consistency
            + 0.25 * (1.0 - features.noise)
            + 0.25 * success_ema
            - 0.25 * features.oscillation
        )
        return float(1.0 / (1.0 + math.exp(-raw)))

    def compute_amplification(
        self,
        features: GradientFeatures,
        state: LayerState,
        max_amp: float | None = None,
    ) -> float:
        if max_amp is None:
            max_amp = self.config.max_amp
        if not features.starved:
            return 1.0
        amp_raw = 1.0 / (features.relative_norm + self.EPS)
        amp_candidate = float(np.clip(amp_raw, 1.0, max_amp))
        # Ensures the first confirmed starvation correction is not a no-op
        # when amp_state is still at its neutral initialization value.
        scheduled_amp = max(state.amp_state, self.config.cold_start_amp)
        return float(min(amp_candidate, scheduled_amp, max_amp))

    def compute_fusion_weights(self, features: GradientFeatures) -> List[float]:
        if features.starved:
            return self.config.starved_fusion_weights
        if features.noisy:
            return [0.45, 0.35, 0.15, 0.05]
        if features.oscillating:
            return [0.50, 0.30, 0.15, 0.05]
        if features.stable:
            return [0.25, 0.10, 0.60, 0.05]
        return [0.50, 0.25, 0.20, 0.05]

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
        del loss_plateau, global_step  # only used by the MLP controller
        return ControlSignals(
            alpha=self.compute_alpha(features, layer_idx, total_layers, alpha_base),
            confidence=self.compute_confidence(features, state.success_ema),
            amplification=self.compute_amplification(features, state),
            trust_radius=state.trust_radius,
            fusion_weights=self.compute_fusion_weights(features),
        )
