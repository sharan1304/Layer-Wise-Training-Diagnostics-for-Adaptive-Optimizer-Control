"""Gradient state analyzer for the Phase 1 AI optimizer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from config import DEFAULT_CONFIG, OptimizerConfig
from layer_memory import LayerState


@dataclass(frozen=True)
class GradientFeatures:
    grad_norm: float
    relative_norm: float
    inter_layer_ratio: float
    consistency: float
    noise: float
    oscillation: float
    stability: float
    starved: bool
    exploding: bool
    noisy: bool
    oscillating: bool
    stable: bool


class GradientStateAnalyzer:
    EXPLOSION_THRESHOLD = 10.0
    NOISE_THRESHOLD = 0.3
    OSC_THRESHOLD = 0.3
    STABILITY_THRESHOLD = 0.7
    EPS = 1e-8

    def __init__(self, config: OptimizerConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def compute_grad_norm(self, g: torch.Tensor) -> float:
        return float(g.norm().item())

    def compute_relative_norm(self, grad_norm: float, grad_median: float) -> float:
        return float(grad_norm / (grad_median + self.EPS))

    def compute_inter_layer_ratio(
        self,
        g_l: torch.Tensor,
        g_next: Optional[torch.Tensor],
    ) -> float:
        if g_next is None:
            return 1.0
        return float(g_l.norm().item() / (g_next.norm().item() + self.EPS))

    def compute_consistency(
        self,
        g_t: torch.Tensor,
        g_prev: Optional[torch.Tensor],
    ) -> float:
        if g_prev is None:
            return 0.5
        g_flat = g_t.reshape(-1).float()
        prev_flat = g_prev.reshape(-1).float()
        if g_flat.numel() != prev_flat.numel():
            return 0.5
        norm_t = g_flat.norm()
        norm_prev = prev_flat.norm()
        if norm_t < self.EPS or norm_prev < self.EPS:
            return 0.5
        cos = torch.dot(g_flat, prev_flat) / (norm_t * norm_prev + self.EPS)
        cos_f = float(torch.clamp(cos, -1.0, 1.0).item())
        return (cos_f + 1.0) / 2.0

    def compute_noise(self, norm_history: deque[float]) -> float:
        if len(norm_history) < 3:
            return 0.0
        norms = np.asarray(list(norm_history), dtype=float)
        noise = float(np.std(norms) / (np.mean(norms) + self.EPS))
        return float(np.clip(noise, 0.0, 1.0))

    def compute_oscillation(self, cosine_history: deque[float]) -> float:
        if len(cosine_history) < 2:
            return 0.0
        sims = list(cosine_history)
        flips = sum(
            1
            for i in range(1, len(sims))
            if (sims[i] < 0.5) != (sims[i - 1] < 0.5)
        )
        return float(np.clip(flips / max(len(sims) - 1, 1), 0.0, 1.0))

    def compute_stability(self, noise: float, oscillation: float) -> float:
        return float(np.clip(1.0 - noise - oscillation, 0.0, 1.0))

    def detect_starvation(
        self,
        relative_norm: float,
        inter_layer_ratio: float,
        success_ema: float,
        loss_plateau: bool,
    ) -> bool:
        strict = (
            relative_norm < self.config.relative_threshold_strict
            and loss_plateau
            and success_ema < self.config.success_threshold_strict
            and inter_layer_ratio < self.config.ratio_threshold_strict
        )
        early = (
            relative_norm < self.config.relative_threshold_early
            and inter_layer_ratio < self.config.ratio_threshold_early
            and success_ema < self.config.success_threshold_early
        )
        if self.config.detector == "strict":
            return strict
        if self.config.detector == "early":
            return early
        if self.config.detector == "hybrid":
            return strict or early
        raise ValueError(f"Unknown detector mode: {self.config.detector}")

    def detect_explosion(self, g: torch.Tensor, threshold: float = EXPLOSION_THRESHOLD) -> bool:
        return float(g.norm().item()) > threshold

    def compute_all(
        self,
        layer_id: int,
        g: torch.Tensor,
        state: LayerState,
        next_grad: Optional[torch.Tensor],
        loss_plateau: bool,
    ) -> GradientFeatures:
        del layer_id
        grad_norm = self.compute_grad_norm(g)
        relative_norm = self.compute_relative_norm(grad_norm, state.grad_median)
        inter_ratio = self.compute_inter_layer_ratio(g, next_grad)
        consistency = self.compute_consistency(g, state.prev_grad)
        noise = self.compute_noise(state.grad_norms)
        oscillation = self.compute_oscillation(state.cosine_sims)
        stability = self.compute_stability(noise, oscillation)
        starved = self.detect_starvation(
            relative_norm,
            inter_ratio,
            state.success_ema,
            loss_plateau,
        )
        exploding = self.detect_explosion(g)
        return GradientFeatures(
            grad_norm=grad_norm,
            relative_norm=relative_norm,
            inter_layer_ratio=inter_ratio,
            consistency=consistency,
            noise=noise,
            oscillation=oscillation,
            stability=stability,
            starved=starved,
            exploding=exploding,
            noisy=noise > self.NOISE_THRESHOLD,
            oscillating=oscillation > self.OSC_THRESHOLD,
            stable=stability > self.STABILITY_THRESHOLD,
        )
