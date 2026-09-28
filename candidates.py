"""Candidate update generation for the Phase 1 AI optimizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from control_policy import ControlSignals
from layer_memory import LayerState


@dataclass(frozen=True)
class CandidateUpdates:
    adamw: torch.Tensor
    rmsprop: torch.Tensor
    sgd: torch.Tensor
    normalized: torch.Tensor


class CandidateGenerator:
    EPS = 1e-8

    def adamw_update(
        self,
        g: torch.Tensor,
        param: torch.Tensor,
        state: LayerState,
        alpha: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> torch.Tensor:
        g_flat = g.reshape(-1).float()
        p_flat = param.detach().reshape(-1).float()
        t = max(state.step_count + 1, 1)

        if state.m_adamw is None or state.m_adamw.shape != g_flat.shape:
            state.m_adamw = torch.zeros_like(g_flat)
            state.v_adamw = torch.zeros_like(g_flat)

        state.m_adamw = beta1 * state.m_adamw + (1.0 - beta1) * g_flat
        state.v_adamw = beta2 * state.v_adamw + (1.0 - beta2) * (g_flat * g_flat)

        m_hat = state.m_adamw / (1.0 - beta1**t)
        v_hat = state.v_adamw / (1.0 - beta2**t)
        delta = -alpha * (m_hat / (v_hat.sqrt() + eps) + weight_decay * p_flat)
        return delta.reshape_as(g).to(dtype=param.dtype, device=param.device)

    def rmsprop_update(
        self,
        g: torch.Tensor,
        state: LayerState,
        alpha: float,
        rho: float = 0.99,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        g_flat = g.reshape(-1).float()
        if state.v_rmsprop is None or state.v_rmsprop.shape != g_flat.shape:
            state.v_rmsprop = torch.zeros_like(g_flat)
        state.v_rmsprop = rho * state.v_rmsprop + (1.0 - rho) * (g_flat * g_flat)
        delta = -alpha * g_flat / (state.v_rmsprop.sqrt() + eps)
        return delta.reshape_as(g).to(dtype=g.dtype, device=g.device)

    def sgd_momentum_update(
        self,
        g: torch.Tensor,
        state: LayerState,
        alpha: float,
        momentum: float = 0.9,
    ) -> torch.Tensor:
        g_flat = g.reshape(-1).float()
        if state.m_sgd is None or state.m_sgd.shape != g_flat.shape:
            state.m_sgd = torch.zeros_like(g_flat)
        state.m_sgd = momentum * state.m_sgd + g_flat
        delta = -alpha * state.m_sgd
        return delta.reshape_as(g).to(dtype=g.dtype, device=g.device)

    def normalized_momentum_recovery_update(
        self,
        g: torch.Tensor,
        alpha: float,
        confidence: float,
        amplification: float,
        recovery_scale: float = 1.0,
        direction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalized recovery candidate.

        Uses the SGD momentum buffer when available, otherwise the current
        gradient. This is intentionally a normalized momentum-borrowed recovery
        direction, not just normalize(g).
        """
        source = direction if direction is not None else g
        source_flat = source.reshape(-1).float()
        unit_g = source_flat / (source_flat.norm() + self.EPS)
        delta = -alpha * unit_g * confidence * amplification * recovery_scale
        return delta.reshape_as(g).to(dtype=g.dtype, device=g.device)

    def generate_all(
        self,
        g: torch.Tensor,
        param: torch.Tensor,
        state: LayerState,
        signals: ControlSignals,
        weight_decay: float = 0.01,
    ) -> CandidateUpdates:
        return CandidateUpdates(
            adamw=self.adamw_update(
                g,
                param,
                state,
                signals.alpha,
                weight_decay=weight_decay,
            ),
            rmsprop=self.rmsprop_update(g, state, signals.alpha),
            sgd=self.sgd_momentum_update(g, state, signals.alpha),
            normalized=self.normalized_momentum_recovery_update(
                g,
                signals.alpha,
                signals.confidence,
                signals.amplification,
                recovery_scale=state.recovery_scale,
                direction=state.m_sgd,
            ),
        )
