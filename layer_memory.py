"""Layer state memory for the Phase 1 AI optimizer.

The memory stores compact per-parameter-group statistics used by the
rule-based controller. It intentionally keeps only one previous gradient tensor
for cosine similarity and optimizer moments; longer histories are summary
statistics such as norms and cosine values.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np
import torch


@dataclass
class LayerState:
    """State tracked for one trainable tensor or grouped layer."""

    grad_norms: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    update_norms: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    cosine_sims: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    loss_responses: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    feature_history: Deque[list[float]] = field(default_factory=lambda: deque(maxlen=5))

    success_ema: float = 1.0
    trust_radius: float = 0.03
    amp_state: float = 1.0
    recovery_scale: float = 1.0
    starved_steps: int = 0
    recovery_attempts: int = 0
    recovery_successes: int = 0
    grad_median: float = 1.0
    prev_grad: Optional[torch.Tensor] = None
    prev_update_norm: float = 0.0
    step_count: int = 0

    m_adamw: Optional[torch.Tensor] = None
    v_adamw: Optional[torch.Tensor] = None
    v_rmsprop: Optional[torch.Tensor] = None
    m_sgd: Optional[torch.Tensor] = None

    emergency_mode: bool = False


class LayerStateMemory:
    """Owns and updates all layer states."""

    TRUST_GROW = 1.08
    TRUST_SHRINK = 0.90
    TRUST_MIN = 0.005
    TRUST_MAX = 0.05
    AMP_GROW = 1.25
    AMP_SHRINK = 0.8
    AMP_MAX = 5.0
    SUCCESS_DECAY = 0.9

    def __init__(self) -> None:
        self.states: Dict[int, LayerState] = {}
        self.global_step = 0

    def get_or_create(self, layer_id: int) -> LayerState:
        if layer_id not in self.states:
            self.states[layer_id] = LayerState()
        return self.states[layer_id]

    def update_after_step(
        self,
        layer_id: int,
        grad_norm: float,
        update_norm: float,
        cosine_sim: float,
        success_flag: float,
    ) -> None:
        state = self.get_or_create(layer_id)
        state.grad_norms.append(float(grad_norm))
        state.update_norms.append(float(update_norm))
        state.cosine_sims.append(float(cosine_sim))
        state.loss_responses.append(float(success_flag))
        state.prev_update_norm = float(update_norm)
        state.step_count += 1
        self.update_success_ema(layer_id, success_flag)
        self.update_grad_median(layer_id, grad_norm)

    def update_success_ema(
        self,
        layer_id: int,
        success: float,
        decay: float = SUCCESS_DECAY,
    ) -> None:
        state = self.get_or_create(layer_id)
        state.success_ema = decay * state.success_ema + (1.0 - decay) * float(success)

    def update_trust_radius(self, layer_id: int, improved: bool) -> None:
        state = self.get_or_create(layer_id)
        if improved:
            state.trust_radius = min(state.trust_radius * self.TRUST_GROW, self.TRUST_MAX)
        else:
            state.trust_radius = max(
                state.trust_radius * self.TRUST_SHRINK,
                self.TRUST_MIN,
            )

    def update_amp_state(self, layer_id: int, starved: bool, improved: bool) -> None:
        state = self.get_or_create(layer_id)
        if not starved:
            state.amp_state = 1.0
            return
        if improved:
            state.amp_state = min(state.amp_state * self.AMP_GROW, self.AMP_MAX)
        else:
            # Do not immediately fall back to no correction after one noisy
            # minibatch. The controller still caps the effective amplification.
            state.amp_state = max(state.amp_state * self.AMP_SHRINK, 1.25)

    def update_recovery_state(
        self,
        layer_id: int,
        starved: bool,
        improved: bool,
        recovered: bool,
        grow: float = 1.20,
        shrink: float = 0.70,
        min_scale: float = 1.0,
        max_scale: float = 10.0,
    ) -> None:
        state = self.get_or_create(layer_id)
        was_starved = state.starved_steps > 0
        if starved:
            state.starved_steps += 1
            state.recovery_attempts += 1
            if improved:
                state.recovery_scale = min(state.recovery_scale * grow, max_scale)
            else:
                state.recovery_scale = min(max(state.recovery_scale * 1.05, min_scale), max_scale)
            return

        if recovered and was_starved:
            state.recovery_successes += 1
            state.recovery_scale = max(state.recovery_scale * shrink, min_scale)
        elif not improved:
            state.recovery_scale = max(state.recovery_scale * shrink, min_scale)
        state.starved_steps = 0

    def append_feature_vector(self, layer_id: int, vector: list[float], maxlen: int = 5) -> None:
        state = self.get_or_create(layer_id)
        if state.feature_history.maxlen != maxlen:
            state.feature_history = deque(state.feature_history, maxlen=maxlen)
        state.feature_history.append([float(v) for v in vector])

    def update_grad_median(self, layer_id: int, norm: float) -> None:
        state = self.get_or_create(layer_id)
        if len(state.grad_norms) >= 3:
            state.grad_median = float(np.median(list(state.grad_norms)))
        else:
            state.grad_median = max(float(norm), 1e-8)

    def get_loss_plateau(
        self,
        loss_history: Deque[float],
        window: int = 35,
        min_improvement: float = 0.003,
    ) -> bool:
        if len(loss_history) < window:
            return False
        recent = list(loss_history)[-window:]
        return recent[0] - recent[-1] < min_improvement

    def emergency_shrink(self, layer_id: int) -> None:
        state = self.get_or_create(layer_id)
        state.trust_radius = max(state.trust_radius * 0.1, self.TRUST_MIN)
        state.amp_state = 1.0
        state.recovery_scale = 1.0
        state.starved_steps = 0
        state.emergency_mode = True

    def reset_emergency(self, layer_id: int) -> None:
        self.get_or_create(layer_id).emergency_mode = False
