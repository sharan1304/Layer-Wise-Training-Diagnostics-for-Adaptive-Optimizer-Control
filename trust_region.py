"""Trust region controller for bounded optimizer updates."""

from __future__ import annotations

import torch

from layer_memory import LayerStateMemory


class TrustRegionController:
    EPS = 1e-8

    def apply(self, delta: torch.Tensor, tau: float) -> torch.Tensor:
        norm = float(delta.norm().item())
        if norm > tau:
            return (tau / (norm + self.EPS)) * delta
        return delta

    def apply_with_info(self, delta: torch.Tensor, tau: float) -> tuple[torch.Tensor, float, bool]:
        pre_clip_norm = float(delta.norm().item())
        clipped = pre_clip_norm > tau
        if clipped:
            return (tau / (pre_clip_norm + self.EPS)) * delta, pre_clip_norm, True
        return delta, pre_clip_norm, False

    def check_instability(
        self,
        loss_t: float,
        loss_prev: float,
        threshold: float = 1.5,
    ) -> bool:
        if loss_prev <= 0:
            return False
        return loss_t > loss_prev * threshold

    def emergency_response(self, memory: LayerStateMemory, all_layer_ids: list[int]) -> None:
        for layer_id in all_layer_ids:
            memory.emergency_shrink(layer_id)
